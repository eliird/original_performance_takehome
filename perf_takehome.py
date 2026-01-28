"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        if not vliw:
            # Simple slot packing that just uses one slot per instruction bundle
            instrs = []
            for item in slots:
                # Handle pre-built instruction bundles (dicts)
                if isinstance(item, dict):
                    instrs.append(item)
                else:
                    # Handle tuples: (engine, slot)
                    engine, slot = item
                    instrs.append({engine: [slot]})
            return instrs

        # VLIW mode: pack multiple slots into instruction bundles respecting slot limits
        instrs = []
        current_bundle = {}
        current_counts = {engine: 0 for engine in SLOT_LIMITS}

        for item in slots:
            # Handle pre-built instruction bundles (dicts)
            if isinstance(item, dict):
                # Flush current bundle first
                if current_bundle:
                    instrs.append(current_bundle)
                    current_bundle = {}
                    current_counts = {engine: 0 for engine in SLOT_LIMITS}
                # Add the pre-built bundle as-is
                instrs.append(item)
                continue

            engine, slot = item
            limit = SLOT_LIMITS.get(engine, 1)

            # Check if we can add this slot to the current bundle
            if current_counts[engine] < limit:
                # Add to current bundle
                if engine not in current_bundle:
                    current_bundle[engine] = []
                current_bundle[engine].append(slot)
                current_counts[engine] += 1
            else:
                # Current bundle is full for this engine, start a new bundle
                if current_bundle:
                    instrs.append(current_bundle)
                current_bundle = {engine: [slot]}
                current_counts = {e: 0 for e in SLOT_LIMITS}
                current_counts[engine] = 1

        # Don't forget the last bundle
        if current_bundle:
            instrs.append(current_bundle)

        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_hash_vec_parallel(self, v_val, v_tmp1, hash_vecs, n_active):
        """
        Optimized vector hash using multiply_add for compatible stages.

        Stages 0,2,4: (a + c1) + (a << n) = a * (1 + 2^n) + c1 -> multiply_add
        Stages 1,3,5: (a ^ c1) ^ (a >> n) or (a + c1) ^ (a << n) -> 3 ops

        hash_vecs contains precomputed broadcast vectors for all constants.
        """
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1_vec, c2_vec = hash_vecs[hi]  # c2 is either multiplier or shift amount

            if op1 == "+" and op2 == "+" and op3 == "<<":
                # Can use multiply_add: val = val * (1 + 2^shift) + c1
                # c2_vec contains (1 + 2^val3) as multiplier
                slots.append({"valu": [("multiply_add", v_val[p], v_val[p], c2_vec, c1_vec)
                                       for p in range(n_active)]})
            else:
                # General case: tmp1 = op1(val, c1), tmp2 = op3(val, c2), val = op2(tmp1, tmp2)
                slots.append({"valu": [(op1, v_tmp1[p], v_val[p], c1_vec) for p in range(n_active)]})
                slots.append({"valu": [(op3, v_val[p], v_val[p], c2_vec) for p in range(n_active)]})
                slots.append({"valu": [(op2, v_val[p], v_tmp1[p], v_val[p]) for p in range(n_active)]})
        return slots

    def build_gather(self, v_idx, v_node_val, tmp_addrs, n_active, pending_stores=None):
        """
        Gather node values: node_val[i] = mem[forest_values_p + idx[i]] for each element in vectors.
        Uses scalar loads since we need non-contiguous memory access.
        Uses double-buffered tmp_addrs to pipeline address computation with loads.
        """
        slots = []
        tmp_addrs_0, tmp_addrs_1 = tmp_addrs[0], tmp_addrs[1]
        store_idx = 0
        stores_to_do = []
        if pending_stores:
            prev_n, prev_idx_addrs, prev_val_addrs, pv_idx, pv_val = pending_stores
            for p in range(prev_n):
                stores_to_do.append([("vstore", prev_idx_addrs[p], pv_idx[p]),
                                     ("vstore", prev_val_addrs[p], pv_val[p])])

        # Compute addresses for vi=0 into buffer 0
        cur_buf = tmp_addrs_0
        next_buf = tmp_addrs_1

        alu_bundle = {"alu": [("+", cur_buf[p], self.scratch["forest_values_p"], v_idx[p] + 0)
                              for p in range(n_active)]}
        if store_idx < len(stores_to_do):
            alu_bundle["store"] = stores_to_do[store_idx]
            store_idx += 1
        slots.append(alu_bundle)

        for vi in range(VLEN):
            # Load from cur_buf, compute next addresses into next_buf
            for load_cycle, p in enumerate(range(0, n_active, 2)):
                load_slots = [("load", v_node_val[p] + vi, cur_buf[p])]
                if p + 1 < n_active:
                    load_slots.append(("load", v_node_val[p + 1] + vi, cur_buf[p + 1]))
                load_bundle = {"load": load_slots}

                # On first load cycle, compute addresses for vi+1 into next_buf
                if load_cycle == 0 and vi + 1 < VLEN:
                    load_bundle["alu"] = [("+", next_buf[q], self.scratch["forest_values_p"], v_idx[q] + vi + 1)
                                          for q in range(n_active)]

                if store_idx < len(stores_to_do):
                    load_bundle["store"] = stores_to_do[store_idx]
                    store_idx += 1
                slots.append(load_bundle)

            # Swap buffers
            cur_buf, next_buf = next_buf, cur_buf

        # Finish any remaining stores
        for i in range(store_idx, len(stores_to_do)):
            slots.append({"store": stores_to_do[i]})

        return slots

    def build_gather_with_compute(self, v_idx_cur, v_node_val_cur, tmp_addrs, n_active_cur,
                                   v_idx_prev, v_val_prev, v_tmp1, hash_vecs,
                                   v_zero, v_two, v_n_nodes, n_active_prev,
                                   pending_stores=None,
                                   next_vloads=None):
        """
        Interleave gather (LOAD+ALU) with hash + index update (VALU) and stores.
        Key optimization: Double-buffered tmp_addrs so we compute addresses for vi+1
        into buffer B while loading from buffer A. No idle ALU cycles!
        """
        slots = []
        tmp_addrs_0, tmp_addrs_1 = tmp_addrs[0], tmp_addrs[1]  # Double buffered

        # Get all VALU operations: hash (12 ops) + index update (6 ops) = 18 ops
        valu_ops = []
        if n_active_prev > 0:
            hash_ops = self.build_hash_vec_parallel(v_val_prev, v_tmp1, hash_vecs, n_active_prev)
            for h in hash_ops:
                if isinstance(h, dict) and "valu" in h:
                    valu_ops.append(h["valu"])

            valu_ops.append([("%", v_tmp1[p], v_val_prev[p], v_two) for p in range(n_active_prev)])
            valu_ops.append([("==", v_tmp1[p], v_tmp1[p], v_zero) for p in range(n_active_prev)])
            valu_ops.append([("-", v_tmp1[p], v_two, v_tmp1[p]) for p in range(n_active_prev)])
            valu_ops.append([("multiply_add", v_idx_prev[p], v_idx_prev[p], v_two, v_tmp1[p]) for p in range(n_active_prev)])
            valu_ops.append([("<", v_tmp1[p], v_idx_prev[p], v_n_nodes) for p in range(n_active_prev)])
            valu_ops.append([("*", v_idx_prev[p], v_idx_prev[p], v_tmp1[p]) for p in range(n_active_prev)])

        valu_idx = 0

        # Prepare stores
        store_idx = 0
        stores_to_do = []
        if pending_stores:
            prev_n, prev_idx_addrs, prev_val_addrs, pv_idx, pv_val = pending_stores
            for p in range(prev_n):
                stores_to_do.append([("vstore", prev_idx_addrs[p], pv_idx[p]),
                                     ("vstore", prev_val_addrs[p], pv_val[p])])

        # Prepare next vloads
        vload_idx = 0
        vloads_to_do = []
        if next_vloads:
            next_n, next_v_idx, next_v_val, next_idx_addrs, next_val_addrs = next_vloads
            for p in range(next_n):
                vloads_to_do.append([("vload", next_v_idx[p], next_idx_addrs[p]),
                                     ("vload", next_v_val[p], next_val_addrs[p])])

        # === Pipelined gather with double-buffered addresses ===
        # Compute addresses for vi=0 into buffer 0
        cur_buf = tmp_addrs_0
        next_buf = tmp_addrs_1

        bundle = {"alu": [("+", cur_buf[p], self.scratch["forest_values_p"], v_idx_cur[p] + 0)
                          for p in range(n_active_cur)]}
        if valu_idx < len(valu_ops):
            bundle["valu"] = valu_ops[valu_idx]
            valu_idx += 1
        if store_idx < len(stores_to_do):
            bundle["store"] = stores_to_do[store_idx]
            store_idx += 1
        if vload_idx < len(vloads_to_do):
            bundle["load"] = vloads_to_do[vload_idx]
            vload_idx += 1
        slots.append(bundle)

        for vi in range(VLEN):
            # Load from cur_buf, compute next addresses into next_buf
            for load_cycle, p in enumerate(range(0, n_active_cur, 2)):
                load_slots = [("load", v_node_val_cur[p] + vi, cur_buf[p])]
                if p + 1 < n_active_cur:
                    load_slots.append(("load", v_node_val_cur[p + 1] + vi, cur_buf[p + 1]))

                load_bundle = {"load": load_slots}

                # On first load cycle of this vi, compute addresses for vi+1 into next_buf
                if load_cycle == 0 and vi + 1 < VLEN:
                    load_bundle["alu"] = [("+", next_buf[q], self.scratch["forest_values_p"], v_idx_cur[q] + vi + 1)
                                          for q in range(n_active_cur)]

                # Add VALU
                if valu_idx < len(valu_ops):
                    load_bundle["valu"] = valu_ops[valu_idx]
                    valu_idx += 1

                # Add store
                if store_idx < len(stores_to_do):
                    load_bundle["store"] = stores_to_do[store_idx]
                    store_idx += 1

                slots.append(load_bundle)

            # Swap buffers for next vi
            cur_buf, next_buf = next_buf, cur_buf

        # Finish remaining VALU/stores/vloads
        while valu_idx < len(valu_ops) or store_idx < len(stores_to_do) or vload_idx < len(vloads_to_do):
            bundle = {}
            if valu_idx < len(valu_ops):
                bundle["valu"] = valu_ops[valu_idx]
                valu_idx += 1
            if store_idx < len(stores_to_do):
                bundle["store"] = stores_to_do[store_idx]
                store_idx += 1
            if vload_idx < len(vloads_to_do):
                bundle["load"] = vloads_to_do[vload_idx]
                vload_idx += 1
            if bundle:
                slots.append(bundle)

        return slots

    def build_index_update(self, v_idx, v_val, v_tmp1, v_zero, v_two, v_n_nodes, n_active):
        """
        Compute next index: idx = 2*idx + (1 if val%2==0 else 2), then wrap if >= n_nodes.
        Uses arithmetic instead of vselect to avoid flow slot bottleneck.
        """
        slots = []
        # tmp1 = (val % 2 == 0), gives 1 if even, 0 if odd
        slots.append({"valu": [("%", v_tmp1[p], v_val[p], v_two) for p in range(n_active)]})
        slots.append({"valu": [("==", v_tmp1[p], v_tmp1[p], v_zero) for p in range(n_active)]})
        # tmp1 = 2 - tmp1: gives 1 if even (2-1=1), 2 if odd (2-0=2)
        slots.append({"valu": [("-", v_tmp1[p], v_two, v_tmp1[p]) for p in range(n_active)]})
        # idx = idx * 2 + tmp1 using multiply_add
        slots.append({"valu": [("multiply_add", v_idx[p], v_idx[p], v_two, v_tmp1[p]) for p in range(n_active)]})
        # Wrap: idx = idx * (idx < n_nodes)
        slots.append({"valu": [("<", v_tmp1[p], v_idx[p], v_n_nodes) for p in range(n_active)]})
        slots.append({"valu": [("*", v_idx[p], v_idx[p], v_tmp1[p]) for p in range(n_active)]})
        return slots

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        """
        Vectorized kernel processing N_PARALLEL * VLEN (6 * 8 = 48) elements per iteration.
        Uses software pipelining to overlap stores with next iteration's loads.
        """
        N_PARALLEL = 6  # Matches VALU slot limit

        # === Initialization ===
        tmp1 = self.alloc_scratch("tmp1")
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        two_const = self.scratch_const(2)

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting loop"))

        # === Allocate vector registers (triple-buffered for deep pipeline overlap) ===
        v_idx = [[self.alloc_scratch(f"v_idx_{s}_{p}", VLEN) for p in range(N_PARALLEL)] for s in range(3)]
        v_val = [[self.alloc_scratch(f"v_val_{s}_{p}", VLEN) for p in range(N_PARALLEL)] for s in range(3)]
        # Double-buffered node values for gather output overlap
        v_node_val = [[self.alloc_scratch(f"v_node_val_{s}_{p}", VLEN) for p in range(N_PARALLEL)] for s in range(2)]
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_{p}", VLEN) for p in range(N_PARALLEL)]

        # Vector constants
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        # Scalar addresses for gather - double buffered for pipelined address computation
        tmp_addrs = [[self.alloc_scratch(f"tmp_addr_{s}_{p}") for p in range(N_PARALLEL)] for s in range(2)]
        # Triple-buffered base addresses
        idx_addrs = [[self.alloc_scratch(f"idx_addr_{s}_{p}") for p in range(N_PARALLEL)] for s in range(3)]
        val_addrs = [[self.alloc_scratch(f"val_addr_{s}_{p}") for p in range(N_PARALLEL)] for s in range(3)]

        # Hash constant vectors - precompute all constants
        # For multiply_add stages (0,2,4): c1=constant, c2=multiplier (1 + 2^shift)
        # For XOR stages (1,3,5): c1=constant, c2=shift amount
        hash_vecs = []
        for hi in range(len(HASH_STAGES)):
            c1_vec = self.alloc_scratch(f"hash_c1_{hi}", VLEN)
            c2_vec = self.alloc_scratch(f"hash_c2_{hi}", VLEN)
            hash_vecs.append((c1_vec, c2_vec))

        # === Main loop body ===
        body = []

        # Broadcast basic constants
        body.append({"valu": [("vbroadcast", v_zero, zero_const), ("vbroadcast", v_two, two_const),
                              ("vbroadcast", v_n_nodes, self.scratch["n_nodes"])]})

        # Precompute and broadcast all hash constants (done once, not per iteration)
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1_vec, c2_vec = hash_vecs[hi]
            c1_const = self.scratch_const(val1)
            if op1 == "+" and op2 == "+" and op3 == "<<":
                # multiply_add: multiplier = 1 + 2^shift
                c2_const = self.scratch_const(1 + (1 << val3))
            else:
                # shift amount for >> or <<
                c2_const = self.scratch_const(val3)
            body.append({"valu": [("vbroadcast", c1_vec, c1_const), ("vbroadcast", c2_vec, c2_const)]})

        stride = N_PARALLEL * VLEN
        assert batch_size % VLEN == 0, f"batch_size must be divisible by VLEN ({VLEN})"

        # Initialize instruction dumper for analysis
        from dump_instructions import InstructionDumper
        dumper = InstructionDumper("instructions.txt")

        # Build list of all iterations
        iterations = []
        for rnd in range(rounds):
            for base_i in range(0, batch_size, stride):
                remaining = batch_size - base_i
                n_active = min(N_PARALLEL, (remaining + VLEN - 1) // VLEN)
                iterations.append((rnd, base_i, n_active))

        # === Deep 3-batch pipeline ===
        # Pipeline stages (for iteration i in steady state):
        #   - STORE[i-2]: Store results from 2 iterations ago
        #   - GATHER[i] + HASH[i-1]: Current gather overlapped with previous hash
        #   - INDEX[i-1]: Index update for previous iteration
        #   - VLOAD[i+1]: Prefetch next iteration's data

        # State tracking for 3 batches in flight
        # Each entry: (n_active, v_idx, v_val, idx_addrs, val_addrs, v_node_val_set, xor_done)
        pipeline = [None, None, None]  # [i-2 (store), i-1 (compute), i (load/gather)]
        use_set = 0  # Cycles through 0, 1, 2 for triple buffering
        node_val_set = 0  # Cycles through 0, 1 for v_node_val double buffering

        for iter_idx, (rnd, base_i, n_active) in enumerate(iterations):
            iter_body = []

            # Current iteration uses this buffer set
            cur_set = use_set
            cur_v_idx, cur_v_val = v_idx[cur_set], v_val[cur_set]
            cur_idx_addrs, cur_val_addrs = idx_addrs[cur_set], val_addrs[cur_set]
            cur_node_val = v_node_val[node_val_set]

            # Previous iteration (for hash computation) - if exists
            prev_state = pipeline[1]  # i-1 slot
            prev_prev_state = pipeline[0]  # i-2 slot (for stores)

            # === Phase 0: Compute addresses and load idx/val for current iteration ===
            alu_slots = []
            for p in range(n_active):
                offset_const = self.scratch_const(base_i + p * VLEN)
                alu_slots.append(("+", cur_idx_addrs[p], self.scratch["inp_indices_p"], offset_const))
                alu_slots.append(("+", cur_val_addrs[p], self.scratch["inp_values_p"], offset_const))
            iter_body.append({"alu": alu_slots})

            # Load idx/val vectors (overlapped with stores from i-2 if available)
            for p in range(n_active):
                bundle = {"load": [("vload", cur_v_idx[p], cur_idx_addrs[p]),
                                   ("vload", cur_v_val[p], cur_val_addrs[p])]}
                if prev_prev_state and p < prev_prev_state[0]:
                    pp_n, pp_v_idx, pp_v_val, pp_idx_addrs, pp_val_addrs, _, _ = prev_prev_state
                    bundle["store"] = [("vstore", pp_idx_addrs[p], pp_v_idx[p]),
                                       ("vstore", pp_val_addrs[p], pp_v_val[p])]
                iter_body.append(bundle)

            # Finish remaining stores from i-2
            if prev_prev_state:
                pp_n, pp_v_idx, pp_v_val, pp_idx_addrs, pp_val_addrs, _, _ = prev_prev_state
                for p in range(n_active, pp_n):
                    iter_body.append({"store": [("vstore", pp_idx_addrs[p], pp_v_idx[p]),
                                                ("vstore", pp_val_addrs[p], pp_v_val[p])]})

            # === Phase 1: Gather current + Hash+Index previous (THE KEY OPTIMIZATION) ===
            if prev_state:
                # We have a previous iteration to compute while gathering
                pv_n, pv_v_idx, pv_v_val, pv_idx_addrs, pv_val_addrs, pv_node_val_set, pv_xor_done = prev_state
                pv_node_val = v_node_val[pv_node_val_set]

                # First, XOR previous iteration's values with gathered node values
                if not pv_xor_done:
                    iter_body.append({"valu": [("^", pv_v_val[p], pv_v_val[p], pv_node_val[p]) for p in range(pv_n)]})

                # Now do gather + hash + index_update in parallel (no extra vloads yet)
                iter_body.extend(self.build_gather_with_compute(
                    cur_v_idx, cur_node_val, tmp_addrs, n_active,
                    pv_v_idx, pv_v_val, v_tmp1, hash_vecs,
                    v_zero, v_two, v_n_nodes, pv_n,
                    None,  # No pending stores - already handled
                    None   # No vloads interleaving for now
                ))
            else:
                # No previous iteration - just gather (first iteration)
                iter_body.extend(self.build_gather(cur_v_idx, cur_node_val, tmp_addrs, n_active))

            # Update pipeline state
            pipeline[0] = pipeline[1]  # Old i-1 becomes i-2 (ready for store)
            pipeline[1] = (n_active, cur_v_idx, cur_v_val, cur_idx_addrs, cur_val_addrs, node_val_set, False)

            # Advance buffer indices
            use_set = (use_set + 1) % 3
            node_val_set = 1 - node_val_set

            # Add to body and dump
            body.extend(iter_body)
            dumper.add_instructions(iter_body)
            dumper.end_iteration(base_i)

            # End round tracking
            if iter_idx + 1 < len(iterations) and iterations[iter_idx + 1][0] != rnd:
                dumper.end_round()

        # === Epilogue: Drain the pipeline ===
        # After all iterations, we still have up to 2 batches in flight

        # Process pipeline[1] (last iteration - needs XOR, hash, index update, then store)
        if pipeline[1]:
            epilogue_body = []
            pv_n, pv_v_idx, pv_v_val, pv_idx_addrs, pv_val_addrs, pv_node_val_set, pv_xor_done = pipeline[1]
            pv_node_val = v_node_val[pv_node_val_set]

            # XOR with gathered node values
            if not pv_xor_done:
                epilogue_body.append({"valu": [("^", pv_v_val[p], pv_v_val[p], pv_node_val[p]) for p in range(pv_n)]})

            # Hash computation (no gather to overlap with)
            epilogue_body.extend(self.build_hash_vec_parallel(pv_v_val, v_tmp1, hash_vecs, pv_n))

            # Index update
            epilogue_body.extend(self.build_index_update(pv_v_idx, pv_v_val, v_tmp1, v_zero, v_two, v_n_nodes, pv_n))

            # Store results (overlapped with stores from pipeline[0] if exists)
            if pipeline[0]:
                pp_n, pp_v_idx, pp_v_val, pp_idx_addrs, pp_val_addrs, _, _ = pipeline[0]
                # Store both batches
                for p in range(max(pv_n, pp_n)):
                    store_slots = []
                    if p < pp_n:
                        store_slots.extend([("vstore", pp_idx_addrs[p], pp_v_idx[p]),
                                            ("vstore", pp_val_addrs[p], pp_v_val[p])])
                    if p < pv_n:
                        if len(store_slots) < 2:
                            store_slots.extend([("vstore", pv_idx_addrs[p], pv_v_idx[p]),
                                                ("vstore", pv_val_addrs[p], pv_v_val[p])])
                        else:
                            epilogue_body.append({"store": store_slots})
                            store_slots = [("vstore", pv_idx_addrs[p], pv_v_idx[p]),
                                           ("vstore", pv_val_addrs[p], pv_v_val[p])]
                    if store_slots:
                        epilogue_body.append({"store": store_slots})
            else:
                # Just store pipeline[1]
                for p in range(pv_n):
                    epilogue_body.append({"store": [("vstore", pv_idx_addrs[p], pv_v_idx[p]),
                                                    ("vstore", pv_val_addrs[p], pv_v_val[p])]})

            body.extend(epilogue_body)
            dumper.add_instructions(epilogue_body)

        elif pipeline[0]:
            # Only pipeline[0] left to store
            epilogue_body = []
            pp_n, pp_v_idx, pp_v_val, pp_idx_addrs, pp_val_addrs, _, _ = pipeline[0]
            for p in range(pp_n):
                epilogue_body.append({"store": [("vstore", pp_idx_addrs[p], pp_v_idx[p]),
                                                ("vstore", pp_val_addrs[p], pp_v_val[p])]})
            body.extend(epilogue_body)
            dumper.add_instructions(epilogue_body)

        # End final round
        if iterations:
            dumper.end_round()

        dumper.finalize()

        self.instrs.extend(self.build(body))
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
