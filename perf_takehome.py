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

    def build_gather(self, v_idx, v_node_val, tmp_addrs, n_active):
        """
        Gather node values: node_val[i] = mem[forest_values_p + idx[i]] for each element in vectors.
        Uses scalar loads since we need non-contiguous memory access.
        """
        slots = []
        for vi in range(VLEN):
            # Compute addresses for all streams in parallel (up to 12 ALU slots)
            slots.append({"alu": [("+", tmp_addrs[p], self.scratch["forest_values_p"], v_idx[p] + vi)
                                  for p in range(n_active)]})
            # Load 2 at a time (2 load slots available)
            for p in range(0, n_active, 2):
                load_slots = [("load", v_node_val[p] + vi, tmp_addrs[p])]
                if p + 1 < n_active:
                    load_slots.append(("load", v_node_val[p + 1] + vi, tmp_addrs[p + 1]))
                slots.append({"load": load_slots})
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

        # === Allocate vector registers (double-buffered for load/compute overlap) ===
        v_idx = [[self.alloc_scratch(f"v_idx_{s}_{p}", VLEN) for p in range(N_PARALLEL)] for s in range(2)]
        v_val = [[self.alloc_scratch(f"v_val_{s}_{p}", VLEN) for p in range(N_PARALLEL)] for s in range(2)]
        v_node_val = [self.alloc_scratch(f"v_node_val_{p}", VLEN) for p in range(N_PARALLEL)]
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_{p}", VLEN) for p in range(N_PARALLEL)]

        # Vector constants
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        # Scalar addresses for gather and double-buffered base addresses
        tmp_addrs = [self.alloc_scratch(f"tmp_addr_{p}") for p in range(N_PARALLEL)]
        idx_addrs = [[self.alloc_scratch(f"idx_addr_{s}_{p}") for p in range(N_PARALLEL)] for s in range(2)]
        val_addrs = [[self.alloc_scratch(f"val_addr_{s}_{p}") for p in range(N_PARALLEL)] for s in range(2)]

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

        pending_stores = None
        use_set = 0
        data_prefetched = False  # Track if current iteration's data was already loaded

        for iter_idx, (rnd, base_i, n_active) in enumerate(iterations):
            cur_v_idx, cur_v_val = v_idx[use_set], v_val[use_set]
            cur_idx_addrs, cur_val_addrs = idx_addrs[use_set], val_addrs[use_set]
            next_set = 1 - use_set
            iter_body = []

            if not data_prefetched:
                # First iteration or data wasn't prefetched - need to load
                # Compute base addresses for current iteration
                alu_slots = []
                for p in range(n_active):
                    offset_const = self.scratch_const(base_i + p * VLEN)
                    alu_slots.append(("+", cur_idx_addrs[p], self.scratch["inp_indices_p"], offset_const))
                    alu_slots.append(("+", cur_val_addrs[p], self.scratch["inp_values_p"], offset_const))
                iter_body.append({"alu": alu_slots})

                # Load current + store previous (software pipelining)
                for p in range(n_active):
                    bundle = {"load": [("vload", cur_v_idx[p], cur_idx_addrs[p]),
                                       ("vload", cur_v_val[p], cur_val_addrs[p])]}
                    if pending_stores and p < pending_stores[0]:
                        prev_n, prev_idx_addrs, prev_val_addrs, pv_idx, pv_val = pending_stores
                        bundle["store"] = [("vstore", prev_idx_addrs[p], pv_idx[p]),
                                           ("vstore", prev_val_addrs[p], pv_val[p])]
                    iter_body.append(bundle)

                # Finish remaining pending stores
                if pending_stores:
                    prev_n, prev_idx_addrs, prev_val_addrs, pv_idx, pv_val = pending_stores
                    for p in range(n_active, prev_n):
                        iter_body.append({"store": [("vstore", prev_idx_addrs[p], pv_idx[p]),
                                                    ("vstore", prev_val_addrs[p], pv_val[p])]})
                    pending_stores = None
            else:
                # Data was prefetched - just do stores from previous iteration
                if pending_stores:
                    prev_n, prev_idx_addrs, prev_val_addrs, pv_idx, pv_val = pending_stores
                    for p in range(prev_n):
                        iter_body.append({"store": [("vstore", prev_idx_addrs[p], pv_idx[p]),
                                                    ("vstore", prev_val_addrs[p], pv_val[p])]})
                    pending_stores = None

            # Gather node values
            iter_body.extend(self.build_gather(cur_v_idx, v_node_val, tmp_addrs, n_active))

            # Check if there's a next iteration to prefetch
            has_next = iter_idx + 1 < len(iterations)
            if has_next:
                next_rnd, next_base_i, next_n_active = iterations[iter_idx + 1]
                next_v_idx, next_v_val = v_idx[next_set], v_val[next_set]
                next_idx_addrs, next_val_addrs = idx_addrs[next_set], val_addrs[next_set]

                # Compute next iteration's base addresses
                next_alu_slots = []
                for p in range(next_n_active):
                    offset_const = self.scratch_const(next_base_i + p * VLEN)
                    next_alu_slots.append(("+", next_idx_addrs[p], self.scratch["inp_indices_p"], offset_const))
                    next_alu_slots.append(("+", next_val_addrs[p], self.scratch["inp_values_p"], offset_const))

            # XOR with node values + compute next addresses
            xor_bundle = {"valu": [("^", cur_v_val[p], cur_v_val[p], v_node_val[p]) for p in range(n_active)]}
            if has_next:
                xor_bundle["alu"] = next_alu_slots
            iter_body.append(xor_bundle)

            # Hash computation - interleave with next iteration's vloads
            hash_slots = self.build_hash_vec_parallel(cur_v_val, v_tmp1, hash_vecs, n_active)

            if has_next:
                # Merge vloads into hash slots (valu and load can run in parallel)
                load_idx = 0
                for slot in hash_slots:
                    if load_idx < next_n_active and isinstance(slot, dict) and "valu" in slot:
                        new_slot = dict(slot)
                        new_slot["load"] = [("vload", next_v_idx[load_idx], next_idx_addrs[load_idx]),
                                            ("vload", next_v_val[load_idx], next_val_addrs[load_idx])]
                        load_idx += 1
                        iter_body.append(new_slot)
                    else:
                        iter_body.append(slot)
                # Finish any remaining loads
                for p in range(load_idx, next_n_active):
                    iter_body.append({"load": [("vload", next_v_idx[p], next_idx_addrs[p]),
                                               ("vload", next_v_val[p], next_val_addrs[p])]})
                data_prefetched = True  # Next iteration's data is now loaded
            else:
                iter_body.extend(hash_slots)
                data_prefetched = False  # No prefetch happened

            # Update indices
            iter_body.extend(self.build_index_update(cur_v_idx, cur_v_val, v_tmp1, v_zero, v_two, v_n_nodes, n_active))

            # Queue stores for next cycle
            pending_stores = (n_active, cur_idx_addrs, cur_val_addrs, cur_v_idx, cur_v_val)
            use_set = next_set

            # Add to body and dump
            body.extend(iter_body)
            dumper.add_instructions(iter_body)
            dumper.end_iteration(base_i)

            # End round tracking
            if iter_idx + 1 < len(iterations) and iterations[iter_idx + 1][0] != rnd:
                dumper.end_round()

        # End final round
        if iterations:
            dumper.end_round()

        # Flush final stores
        if pending_stores:
            prev_n, prev_idx_addrs, prev_val_addrs, pv_idx, pv_val = pending_stores
            flush_body = []
            for p in range(prev_n):
                flush_body.append({"store": [("vstore", prev_idx_addrs[p], pv_idx[p]),
                                             ("vstore", prev_val_addrs[p], pv_val[p])]})
            body.extend(flush_body)
            dumper.add_instructions(flush_body)

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
