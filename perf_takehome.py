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
from dataclasses import dataclass
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


@dataclass
class MemBuffer:
    """Double buffer for pipelining loads with computation. Holds 16 elements (2 vectors)."""
    vidx0: int      # scratch address for first 8 indices
    vidx1: int      # scratch address for second 8 indices
    vval0: int      # scratch address for first 8 values
    vval1: int      # scratch address for second 8 values
    vnode_val0: int # scratch address for first 8 node values
    vnode_val1: int # scratch address for second 8 node values


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
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def add_bundle(self, *ops):
        """
        Add a VLIW instruction bundle with multiple operations in parallel.
        Each op is (engine, slot) tuple.
        Example: add_bundle(("alu", ("+", dst, a, b)), ("load", ("vload", dst, addr)))
        """
        bundle = {}
        for engine, slot in ops:
            if engine not in bundle:
                bundle[engine] = []
            bundle[engine].append(slot)
        self.instrs.append(bundle)

    def build_bundles(self, bundles):
        """
        Build instruction list from list of bundles.
        Each bundle is a list of (engine, slot) tuples.
        """
        instrs = []
        for bundle in bundles:
            instr = {}
            for engine, slot in bundle:
                if engine not in instr:
                    instr[engine] = []
                instr[engine].append(slot)
            instrs.append(instr)
        return instrs

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

    def build_hash_vec(self, vval_addr, vtmp1, vtmp2, vconsts):
        """
        Vectorized hash function - processes 8 values at once.

        vval_addr: scratch address of 8 contiguous values to hash (modified in place)
        vtmp1: scratch address for 8-element temp vector
        vtmp2: scratch address for 8-element temp vector
        vconsts: dict mapping constant values to their vector scratch addresses
        """
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            # tmp1 = val op1 const1 (e.g., val + 0x7ED55D16)
            slots.append(("valu", (op1, vtmp1, vval_addr, vconsts[val1])))
            # tmp2 = val op3 const3 (e.g., val << 12)
            slots.append(("valu", (op3, vtmp2, vval_addr, vconsts[val3])))
            # val = tmp1 op2 tmp2 (e.g., tmp1 + tmp2)
            slots.append(("valu", (op2, vval_addr, vtmp1, vtmp2)))

        return slots

    def build_hash_vec_packed(self, vval_addr, vtmp1, vtmp2, vconsts):
        """
        Vectorized hash with VLIW packing - two independent ops per cycle.
        Returns list of bundles (each bundle is list of (engine, slot)).
        """
        bundles = []

        for op1, val1, op2, op3, val3 in HASH_STAGES:
            # PARALLEL: tmp1 and tmp2 calculations are independent (both read vval)
            bundles.append([
                ("valu", (op1, vtmp1, vval_addr, vconsts[val1])),
                ("valu", (op3, vtmp2, vval_addr, vconsts[val3])),
            ])
            # val = tmp1 op2 tmp2 (depends on above)
            bundles.append([("valu", (op2, vval_addr, vtmp1, vtmp2))])

        return bundles

    def emit_load_data(self, buf, base_i_const, base_i_plus_8_const, tmp_addr1, tmp_addr2):
        """
        Generate bundles to load 16 elements (2 vectors) for one iteration into buffer.
        Uses LOAD and ALU engines only.
        Returns list of bundles.
        """
        bundles = []

        # Load first vector (elements 0-7)
        # Compute addresses for vidx0 and vval0
        bundles.append([
            ("alu", ("+", tmp_addr1, self.scratch["inp_indices_p"], base_i_const)),
            ("alu", ("+", tmp_addr2, self.scratch["inp_values_p"], base_i_const)),
        ])
        bundles.append([
            ("load", ("vload", buf.vidx0, tmp_addr1)),
            ("load", ("vload", buf.vval0, tmp_addr2)),
        ])

        # Load second vector (elements 8-15)
        bundles.append([
            ("alu", ("+", tmp_addr1, self.scratch["inp_indices_p"], base_i_plus_8_const)),
            ("alu", ("+", tmp_addr2, self.scratch["inp_values_p"], base_i_plus_8_const)),
        ])
        bundles.append([
            ("load", ("vload", buf.vidx1, tmp_addr1)),
            ("load", ("vload", buf.vval1, tmp_addr2)),
        ])

        # Gather load for vnode_val0 (first 8 elements, 2 at a time)
        for j in range(0, VLEN, 2):
            bundles.append([
                ("alu", ("+", tmp_addr1, self.scratch["forest_values_p"], buf.vidx0 + j)),
                ("alu", ("+", tmp_addr2, self.scratch["forest_values_p"], buf.vidx0 + j + 1)),
            ])
            bundles.append([
                ("load", ("load", buf.vnode_val0 + j, tmp_addr1)),
                ("load", ("load", buf.vnode_val0 + j + 1, tmp_addr2)),
            ])

        # Gather load for vnode_val1 (second 8 elements, 2 at a time)
        for j in range(0, VLEN, 2):
            bundles.append([
                ("alu", ("+", tmp_addr1, self.scratch["forest_values_p"], buf.vidx1 + j)),
                ("alu", ("+", tmp_addr2, self.scratch["forest_values_p"], buf.vidx1 + j + 1)),
            ])
            bundles.append([
                ("load", ("load", buf.vnode_val1 + j, tmp_addr1)),
                ("load", ("load", buf.vnode_val1 + j + 1, tmp_addr2)),
            ])

        return bundles

    def emit_compute(self, buf, vtmp1_0, vtmp1_1, vtmp2_0, vtmp2_1, vtmp3_0, vtmp3_1, vzero, vone, vtwo, vn_nodes, vconsts):
        """
        Generate bundles for hash computation on 16 elements (2 vectors in parallel).
        Uses VALU and FLOW engines. Processes both vectors simultaneously using 4 VALU slots.
        Returns list of bundles.
        """
        bundles = []

        # vval = vval ^ vnode_val (both vectors in parallel)
        bundles.append([
            ("valu", ("^", buf.vval0, buf.vval0, buf.vnode_val0)),
            ("valu", ("^", buf.vval1, buf.vval1, buf.vnode_val1)),
        ])

        # Hash computation for both vectors in parallel (4 VALU slots per cycle)
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            # Both vectors: tmp1 and tmp2 calculations
            bundles.append([
                ("valu", (op1, vtmp1_0, buf.vval0, vconsts[val1])),
                ("valu", (op3, vtmp2_0, buf.vval0, vconsts[val3])),
                ("valu", (op1, vtmp1_1, buf.vval1, vconsts[val1])),
                ("valu", (op3, vtmp2_1, buf.vval1, vconsts[val3])),
            ])
            # Both vectors: combine tmp1 and tmp2
            bundles.append([
                ("valu", (op2, buf.vval0, vtmp1_0, vtmp2_0)),
                ("valu", (op2, buf.vval1, vtmp1_1, vtmp2_1)),
            ])

        # vidx = 2*vidx + (1 if vval % 2 == 0 else 2) - both vectors
        bundles.append([
            ("valu", ("%", vtmp1_0, buf.vval0, vtwo)),
            ("valu", ("*", buf.vidx0, buf.vidx0, vtwo)),
            ("valu", ("%", vtmp1_1, buf.vval1, vtwo)),
            ("valu", ("*", buf.vidx1, buf.vidx1, vtwo)),
        ])
        bundles.append([
            ("valu", ("==", vtmp1_0, vtmp1_0, vzero)),
            ("valu", ("==", vtmp1_1, vtmp1_1, vzero)),
        ])
        # Only 1 FLOW slot per cycle, so do vselects separately
        bundles.append([("flow", ("vselect", vtmp3_0, vtmp1_0, vone, vtwo))])
        bundles.append([("flow", ("vselect", vtmp3_1, vtmp1_1, vone, vtwo))])
        bundles.append([
            ("valu", ("+", buf.vidx0, buf.vidx0, vtmp3_0)),
            ("valu", ("+", buf.vidx1, buf.vidx1, vtmp3_1)),
        ])

        # vidx = 0 if vidx >= n_nodes else vidx - both vectors
        bundles.append([
            ("valu", ("<", vtmp1_0, buf.vidx0, vn_nodes)),
            ("valu", ("<", vtmp1_1, buf.vidx1, vn_nodes)),
        ])
        # Only 1 FLOW slot per cycle
        bundles.append([("flow", ("vselect", buf.vidx0, vtmp1_0, buf.vidx0, vzero))])
        bundles.append([("flow", ("vselect", buf.vidx1, vtmp1_1, buf.vidx1, vzero))])

        return bundles

    def emit_store_data(self, buf, base_i_const, base_i_plus_8_const, tmp_addr1, tmp_addr2):
        """
        Generate bundles to store 16 elements (2 vectors).
        Uses STORE and ALU engines only.
        Returns list of bundles.
        """
        bundles = []

        # Store first vector (elements 0-7)
        bundles.append([
            ("alu", ("+", tmp_addr1, self.scratch["inp_indices_p"], base_i_const)),
            ("alu", ("+", tmp_addr2, self.scratch["inp_values_p"], base_i_const)),
        ])
        bundles.append([
            ("store", ("vstore", tmp_addr1, buf.vidx0)),
            ("store", ("vstore", tmp_addr2, buf.vval0)),
        ])

        # Store second vector (elements 8-15)
        bundles.append([
            ("alu", ("+", tmp_addr1, self.scratch["inp_indices_p"], base_i_plus_8_const)),
            ("alu", ("+", tmp_addr2, self.scratch["inp_values_p"], base_i_plus_8_const)),
        ])
        bundles.append([
            ("store", ("vstore", tmp_addr1, buf.vidx1)),
            ("store", ("vstore", tmp_addr2, buf.vval1)),
        ])

        return bundles

    def merge_bundles(self, compute_bundles, load_bundles):
        """
        Merge compute and load bundles to run in parallel.
        Compute uses VALU/FLOW, Load uses LOAD/ALU - no conflicts.
        """
        merged = []
        max_len = max(len(compute_bundles), len(load_bundles))

        for i in range(max_len):
            bundle = []
            if i < len(compute_bundles):
                bundle.extend(compute_bundles[i])
            if i < len(load_bundles):
                bundle.extend(load_bundles[i])
            merged.append(bundle)

        return merged

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized kernel with VLIW packing and double buffering.
        Processes 16 elements per iteration (2 vectors).
        Overlaps loading next iteration's data with current iteration's computation.
        """
        # Scalar temporaries
        tmp1 = self.alloc_scratch("tmp1")
        tmp_addr1 = self.alloc_scratch("tmp_addr1")
        tmp_addr2 = self.alloc_scratch("tmp_addr2")

        # Scratch space addresses for init vars
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        # Scalar constants
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        eight_const = self.scratch_const(8)

        # Double buffer: two sets of 16-element buffers (2 vectors each)
        bufA = MemBuffer(
            vidx0=self.alloc_scratch("vidx_A0", VLEN),
            vidx1=self.alloc_scratch("vidx_A1", VLEN),
            vval0=self.alloc_scratch("vval_A0", VLEN),
            vval1=self.alloc_scratch("vval_A1", VLEN),
            vnode_val0=self.alloc_scratch("vnode_val_A0", VLEN),
            vnode_val1=self.alloc_scratch("vnode_val_A1", VLEN),
        )
        bufB = MemBuffer(
            vidx0=self.alloc_scratch("vidx_B0", VLEN),
            vidx1=self.alloc_scratch("vidx_B1", VLEN),
            vval0=self.alloc_scratch("vval_B0", VLEN),
            vval1=self.alloc_scratch("vval_B1", VLEN),
            vnode_val0=self.alloc_scratch("vnode_val_B0", VLEN),
            vnode_val1=self.alloc_scratch("vnode_val_B1", VLEN),
        )

        # Vector temporaries - need 2 sets for processing 2 vectors in parallel
        vtmp1_0 = self.alloc_scratch("vtmp1_0", VLEN)
        vtmp1_1 = self.alloc_scratch("vtmp1_1", VLEN)
        vtmp2_0 = self.alloc_scratch("vtmp2_0", VLEN)
        vtmp2_1 = self.alloc_scratch("vtmp2_1", VLEN)
        vtmp3_0 = self.alloc_scratch("vtmp3_0", VLEN)
        vtmp3_1 = self.alloc_scratch("vtmp3_1", VLEN)

        # Vector constants
        vzero = self.alloc_scratch("vzero", VLEN)
        vone = self.alloc_scratch("vone", VLEN)
        vtwo = self.alloc_scratch("vtwo", VLEN)
        vn_nodes = self.alloc_scratch("vn_nodes", VLEN)

        # Broadcast scalar constants to vectors
        self.add("valu", ("vbroadcast", vzero, zero_const))
        self.add("valu", ("vbroadcast", vone, one_const))
        self.add("valu", ("vbroadcast", vtwo, two_const))
        self.add("valu", ("vbroadcast", vn_nodes, self.scratch["n_nodes"]))

        # Hash constants - need vector versions
        hash_consts = set()
        for _, val1, _, _, val3 in HASH_STAGES:
            hash_consts.add(val1)
            hash_consts.add(val3)

        vconsts = {}
        for val in hash_consts:
            scalar_addr = self.scratch_const(val)
            vec_addr = self.alloc_scratch(f"vconst_{val:x}", VLEN)
            self.add("valu", ("vbroadcast", vec_addr, scalar_addr))
            vconsts[val] = vec_addr

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting loop"))

        body = []

        # Each iteration processes 16 elements (2 vectors)
        elements_per_iter = 16
        n_iters_per_round = batch_size // elements_per_iter  # 256/16 = 16 iterations per round
        total_iters = rounds * n_iters_per_round

        def get_iter_info(iter_idx):
            """Get round, iteration within round, and base_i for a given iteration."""
            r = iter_idx // n_iters_per_round
            vi = iter_idx % n_iters_per_round
            base_i = vi * elements_per_iter
            return r, vi, base_i

        # Start with bufA as active (for compute), bufB as loading
        active = bufA
        loading = bufB

        # PROLOGUE: Load first iteration (16 elements) into active buffer
        r0, vi0, base_i0 = get_iter_info(0)
        base_i0_const = self.scratch_const(base_i0)
        base_i0_plus_8_const = self.scratch_const(base_i0 + 8)
        body.append([("debug", ("comment", f"--- PROLOGUE: Load iter 0 (round {r0}, batch {base_i0}-{base_i0+15}) START ---"))])
        body.extend(self.emit_load_data(active, base_i0_const, base_i0_plus_8_const, tmp_addr1, tmp_addr2))
        body.append([("debug", ("comment", f"--- PROLOGUE END ---"))])

        # MAIN LOOP: For iterations 0 to total_iters-2
        # Compute on active buffer while loading next iteration into loading buffer
        for iter_idx in range(total_iters - 1):
            r, vi, base_i = get_iter_info(iter_idx)
            next_r, next_vi, next_base_i = get_iter_info(iter_idx + 1)

            base_i_const = self.scratch_const(base_i)
            base_i_plus_8_const = self.scratch_const(base_i + 8)
            next_base_i_const = self.scratch_const(next_base_i)
            next_base_i_plus_8_const = self.scratch_const(next_base_i + 8)

            body.append([("debug", ("comment", f"--- ITER {iter_idx}: Compute (r{r},batch {base_i}-{base_i+15}) | Load (r{next_r},batch {next_base_i}-{next_base_i+15}) START ---"))])

            # Generate compute ops for active buffer (16 elements)
            compute_ops = self.emit_compute(active, vtmp1_0, vtmp1_1, vtmp2_0, vtmp2_1, vtmp3_0, vtmp3_1, vzero, vone, vtwo, vn_nodes, vconsts)

            # Generate load ops for loading buffer (next iteration, 16 elements)
            load_ops = self.emit_load_data(loading, next_base_i_const, next_base_i_plus_8_const, tmp_addr1, tmp_addr2)

            # Merge them to run in parallel
            body.extend(self.merge_bundles(compute_ops, load_ops))

            # Store results from active buffer (16 elements)
            body.extend(self.emit_store_data(active, base_i_const, base_i_plus_8_const, tmp_addr1, tmp_addr2))

            body.append([("debug", ("comment", f"--- ITER {iter_idx} END ---"))])

            # Swap buffers
            active, loading = loading, active

        # EPILOGUE: Compute last iteration (no more loading needed)
        last_r, last_vi, last_base_i = get_iter_info(total_iters - 1)
        last_base_i_const = self.scratch_const(last_base_i)
        last_base_i_plus_8_const = self.scratch_const(last_base_i + 8)
        body.append([("debug", ("comment", f"--- EPILOGUE: Compute iter {total_iters-1} (round {last_r}, batch {last_base_i}-{last_base_i+15}) START ---"))])
        body.extend(self.emit_compute(active, vtmp1_0, vtmp1_1, vtmp2_0, vtmp2_1, vtmp3_0, vtmp3_1, vzero, vone, vtwo, vn_nodes, vconsts))
        body.extend(self.emit_store_data(active, last_base_i_const, last_base_i_plus_8_const, tmp_addr1, tmp_addr2))
        body.append([("debug", ("comment", f"--- EPILOGUE END ---"))])

        body_instrs = self.build_bundles(body)
        self.instrs.extend(body_instrs)
        self.instrs.append({"flow": [("pause",)]})

        # Write instructions to file for debugging
        self._write_instruction_log(rounds, batch_size)

    def _write_instruction_log(self, rounds, batch_size):
        """Write instruction log with cycle-by-cycle table and stats."""
        import os
        engines = ["load", "valu", "alu", "store", "flow"]
        col_widths = {"load": 55, "valu": 70, "alu": 70, "store": 45, "flow": 35}

        with open(f'{os.getcwd()}/instructions.txt', 'w') as f:
            f.write(f"Total instructions: {len(self.instrs)}\n")
            f.write(f"Rounds: {rounds}, Batch size: {batch_size}\n")
            f.write("=" * 150 + "\n\n")

            # Header
            header = f"{'CYC':<5}|"
            for eng in engines:
                header += f" {eng.upper():<{col_widths[eng]}}|"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            # Track stats - overall
            total_cycles = 0
            total_stats = {eng: 0 for eng in engines}
            total_ops = {eng: 0 for eng in engines}

            # Track stats - per round
            round_cycles = 0
            round_stats = {eng: 0 for eng in engines}
            round_ops = {eng: 0 for eng in engines}

            # Track stats - per batch
            batch_cycles = 0
            batch_stats = {eng: 0 for eng in engines}
            batch_ops = {eng: 0 for eng in engines}

            def write_summary(label, cycles, stats):
                if cycles == 0:
                    return
                f.write(f"  {label}: {cycles} cycles | ")
                parts = []
                for eng in engines:
                    if stats[eng] > 0:
                        pct = (stats[eng] / cycles * 100)
                        parts.append(f"{eng.upper()}:{stats[eng]}({pct:.0f}%)")
                f.write(" ".join(parts) + "\n")

            # Each instruction is one cycle
            for cycle, instr in enumerate(self.instrs):
                # Check for markers in debug
                if "debug" in instr and instr["debug"][0][0] == "comment":
                    marker = str(instr["debug"][0][1])

                    # Round markers (=====)
                    if "=====" in marker:
                        if "END" in marker:
                            f.write(f"\n{marker}\n")
                            write_summary("ROUND SUMMARY", round_cycles, round_stats)
                            f.write("-" * len(header) + "\n\n")
                            # Reset round stats
                            round_cycles = 0
                            round_stats = {eng: 0 for eng in engines}
                            round_ops = {eng: 0 for eng in engines}
                        else:
                            f.write(f"\n{marker}\n")
                            f.write("-" * len(header) + "\n")
                        continue

                    # Batch markers (---)
                    if "---" in marker:
                        if "END" in marker:
                            write_summary("BATCH", batch_cycles, batch_stats)
                            # Reset batch stats
                            batch_cycles = 0
                            batch_stats = {eng: 0 for eng in engines}
                            batch_ops = {eng: 0 for eng in engines}
                        else:
                            f.write(f"\n{marker}\n")
                        continue

                # Skip debug-only instructions
                if list(instr.keys()) == ["debug"]:
                    continue

                # Count this as an actual execution cycle
                total_cycles += 1
                round_cycles += 1
                batch_cycles += 1

                # Count stats
                for eng in engines:
                    if eng in instr:
                        total_stats[eng] += 1
                        total_ops[eng] += len(instr[eng])
                        round_stats[eng] += 1
                        round_ops[eng] += len(instr[eng])
                        batch_stats[eng] += 1
                        batch_ops[eng] += len(instr[eng])

                row = f"{cycle:<5}|"
                for eng in engines:
                    if eng in instr:
                        ops = instr[eng]
                        ops_str = str(ops)
                        if len(ops_str) > col_widths[eng]:
                            ops_str = ops_str[:col_widths[eng]-3] + "..."
                        row += f" {ops_str:<{col_widths[eng]}}|"
                    else:
                        row += f" {'.':<{col_widths[eng]}}|"
                f.write(row + "\n")

            # Write overall summary at end
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"OVERALL SUMMARY: {total_cycles} cycles\n")
            f.write(f"Engine utilization:\n")
            for eng in engines:
                pct = (total_stats[eng] / total_cycles * 100) if total_cycles > 0 else 0
                f.write(f"  {eng.upper():<6}: {total_stats[eng]:4d} cycles active ({pct:5.1f}%), {total_ops[eng]:5d} ops\n")

        print(f"Instructions written to {os.getcwd()}/instructions.txt")

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
