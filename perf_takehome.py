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

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized kernel with VLIW packing - processes 8 batch items per iteration.
        """
        # Scalar temporaries - need two addr registers for parallel loads
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

        # Vector scratch registers (8 elements each)
        vidx = self.alloc_scratch("vidx", VLEN)
        vval = self.alloc_scratch("vval", VLEN)
        vnode_val = self.alloc_scratch("vnode_val", VLEN)
        vtmp1 = self.alloc_scratch("vtmp1", VLEN)
        vtmp2 = self.alloc_scratch("vtmp2", VLEN)
        vtmp3 = self.alloc_scratch("vtmp3", VLEN)

        # Vector constants (broadcast scalar to vector)
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

        body = []  # Now a list of bundles, each bundle is a list of (engine, slot)

        # Process batch_size/VLEN groups of 8 elements
        n_vectors = batch_size // VLEN
        for round in range(rounds):
            body.append([("debug", ("comment", f"===== ROUND {round} START ====="))])
            for vi in range(n_vectors):
                base_i = vi * VLEN
                body.append([("debug", ("comment", f"--- ROUND {round}, VECTOR {vi} (batch {base_i}-{base_i+7}) ---"))])

                base_i_const = self.scratch_const(base_i)

                # PARALLEL: Compute both addresses in one cycle
                body.append([
                    ("alu", ("+", tmp_addr1, self.scratch["inp_indices_p"], base_i_const)),
                    ("alu", ("+", tmp_addr2, self.scratch["inp_values_p"], base_i_const)),
                ])

                # PARALLEL: Load both vidx and vval in one cycle (2 LOAD slots)
                body.append([
                    ("load", ("vload", vidx, tmp_addr1)),
                    ("load", ("vload", vval, tmp_addr2)),
                ])

                # Gather load: vnode_val[j] = mem[forest_values_p + vidx[j]] for j in 0..7
                # We have 2 LOAD slots and 12 ALU slots, so we can do 2 at a time
                for j in range(0, VLEN, 2):
                    body.append([
                        ("alu", ("+", tmp_addr1, self.scratch["forest_values_p"], vidx + j)),
                        ("alu", ("+", tmp_addr2, self.scratch["forest_values_p"], vidx + j + 1)),
                    ])
                    body.append([
                        ("load", ("load", vnode_val + j, tmp_addr1)),
                        ("load", ("load", vnode_val + j + 1, tmp_addr2)),
                    ])

                # vval = vval ^ vnode_val
                body.append([("valu", ("^", vval, vval, vnode_val))])

                # vval = hash(vval) - pack the two independent ops per stage
                body.extend(self.build_hash_vec_packed(vval, vtmp1, vtmp2, vconsts))

                # vidx = 2*vidx + (1 if vval % 2 == 0 else 2)
                # PARALLEL: % and * are independent (both read from different sources)
                body.append([
                    ("valu", ("%", vtmp1, vval, vtwo)),
                    ("valu", ("*", vidx, vidx, vtwo)),
                ])
                body.append([("valu", ("==", vtmp1, vtmp1, vzero))])
                body.append([("flow", ("vselect", vtmp3, vtmp1, vone, vtwo))])
                body.append([("valu", ("+", vidx, vidx, vtmp3))])

                # vidx = 0 if vidx >= n_nodes else vidx
                body.append([("valu", ("<", vtmp1, vidx, vn_nodes))])
                body.append([("flow", ("vselect", vidx, vtmp1, vidx, vzero))])

                # PARALLEL: Compute both store addresses
                body.append([
                    ("alu", ("+", tmp_addr1, self.scratch["inp_indices_p"], base_i_const)),
                    ("alu", ("+", tmp_addr2, self.scratch["inp_values_p"], base_i_const)),
                ])

                # PARALLEL: Store both vidx and vval (2 STORE slots)
                body.append([
                    ("store", ("vstore", tmp_addr1, vidx)),
                    ("store", ("vstore", tmp_addr2, vval)),
                ])

                body.append([("debug", ("comment", f"--- ROUND {round}, VECTOR {vi} END ---"))])

            body.append([("debug", ("comment", f"===== ROUND {round} END ====="))])

        body_instrs = self.build_bundles(body)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

        # Write instructions to file for debugging - table format by cycle
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
