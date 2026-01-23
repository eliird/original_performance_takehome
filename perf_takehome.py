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
        """Build hash with parallelized independent ALU operations"""
        instrs = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            # Bundle two independent ALU operations in one cycle
            instrs.append({"alu": [
                (op1, tmp1, val_hash_addr, self.scratch_const(val1)),
                (op3, tmp2, val_hash_addr, self.scratch_const(val3))
            ]})
            instrs.append({"alu": [(op2, val_hash_addr, tmp1, tmp2)]})
            instrs.append({"debug": [("compare", val_hash_addr, (round, i, "hash_stage", hi))]})

        return instrs

    def build_hash_vectorized(self, vec_val, vec_tmp1, vec_tmp2, hash_const_regs, round, base_i):
        """Vectorized hash - processes VLEN elements in parallel
        hash_const_regs is a list of:
        - For stages 0,2,4: (const_vector, multiplier_vector) for multiply_add
        - For stages 1,3,5: (reg1, reg2) tuples containing pre-broadcast constants

        Uses multiply_add optimization for stages that match pattern: (val + const) + (val << shift)
        - Stage 0: (val + const) + (val << 12) = val*4097 + const
        - Stage 2: (val + const) + (val << 5) = val*33 + const
        - Stage 4: (val + const) + (val << 3) = val*9 + const
        """
        instrs = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const_reg1, const_reg2 = hash_const_regs[hi]

            # Check if this stage can use multiply_add (stages 0, 2, 4)
            # Pattern: (val + const) + (val << shift) where both ops are '+'
            if hi in [0, 2, 4] and op1 == "+" and op2 == "+" and op3 == "<<":
                # For multiply_add stages, const_reg1 = constant, const_reg2 = multiplier
                # Use multiply_add: val = val * multiplier + const (1 cycle!)
                instrs.append({"valu": [("multiply_add", vec_val, vec_val,
                                        const_reg2, const_reg1)]})
            else:
                # Standard 2-cycle hash stage
                instrs.append({"valu": [
                    (op1, vec_tmp1, vec_val, const_reg1),
                    (op3, vec_tmp2, vec_val, const_reg2)
                ]})
                instrs.append({"valu": [(op2, vec_val, vec_tmp1, vec_tmp2)]})

            # Debug compare for each element
            for lane in range(VLEN):
                instrs.append({"debug": [("compare", vec_val + lane, (round, base_i + lane, "hash_stage", hi))]})

        return instrs

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Main kernel entry point - uses vectorized implementation for best performance.
        """
        return self.build_kernel_vectorized(forest_height, n_nodes, batch_size, rounds)

    def build_kernel_scalar(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Scalar implementation - kept for reference but not used.
        The vectorized version is used by default for better performance.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
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

        # Parallelized initialization using 2 load slots per cycle
        tmp_init = self.alloc_scratch("tmp_init")

        # Process pairs of variables
        for i in range(0, len(init_vars), 2):
            if i + 1 < len(init_vars):
                # Load two constants in parallel
                self.instrs.append({"load": [
                    ("const", tmp1, i),
                    ("const", tmp_init, i + 1)
                ]})
                # Load two values in parallel
                self.instrs.append({"load": [
                    ("load", self.scratch[init_vars[i]], tmp1),
                    ("load", self.scratch[init_vars[i + 1]], tmp_init)
                ]})
            else:
                # Odd number of variables, handle last one
                self.add("load", ("const", tmp1, i))
                self.add("load", ("load", self.scratch[init_vars[i]], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")
        tmp_addr2 = self.alloc_scratch("tmp_addr2")  # Second address register for parallel loads
        # Dedicated address registers for input/output (reused between load and store)
        addr_indices = self.alloc_scratch("addr_indices")
        addr_values = self.alloc_scratch("addr_values")

        for round in range(rounds):
            for i in range(batch_size):
                i_const = self.scratch_const(i)

                # Parallel load of idx and val - compute both addresses in parallel (ONCE)
                self.instrs.append({"alu": [
                    ("+", addr_indices, self.scratch["inp_indices_p"], i_const),
                    ("+", addr_values, self.scratch["inp_values_p"], i_const)
                ]})
                # Load both values in parallel
                self.instrs.append({"load": [
                    ("load", tmp_idx, addr_indices),
                    ("load", tmp_val, addr_values)
                ]})
                self.add("debug", ("compare", tmp_idx, (round, i, "idx")))
                self.add("debug", ("compare", tmp_val, (round, i, "val")))

                # node_val = mem[forest_values_p + idx]
                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_node_val, (round, i, "node_val"))))
                # val = myhash(val ^ node_val)
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))

                # Process body up to hash
                body_instrs = self.build(body)
                self.instrs.extend(body_instrs)
                body = []

                # Add parallelized hash instructions
                self.instrs.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                self.add("debug", ("compare", tmp_val, (round, i, "hashed_val")))

                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                # Optimized: use bitwise AND instead of modulo, parallelize independent ops
                # Cycle 1: Parallel - get parity bit and double index
                self.instrs.append({"alu": [
                    ("&", tmp1, tmp_val, one_const),        # tmp1 = val & 1 (0 if even, 1 if odd)
                    ("*", tmp_idx, tmp_idx, two_const)      # idx = idx * 2
                ]})
                # Cycle 2: tmp3 = 1 + tmp1 (gives 1 if even, 2 if odd)
                body.append(("alu", ("+", tmp3, one_const, tmp1)))
                # Cycle 3: idx = idx + tmp3
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "next_idx"))))
                
                # idx = 0 if idx >= n_nodes else idx
                body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "wrapped_idx"))))

                # Process the remaining body instructions
                body_instrs = self.build(body)
                self.instrs.extend(body_instrs)
                body = []

                # Store both values in parallel - reuse addresses from load (no recomputation!)
                self.instrs.append({"store": [
                    ("store", addr_indices, tmp_idx),
                    ("store", addr_values, tmp_val)
                ]})
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

    def build_kernel_vectorized(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized kernel using VLEN=8 SIMD operations.
        Processes 8 batch items in parallel per iteration.
        """
        # Temporary scalar registers
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")

        # Initialize same as scalar version
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        # Parallel initialization
        tmp_init = self.alloc_scratch("tmp_init")
        for i in range(0, len(init_vars), 2):
            if i + 1 < len(init_vars):
                self.instrs.append({"load": [
                    ("const", tmp1, i),
                    ("const", tmp_init, i + 1)
                ]})
                self.instrs.append({"load": [
                    ("load", self.scratch[init_vars[i]], tmp1),
                    ("load", self.scratch[init_vars[i + 1]], tmp_init)
                ]})
            else:
                self.add("load", ("const", tmp1, i))
                self.add("load", ("load", self.scratch[init_vars[i]], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting vectorized loop"))

        # Vector scratch registers (8 elements each)
        vec_idx = self.alloc_scratch("vec_idx", VLEN)
        vec_val = self.alloc_scratch("vec_val", VLEN)
        vec_node_val = self.alloc_scratch("vec_node_val", VLEN)
        vec_tmp1 = self.alloc_scratch("vec_tmp1", VLEN)
        vec_tmp2 = self.alloc_scratch("vec_tmp2", VLEN)
        vec_tmp3 = self.alloc_scratch("vec_tmp3", VLEN)
        vec_addr = self.alloc_scratch("vec_addr", VLEN)  # For gather addresses

        # Pre-allocate dedicated vector registers for all hash constants
        # 6 hash stages × 2 constants each = 12 total registers
        hash_const_regs = []
        for stage_idx in range(len(HASH_STAGES)):
            reg1 = self.alloc_scratch(f"hash_c{stage_idx}_v1", VLEN)
            reg2 = self.alloc_scratch(f"hash_c{stage_idx}_v2", VLEN)
            hash_const_regs.append((reg1, reg2))

        # Scalar address registers
        addr_base_indices = self.alloc_scratch("addr_base_indices")
        addr_base_values = self.alloc_scratch("addr_base_values")

        # Broadcast constants to vectors
        vec_one = self.alloc_scratch("vec_one", VLEN)
        vec_two = self.alloc_scratch("vec_two", VLEN)
        vec_zero = self.alloc_scratch("vec_zero", VLEN)
        vec_n_nodes = self.alloc_scratch("vec_n_nodes", VLEN)
        vec_forest_p = self.alloc_scratch("vec_forest_p", VLEN)

        self.add("valu", ("vbroadcast", vec_one, one_const))
        self.add("valu", ("vbroadcast", vec_two, two_const))
        self.add("valu", ("vbroadcast", vec_zero, zero_const))
        self.add("valu", ("vbroadcast", vec_n_nodes, self.scratch["n_nodes"]))
        self.add("valu", ("vbroadcast", vec_forest_p, self.scratch["forest_values_p"]))

        for round in range(rounds):
            for i in range(0, batch_size, VLEN):
                i_const = self.scratch_const(i)

                # Vector load of indices and values
                self.instrs.append({"alu": [
                    ("+", addr_base_indices, self.scratch["inp_indices_p"], i_const),
                    ("+", addr_base_values, self.scratch["inp_values_p"], i_const)
                ]})
                self.instrs.append({"load": [
                    ("vload", vec_idx, addr_base_indices),
                    ("vload", vec_val, addr_base_values)
                ]})

                # Debug compares for loaded values
                for lane in range(VLEN):
                    self.add("debug", ("compare", vec_idx + lane, (round, i + lane, "idx")))
                    self.add("debug", ("compare", vec_val + lane, (round, i + lane, "val")))

                # Gather: node_val = mem[forest_values_p + idx] for each lane
                # Compute all addresses: vec_addr = vec_forest_p + vec_idx
                # Also pre-broadcast hash constants during gather to utilize idle VALU slots
                # For stages 0,2,4: broadcast constant and multiplier for multiply_add
                # For stages 1,3,5: broadcast val1 and val3 as before

                # Cycle 1: Compute addresses + broadcast constants for hash stages 0 & 1
                # Stage 0: multiply_add needs const=0x7ed55d16, multiplier=4097 (1 + 2^12)
                # Stage 1: standard needs 0xc761c23c and 19
                self.instrs.append({"valu": [
                    ("+", vec_addr, vec_forest_p, vec_idx),
                    ("vbroadcast", hash_const_regs[0][0], self.scratch_const(HASH_STAGES[0][1])),  # const
                    ("vbroadcast", hash_const_regs[0][1], self.scratch_const(1 + (1 << HASH_STAGES[0][4]))),  # multiplier 4097
                    ("vbroadcast", hash_const_regs[1][0], self.scratch_const(HASH_STAGES[1][1]))
                ]})

                # Cycle 2: Load lanes 0-1 + broadcast stage 1 val3 and stage 2 both
                # Stage 2: multiply_add needs const=0x165667b1, multiplier=33 (1 + 2^5)
                self.instrs.append({
                    "load": [
                        ("load", vec_node_val + 0, vec_addr + 0),
                        ("load", vec_node_val + 1, vec_addr + 1)
                    ],
                    "valu": [
                        ("vbroadcast", hash_const_regs[1][1], self.scratch_const(HASH_STAGES[1][4])),
                        ("vbroadcast", hash_const_regs[2][0], self.scratch_const(HASH_STAGES[2][1])),  # const
                        ("vbroadcast", hash_const_regs[2][1], self.scratch_const(1 + (1 << HASH_STAGES[2][4])))  # multiplier 33
                    ]
                })

                # Cycle 3: Load lanes 2-3 + broadcast stage 3 both
                # Stage 3: standard needs 0xd3a2646c and 9
                self.instrs.append({
                    "load": [
                        ("load", vec_node_val + 2, vec_addr + 2),
                        ("load", vec_node_val + 3, vec_addr + 3)
                    ],
                    "valu": [
                        ("vbroadcast", hash_const_regs[3][0], self.scratch_const(HASH_STAGES[3][1])),
                        ("vbroadcast", hash_const_regs[3][1], self.scratch_const(HASH_STAGES[3][4]))
                    ]
                })

                # Cycle 4: Load lanes 4-5 + broadcast stage 4 both
                # Stage 4: multiply_add needs const=0xfd7046c5, multiplier=9 (1 + 2^3)
                self.instrs.append({
                    "load": [
                        ("load", vec_node_val + 4, vec_addr + 4),
                        ("load", vec_node_val + 5, vec_addr + 5)
                    ],
                    "valu": [
                        ("vbroadcast", hash_const_regs[4][0], self.scratch_const(HASH_STAGES[4][1])),  # const
                        ("vbroadcast", hash_const_regs[4][1], self.scratch_const(1 + (1 << HASH_STAGES[4][4])))  # multiplier 9
                    ]
                })

                # Cycle 5: Load lanes 6-7 + broadcast stage 5 both
                # Stage 5: standard needs 0xb55a4f09 and 16
                self.instrs.append({
                    "load": [
                        ("load", vec_node_val + 6, vec_addr + 6),
                        ("load", vec_node_val + 7, vec_addr + 7)
                    ],
                    "valu": [
                        ("vbroadcast", hash_const_regs[5][0], self.scratch_const(HASH_STAGES[5][1])),
                        ("vbroadcast", hash_const_regs[5][1], self.scratch_const(HASH_STAGES[5][4]))
                    ]
                })

                # Debug node_val
                for lane in range(VLEN):
                    self.add("debug", ("compare", vec_node_val + lane, (round, i + lane, "node_val")))

                # XOR: vec_val = vec_val ^ vec_node_val
                self.add("valu", ("^", vec_val, vec_val, vec_node_val))

                # Vectorized hash (using pre-broadcast constants)
                hash_instrs = self.build_hash_vectorized(vec_val, vec_tmp1, vec_tmp2, hash_const_regs, round, i)
                self.instrs.extend(hash_instrs)

                # Debug hashed values
                for lane in range(VLEN):
                    self.add("debug", ("compare", vec_val + lane, (round, i + lane, "hashed_val")))

                # Index computation: vec_idx = 2*vec_idx + (1 if vec_val % 2 == 0 else 2)
                # vec_tmp1 = vec_val & 1
                # vec_idx = vec_idx * 2
                self.instrs.append({"valu": [
                    ("&", vec_tmp1, vec_val, vec_one),
                    ("*", vec_idx, vec_idx, vec_two)
                ]})
                # vec_tmp3 = 1 + vec_tmp1
                self.add("valu", ("+", vec_tmp3, vec_one, vec_tmp1))
                # vec_idx = vec_idx + vec_tmp3
                self.add("valu", ("+", vec_idx, vec_idx, vec_tmp3))

                # Debug next_idx
                for lane in range(VLEN):
                    self.add("debug", ("compare", vec_idx + lane, (round, i + lane, "next_idx")))

                # Bounds check: vec_idx = 0 if vec_idx >= n_nodes else vec_idx
                self.add("valu", ("<", vec_tmp1, vec_idx, vec_n_nodes))
                self.add("flow", ("vselect", vec_idx, vec_tmp1, vec_idx, vec_zero))

                # Debug wrapped_idx
                for lane in range(VLEN):
                    self.add("debug", ("compare", vec_idx + lane, (round, i + lane, "wrapped_idx")))

                # Vector store of results
                self.instrs.append({"store": [
                    ("vstore", addr_base_indices, vec_idx),
                    ("vstore", addr_base_values, vec_val)
                ]})

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


def do_kernel_test_vectorized(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"VECTORIZED: {forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel_vectorized(forest.height, len(forest.values), len(inp.indices), rounds)

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

    def test_kernel_vectorized(self):
        # Test vectorized version
        do_kernel_test_vectorized(10, 16, 256)

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
