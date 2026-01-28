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

    def build_hash_vec_parallel(self, v_val, v_tmp1, v_tmp2, hash_const_vecs, n_active, round, base_i):
        """
        Vector hash computation for n_active parallel vectors.
        Processes all vectors through each hash stage using VALU parallelism.

        Args:
            v_val: list of vector value addresses (will be modified in place)
            v_tmp1: list of vector temp1 addresses
            v_tmp2: list of vector temp2 addresses
            hash_const_vecs: list of (val1_vec, val3_vec) tuples for hash constants
            n_active: number of active parallel streams
            round: current round (for debug)
            base_i: base index for this batch (for debug)
        """
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            val1_vec, val3_vec = hash_const_vecs[hi]
            val1_const = self.scratch_const(val1)
            val3_const = self.scratch_const(val3)

            # Broadcast constants (only need to do once per stage)
            slots.append({
                "valu": [
                    ("vbroadcast", val1_vec, val1_const),
                    ("vbroadcast", val3_vec, val3_const),
                ]
            })

            # tmp1[p] = op1(v_val[p], val1_vec) for all active p
            slots.append({
                "valu": [(op1, v_tmp1[p], v_val[p], val1_vec) for p in range(n_active)]
            })

            # tmp2[p] = op3(v_val[p], val3_vec) for all active p
            slots.append({
                "valu": [(op3, v_tmp2[p], v_val[p], val3_vec) for p in range(n_active)]
            })

            # v_val[p] = op2(tmp1[p], tmp2[p]) for all active p
            slots.append({
                "valu": [(op2, v_val[p], v_tmp1[p], v_tmp2[p]) for p in range(n_active)]
            })

            # Debug compare
            for p in range(n_active):
                offset = base_i + p * VLEN
                slots.append(("debug", ("vcompare", v_val[p],
                    tuple((round, offset + vi, "hash_stage", hi) for vi in range(VLEN)))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized implementation processing N_PARALLEL * VLEN (6 * 8 = 48) elements at a time.
        Uses all 6 VALU slots in parallel.
        """
        N_PARALLEL = 6  # Number of vectors to process in parallel (matches VALU slot limit)

        # Scalar temporary (used for loading init vars)
        tmp1 = self.alloc_scratch("tmp1")

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

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps.
        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Allocate N_PARALLEL sets of vector registers
        v_idx = [self.alloc_scratch(f"v_idx_{p}", VLEN) for p in range(N_PARALLEL)]
        v_val = [self.alloc_scratch(f"v_val_{p}", VLEN) for p in range(N_PARALLEL)]
        v_node_val = [self.alloc_scratch(f"v_node_val_{p}", VLEN) for p in range(N_PARALLEL)]
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_{p}", VLEN) for p in range(N_PARALLEL)]
        v_tmp2 = [self.alloc_scratch(f"v_tmp2_{p}", VLEN) for p in range(N_PARALLEL)]
        v_tmp3 = [self.alloc_scratch(f"v_tmp3_{p}", VLEN) for p in range(N_PARALLEL)]

        # Shared vector constants (broadcast once)
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        # Scalar addresses for gather/scatter (one per parallel stream)
        tmp_addrs = [self.alloc_scratch(f"tmp_addr_{p}") for p in range(N_PARALLEL)]
        idx_base_addrs = [self.alloc_scratch(f"idx_base_addr_{p}") for p in range(N_PARALLEL)]
        val_base_addrs = [self.alloc_scratch(f"val_base_addr_{p}") for p in range(N_PARALLEL)]

        # Pre-allocate hash constant vectors (shared across all parallel streams)
        hash_const_vecs = []
        for hi in range(len(HASH_STAGES)):
            val1_vec = self.alloc_scratch(f"hash_val1_{hi}", VLEN)
            val3_vec = self.alloc_scratch(f"hash_val3_{hi}", VLEN)
            hash_const_vecs.append((val1_vec, val3_vec))

        # Initialize vector constants (can do up to 6 in parallel)
        body.append({
            "valu": [
                ("vbroadcast", v_zero, zero_const),
                ("vbroadcast", v_one, one_const),
                ("vbroadcast", v_two, two_const),
                ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
            ]
        })

        # Process batch_size elements in groups of N_PARALLEL * VLEN
        # With 256 batch_size and stride=48, we process: 0-47, 48-95, 96-143, 144-191, 192-239, then 240-255 (partial)
        # For simplicity, we require batch_size to be divisible by stride, or handle partial batches
        stride = N_PARALLEL * VLEN
        assert batch_size % VLEN == 0, f"batch_size must be divisible by VLEN ({VLEN})"

        for round in range(rounds):
            for base_i in range(0, batch_size, stride):
                # Calculate how many parallel streams we can use for this iteration
                remaining = batch_size - base_i
                n_active = min(N_PARALLEL, (remaining + VLEN - 1) // VLEN)

                # Compute base addresses for active parallel streams (use scalar ALU, up to 12 slots)
                alu_slots = []
                for p in range(n_active):
                    offset = base_i + p * VLEN
                    offset_const = self.scratch_const(offset)
                    alu_slots.append(("+", idx_base_addrs[p], self.scratch["inp_indices_p"], offset_const))
                    alu_slots.append(("+", val_base_addrs[p], self.scratch["inp_values_p"], offset_const))
                body.append({"alu": alu_slots})

                # Vector load indices and values (2 load slots per cycle)
                for p in range(n_active):
                    body.append({"load": [
                        ("vload", v_idx[p], idx_base_addrs[p]),
                        ("vload", v_val[p], val_base_addrs[p]),
                    ]})

                # Debug compare for indices and values
                for p in range(n_active):
                    offset = base_i + p * VLEN
                    body.append(("debug", ("vcompare", v_idx[p],
                        tuple((round, offset + vi, "idx") for vi in range(VLEN)))))
                    body.append(("debug", ("vcompare", v_val[p],
                        tuple((round, offset + vi, "val") for vi in range(VLEN)))))

                # Gather node_val = mem[forest_values_p + idx[i]] for each element
                # Use all 12 scalar ALU slots to compute addresses, then load
                # We have 8 elements per vector * n_active vectors gathers
                # With 12 ALU slots and 2 load slots per cycle, this is the bottleneck
                for vi in range(VLEN):
                    # Compute all n_active addresses in parallel (up to 12 ALU slots)
                    alu_slots = []
                    for p in range(n_active):
                        alu_slots.append(("+", tmp_addrs[p], self.scratch["forest_values_p"], v_idx[p] + vi))
                    body.append({"alu": alu_slots})

                    # Load 2 at a time (2 load slots)
                    for p in range(0, n_active, 2):
                        load_slots = [("load", v_node_val[p] + vi, tmp_addrs[p])]
                        if p + 1 < n_active:
                            load_slots.append(("load", v_node_val[p+1] + vi, tmp_addrs[p+1]))
                        body.append({"load": load_slots})

                # Debug compare for node_val
                for p in range(n_active):
                    offset = base_i + p * VLEN
                    body.append(("debug", ("vcompare", v_node_val[p],
                        tuple((round, offset + vi, "node_val") for vi in range(VLEN)))))

                # val = val ^ node_val (up to 6 VALUs in parallel)
                body.append({
                    "valu": [("^", v_val[p], v_val[p], v_node_val[p]) for p in range(n_active)]
                })

                # Hash computation - process all n_active vectors through each hash stage
                body.extend(self.build_hash_vec_parallel(
                    v_val, v_tmp1, v_tmp2, hash_const_vecs, n_active, round, base_i
                ))

                # Debug compare for hashed_val
                for p in range(n_active):
                    offset = base_i + p * VLEN
                    body.append(("debug", ("vcompare", v_val[p],
                        tuple((round, offset + vi, "hashed_val") for vi in range(VLEN)))))

                # idx = 2*idx + (1 if val % 2 == 0 else 2) - all n_active vectors in parallel
                # tmp1 = val % 2
                body.append({
                    "valu": [("%", v_tmp1[p], v_val[p], v_two) for p in range(n_active)]
                })
                # tmp1 = (tmp1 == 0)
                body.append({
                    "valu": [("==", v_tmp1[p], v_tmp1[p], v_zero) for p in range(n_active)]
                })
                # tmp3 = select(tmp1, 1, 2) - only 1 flow slot, so sequential
                for p in range(n_active):
                    body.append(("flow", ("vselect", v_tmp3[p], v_tmp1[p], v_one, v_two)))
                # idx = idx * 2
                body.append({
                    "valu": [("*", v_idx[p], v_idx[p], v_two) for p in range(n_active)]
                })
                # idx = idx + tmp3
                body.append({
                    "valu": [("+", v_idx[p], v_idx[p], v_tmp3[p]) for p in range(n_active)]
                })

                # Debug compare for next_idx
                for p in range(n_active):
                    offset = base_i + p * VLEN
                    body.append(("debug", ("vcompare", v_idx[p],
                        tuple((round, offset + vi, "next_idx") for vi in range(VLEN)))))

                # idx = 0 if idx >= n_nodes else idx
                # tmp1 = idx < n_nodes
                body.append({
                    "valu": [("<", v_tmp1[p], v_idx[p], v_n_nodes) for p in range(n_active)]
                })
                # idx = select(tmp1, idx, 0) - only 1 flow slot, so sequential
                for p in range(n_active):
                    body.append(("flow", ("vselect", v_idx[p], v_tmp1[p], v_idx[p], v_zero)))

                # Debug compare for wrapped_idx
                for p in range(n_active):
                    offset = base_i + p * VLEN
                    body.append(("debug", ("vcompare", v_idx[p],
                        tuple((round, offset + vi, "wrapped_idx") for vi in range(VLEN)))))

                # Vector store indices and values back (2 stores per cycle)
                for p in range(n_active):
                    body.append({"store": [
                        ("vstore", idx_base_addrs[p], v_idx[p]),
                        ("vstore", val_base_addrs[p], v_val[p]),
                    ]})

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
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
