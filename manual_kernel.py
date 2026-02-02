"""
Manual kernel construction for understanding optimal scheduling.
Start simple: 1 round, 2 batches (96 elements total, 48 per batch).
"""

from problem import (
    SLOT_LIMITS,
    VLEN,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    build_mem_image,
    reference_kernel2,
)

# Constants
N_PARALLEL = 6  # 6 vectors per batch
BATCH_ELEMENTS = N_PARALLEL * VLEN  # 48 elements per batch


class ManualKernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch_ptr = 0
        self.scratch = {}
        self.const_map = {}

    def alloc(self, name, length=1):
        addr = self.scratch_ptr
        self.scratch[name] = addr
        self.scratch_ptr += length
        return addr

    def const(self, val):
        if val not in self.const_map:
            addr = self.alloc(f"const_{val}")
            self.const_map[val] = addr
        return self.const_map[val]

    def emit(self, instr):
        """Emit a single instruction bundle."""
        self.instrs.append(instr)

    def build_kernel(self, forest_height=10, n_nodes=2047, batch_size=96, rounds=1):
        """
        Manually constructed kernel for 1 round, 2 batches.
        batch_size=96 means 2 batches of 48 elements each.
        """
        assert batch_size == 96, "This manual kernel is for 96 elements (2 batches)"
        assert rounds == 1, "This manual kernel is for 1 round"

        # === SCRATCH ALLOCATION ===
        # Header vars
        tmp1 = self.alloc("tmp1")
        for v in ["rounds", "n_nodes", "batch_size", "forest_height",
                  "forest_values_p", "inp_indices_p", "inp_values_p"]:
            self.alloc(v)

        # Vector registers for batch 0
        v_idx_0 = [self.alloc(f"v_idx_0_{p}", VLEN) for p in range(N_PARALLEL)]
        v_val_0 = [self.alloc(f"v_val_0_{p}", VLEN) for p in range(N_PARALLEL)]
        v_node_0 = [self.alloc(f"v_node_0_{p}", VLEN) for p in range(N_PARALLEL)]

        # Vector registers for batch 1
        v_idx_1 = [self.alloc(f"v_idx_1_{p}", VLEN) for p in range(N_PARALLEL)]
        v_val_1 = [self.alloc(f"v_val_1_{p}", VLEN) for p in range(N_PARALLEL)]
        v_node_1 = [self.alloc(f"v_node_1_{p}", VLEN) for p in range(N_PARALLEL)]

        # Temp vectors for hash
        v_tmp = [self.alloc(f"v_tmp_{p}", VLEN) for p in range(N_PARALLEL)]

        # Scalar addresses for gather
        gather_addr = [self.alloc(f"gather_addr_{p}") for p in range(N_PARALLEL)]

        # Base addresses for vload/vstore
        idx_addr_0 = [self.alloc(f"idx_addr_0_{p}") for p in range(N_PARALLEL)]
        val_addr_0 = [self.alloc(f"val_addr_0_{p}") for p in range(N_PARALLEL)]
        idx_addr_1 = [self.alloc(f"idx_addr_1_{p}") for p in range(N_PARALLEL)]
        val_addr_1 = [self.alloc(f"val_addr_1_{p}") for p in range(N_PARALLEL)]

        # Vector constants
        v_zero = self.alloc("v_zero", VLEN)
        v_two = self.alloc("v_two", VLEN)
        v_n_nodes = self.alloc("v_n_nodes", VLEN)

        # Hash constants (6 stages × 2 constants each)
        hash_c1 = [self.alloc(f"hash_c1_{i}", VLEN) for i in range(6)]
        hash_c2 = [self.alloc(f"hash_c2_{i}", VLEN) for i in range(6)]

        # === INITIALIZATION ===
        # Load header from memory
        for i, v in enumerate(["rounds", "n_nodes", "batch_size", "forest_height",
                               "forest_values_p", "inp_indices_p", "inp_values_p"]):
            self.emit({"load": [("const", tmp1, i)]})
            self.emit({"load": [("load", self.scratch[v], tmp1)]})

        zero = self.const(0)
        two = self.const(2)

        self.emit({"flow": [("pause",)]})

        # Broadcast constants
        self.emit({"valu": [("vbroadcast", v_zero, zero),
                           ("vbroadcast", v_two, two),
                           ("vbroadcast", v_n_nodes, self.scratch["n_nodes"])]})

        # Broadcast hash constants
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1 = self.const(val1)
            if op1 == "+" and op2 == "+" and op3 == "<<":
                c2 = self.const(1 + (1 << val3))
            else:
                c2 = self.const(val3)
            self.emit({"valu": [("vbroadcast", hash_c1[hi], c1),
                               ("vbroadcast", hash_c2[hi], c2)]})

        # Compute vload addresses for batch 0
        for p in range(N_PARALLEL):
            offset = self.const(p * VLEN)
            self.emit({"alu": [("+", idx_addr_0[p], self.scratch["inp_indices_p"], offset),
                              ("+", val_addr_0[p], self.scratch["inp_values_p"], offset)]})

        # Compute vload addresses for batch 1
        for p in range(N_PARALLEL):
            offset = self.const(BATCH_ELEMENTS + p * VLEN)
            self.emit({"alu": [("+", idx_addr_1[p], self.scratch["inp_indices_p"], offset),
                              ("+", val_addr_1[p], self.scratch["inp_values_p"], offset)]})

        # === MAIN COMPUTATION ===
        # Now let's write the optimal schedule manually!

        # Cycle-by-cycle planning:
        #
        # BATCH 0:
        #   VLOAD: 6 cycles (12 vloads)
        #   GATHER: 24 cycles (48 loads)
        #   XOR + HASH + INDEX: ~19 VALU ops
        #   (overlapped where possible)
        #
        # BATCH 1:
        #   Same, but can overlap with BATCH 0's stores
        #
        # STORES:
        #   6 cycles per batch (12 vstores)

        # Let's trace through cycle by cycle:

        print("=== BATCH 0 VLOAD (6 cycles) ===")
        # Cycles 0-5: Load idx/val for batch 0
        for p in range(0, N_PARALLEL, 2):
            self.emit({
                "load": [("vload", v_idx_0[p], idx_addr_0[p]),
                        ("vload", v_idx_0[p+1], idx_addr_0[p+1]) if p+1 < N_PARALLEL else None],
            })
        for p in range(0, N_PARALLEL, 2):
            self.emit({
                "load": [("vload", v_val_0[p], val_addr_0[p]),
                        ("vload", v_val_0[p+1], val_addr_0[p+1]) if p+1 < N_PARALLEL else None],
            })
        # Actually, let's do idx and val interleaved to use both load slots
        # Rewrite:

        # Clear and redo properly
        self.instrs = self.instrs[:-6]  # Remove last 6

        for p in range(N_PARALLEL):
            self.emit({
                "load": [("vload", v_idx_0[p], idx_addr_0[p]),
                        ("vload", v_val_0[p], val_addr_0[p])],
            })

        print("=== BATCH 0 GATHER (24 cycles) + Start BATCH 1 VLOAD ===")
        # During gather, we can also start computing batch 1's vload addresses
        # and do batch 0's XOR + hash as soon as gather completes

        # Gather: for each vi in 0..7, load node values for all 6 vectors
        # 8 vi values × 3 cycles each (6 vectors / 2 loads per cycle) = 24 cycles

        for vi in range(VLEN):
            # Compute addresses for this vi
            if vi == 0:
                # First vi: compute addresses
                self.emit({
                    "alu": [("+", gather_addr[p], self.scratch["forest_values_p"], v_idx_0[p] + vi)
                            for p in range(N_PARALLEL)],
                })

            # Load 2 nodes at a time
            for load_pair in range(0, N_PARALLEL, 2):
                bundle = {
                    "load": [("load", v_node_0[load_pair] + vi, gather_addr[load_pair])]
                }
                if load_pair + 1 < N_PARALLEL:
                    bundle["load"].append(("load", v_node_0[load_pair+1] + vi, gather_addr[load_pair+1]))

                # Compute addresses for next vi during first load cycle
                if load_pair == 0 and vi + 1 < VLEN:
                    bundle["alu"] = [("+", gather_addr[p], self.scratch["forest_values_p"], v_idx_0[p] + vi + 1)
                                     for p in range(N_PARALLEL)]

                self.emit(bundle)

        print("=== BATCH 0 XOR + HASH + INDEX (overlapped with BATCH 1 VLOAD) ===")
        # XOR: val ^= node_val
        self.emit({"valu": [("^", v_val_0[p], v_val_0[p], v_node_0[p]) for p in range(N_PARALLEL)]})

        # Hash stage 0: multiply_add
        self.emit({"valu": [("multiply_add", v_val_0[p], v_val_0[p], hash_c2[0], hash_c1[0]) for p in range(N_PARALLEL)]})

        # Hash stage 1: XOR with shift
        self.emit({"valu": [("^", v_tmp[p], v_val_0[p], hash_c1[1]) for p in range(N_PARALLEL)]})
        self.emit({"valu": [(">>", v_tmp[p], v_val_0[p], hash_c2[1]) for p in range(N_PARALLEL)]})
        self.emit({"valu": [("^", v_val_0[p], v_tmp[p], v_val_0[p]) for p in range(N_PARALLEL)]})  # Wrong order?

        # Continue hash... this is getting complex. Let me use the existing hash builder.

        # For now, let's just emit a placeholder and see the structure
        print("(Hash stages 2-5 would go here)")

        # Index update
        self.emit({"valu": [("%", v_tmp[p], v_val_0[p], v_two) for p in range(N_PARALLEL)]})
        self.emit({"valu": [("==", v_tmp[p], v_tmp[p], v_zero) for p in range(N_PARALLEL)]})
        self.emit({"valu": [("-", v_tmp[p], v_two, v_tmp[p]) for p in range(N_PARALLEL)]})
        self.emit({"valu": [("multiply_add", v_idx_0[p], v_idx_0[p], v_two, v_tmp[p]) for p in range(N_PARALLEL)]})
        self.emit({"valu": [("<", v_tmp[p], v_idx_0[p], v_n_nodes) for p in range(N_PARALLEL)]})
        self.emit({"valu": [("*", v_idx_0[p], v_idx_0[p], v_tmp[p]) for p in range(N_PARALLEL)]})

        # Store batch 0
        for p in range(N_PARALLEL):
            self.emit({
                "store": [("vstore", idx_addr_0[p], v_idx_0[p]),
                         ("vstore", val_addr_0[p], v_val_0[p])],
            })

        # Pause at end
        self.emit({"flow": [("pause",)]})

        return {
            "body": self.instrs,
            "scratch": self.scratch,
        }


def test_manual_kernel():
    """Test the manual kernel."""
    from dump_instructions import InstructionDumper

    forest_height = 10
    batch_size = 96
    rounds = 1

    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = ManualKernelBuilder()
    kernel = kb.build_kernel(forest_height, len(forest.values), batch_size, rounds)

    print(f"\nTotal instructions: {len(kernel['body'])}")
    print(f"Scratch used: {kb.scratch_ptr} / {SCRATCH_SIZE}")

    # Use InstructionDumper to create formatted output
    dumper = InstructionDumper("manual_instructions.txt")
    dumper.add_instructions(kernel['body'])
    dumper.finalize()

    print(f"\nInstruction dump written to manual_instructions.txt")

    # Also run the machine to verify and get actual cycle count
    try:
        from problem import DebugInfo
        debug_info = DebugInfo(scratch_map={})
        machine = Machine(mem, kernel['body'], debug_info, n_cores=1)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        print(f"Machine cycles: {machine.cycle}")

        # Compare with reference
        ref_mem = None
        for ref_mem in reference_kernel2(mem):
            pass

        if ref_mem is not None:
            inp_values_p = ref_mem[6]
            if machine.mem[inp_values_p:inp_values_p + batch_size] == ref_mem[inp_values_p:inp_values_p + batch_size]:
                print("CORRECT: Output matches reference!")
            else:
                print("ERROR: Output does not match reference!")
    except Exception as e:
        print(f"Machine execution failed: {e}")


if __name__ == "__main__":
    test_manual_kernel()
