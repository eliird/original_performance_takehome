┌─────────────────────────────────────────────────────────────────┐
│                    KERNEL INITIALIZATION                         │
├─────────────────────────────────────────────────────────────────┤
│ 1. Allocate scratch registers (tmp1, tmp2, tmp3, etc.)          │
│ 2. Load parameters from memory into scratch:                    │
│    - rounds, n_nodes, batch_size, forest_height                 │
│    - forest_values_p, inp_indices_p, inp_values_p (pointers)    │
│ 3. Setup constants (0, 1, 2) in scratch                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
      ┌──────────────────────────────────────────┐
      │  FOR round = 0 to rounds-1 (16 times)    │
      │  ┌────────────────────────────────────┐  │
      │  │ FOR i = 0 to batch_size-1 (256)    │  │
      │  │                                    │  │
      │  │  ┌─────────────────────────────┐   │  │
      │  │  │  LOAD INPUTS                │   │  │
      │  │  │  • Load idx from inp_indices│   │  │
      │  │  │  • Load val from inp_values │   │  │
      │  │  └──────────┬──────────────────┘   │  │
      │  │             │                      │  │
      │  │             ▼                      │  │
      │  │  ┌─────────────────────────────┐   │  │
      │  │  │  TREE LOOKUP                │   │  │
      │  │  │  • node_val = forest[idx]   │   │  │
      │  │  └──────────┬──────────────────┘   │  │
      │  │             │                      │  │
      │  │             ▼                      │  │
      │  │  ┌─────────────────────────────┐   │  │
      │  │  │  HASH COMPUTATION           │   │  │
      │  │  │  • val = val ^ node_val     │   │  │
      │  │  │  • val = hash(val)          │   │  │
      │  │  │    ┌──────────────────┐     │   │  │
      │  │  │    │ HASH has N stages│     │   │  │
      │  │  │    │ (from HASH_STAGES│     │   │  │
      │  │  │    │  Each stage:     │     │   │  │
      │  │  │    │  tmp1 = op1(val) │     │   │  │
      │  │  │    │  tmp2 = op3(val) │     │   │  │
      │  │  │    │  val = op2(t1,t2)│     │   │  │
      │  │  │    └──────────────────┘     │   │  │
      │  │  └──────────┬──────────────────┘   │  │
      │  │             │                      │  │
      │  │             ▼                      │  │
      │  │  ┌─────────────────────────────┐   │  │
      │  │  │  COMPUTE NEXT INDEX         │   │  │
      │  │  │  • direction = 1 if val%2==0│   │  │
      │  │  │               else 2        │   │  │
      │  │  │  • idx = 2*idx + direction  │   │  │
      │  │  └──────────┬──────────────────┘   │  │
      │  │             │                      │  │
      │  │             ▼                      │  │
      │  │  ┌─────────────────────────────┐   │  │
      │  │  │  WRAP INDEX                 │   │  │
      │  │  │  • if idx >= n_nodes:       │   │  │
      │  │  │      idx = 0                │   │  │
      │  │  └──────────┬──────────────────┘   │  │
      │  │             │                      │  │
      │  │             ▼                      │  │
      │  │  ┌─────────────────────────────┐  │  │
      │  │  │  STORE RESULTS              │  │  │
      │  │  │  • inp_indices[i] = idx     │  │  │
      │  │  │  • inp_values[i] = val      │  │  │
      │  │  └─────────────────────────────┘  │  │
      │  │                                    │  │
      │  └────────────────────────────────────┘  │
      └──────────────────────────────────────────┘
                         │
                         ▼
                    ┌────────┐
                    │  DONE  │
                    └────────┘


For each of 256 batch items, 16 times:

   Start Position         Tree Traversal          End Position
   ══════════════         ══════════════          ════════════
   
   idx[i] ──────┐         Binary Tree             Updated idx
   val[i] ──────┼──►      (Navigate based    ──►  Updated val
                │         on hash parity)
                │
                └──► 1. Read tree node at idx
                     2. Mix: val ^= tree[idx]
                     3. Hash the mixed value
                     4. Go left (if hash even) or right (if hash odd)
                     5. Navigate: idx = 2*idx + (1 or 2)
                     6. Wrap if out of bounds


--------------------------------------------------------

═══════════════════════════════════════════════════════════════════════════════
                        SIMULATED VLIW-SIMD ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

CONFIGURATION:
  • Cores: N_CORES = 1 (single core)
  • Vector Length: VLEN = 8 (8-wide SIMD)
  • Scratch Memory: 1536 words (32-bit each)
  • Main Memory: Dynamic size (contains forest + inputs)

═══════════════════════════════════════════════════════════════════════════════
                              EXECUTION MODEL
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                              SINGLE CYCLE                                   │
│                                                                             │
│  All engines execute in parallel within a cycle                            │
│  Effects take place at END of cycle (after all engines read inputs)        │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │   ALU       │  │   VALU      │  │   LOAD      │  │   STORE     │       │
│  │  (Scalar)   │  │  (Vector)   │  │             │  │             │       │
│  │             │  │             │  │             │  │             │       │
│  │ 12 slots    │  │  6 slots    │  │  2 slots    │  │  2 slots    │       │
│  │             │  │             │  │             │  │             │       │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       │
│         │                │                │                │               │
│         └────────────────┴────────────────┴────────────────┘               │
│                                  │                                          │
│                                  ▼                                          │
│                         All writes happen                                   │
│                         at END of cycle                                     │
│                                                                             │
│  ┌─────────────┐                                                            │
│  │   FLOW      │  (Control flow, conditional selects)                      │
│  │             │                                                            │
│  │  1 slot     │                                                            │
│  └─────────────┘                                                            │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
                           EXECUTION ENGINES DETAIL
═══════════════════════════════════════════════════════════════════════════════

┌───────────────────────────────────────────────────────────────────────┐
│ ALU ENGINE (Scalar Arithmetic/Logic)                     Max: 12 slots│
├───────────────────────────────────────────────────────────────────────┤
│ Operations: +, -, *, //, cdiv, ^, &, |, <<, >>, %, <, ==            │
│                                                                       │
│ Format: ("alu", (op, dest, arg1, arg2))                              │
│ Example: ("alu", ("+", dest_addr, base_addr, offset))                │
│                                                                       │
│ • Works on single 32-bit words                                        │
│ • Reads from scratch space, writes to scratch space                  │
│ • All operations modulo 2^32                                          │
└───────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│ VALU ENGINE (Vector Arithmetic/Logic)                     Max: 6 slots│
├───────────────────────────────────────────────────────────────────────┤
│ Operations: All ALU ops + special vector ops                         │
│                                                                       │
│ Special Ops:                                                          │
│   • vbroadcast: Broadcast scalar to vector (VLEN=8 copies)           │
│   • multiply_add: Fused multiply-add (a*b + c)                       │
│                                                                       │
│ Format: ("valu", (op, dest, arg1, arg2))                             │
│ Example: ("valu", ("+", vec_dest, vec_a, vec_b))                     │
│          Computes vec_dest[i] = vec_a[i] + vec_b[i] for i=0..7       │
│                                                                       │
│ • Operates on vectors of VLEN=8 consecutive scratch words            │
│ • Single instruction processes 8 elements in parallel                │
└───────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│ LOAD ENGINE (Memory → Scratch)                            Max: 2 slots│
├───────────────────────────────────────────────────────────────────────┤
│ Operations:                                                           │
│   • load: Load single word from memory                               │
│   • vload: Load VLEN consecutive words (vector load)                 │
│   • const: Load immediate constant                                   │
│   • load_offset: Load with offset (for vectorization)                │
│                                                                       │
│ Format Examples:                                                      │
│   ("load", ("load", dest, addr_reg))      # scratch[dest] = mem[addr]│
│   ("load", ("vload", vec_dest, addr_reg)) # Load 8 words             │
│   ("load", ("const", dest, 42))           # scratch[dest] = 42       │
│                                                                       │
│ • Can issue 2 load operations per cycle                              │
│ • addr must be in scratch (indirect addressing)                      │
└───────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│ STORE ENGINE (Scratch → Memory)                           Max: 2 slots│
├───────────────────────────────────────────────────────────────────────┤
│ Operations:                                                           │
│   • store: Store single word to memory                               │
│   • vstore: Store VLEN consecutive words (vector store)              │
│                                                                       │
│ Format Examples:                                                      │
│   ("store", ("store", addr_reg, src))     # mem[addr] = scratch[src] │
│   ("store", ("vstore", addr_reg, vec_src))# Store 8 words            │
│                                                                       │
│ • Can issue 2 store operations per cycle                             │
│ • addr must be in scratch (indirect addressing)                      │
└───────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│ FLOW ENGINE (Control Flow)                                Max: 1 slot │
├───────────────────────────────────────────────────────────────────────┤
│ Operations:                                                           │
│   • select: Conditional select (ternary operator)                    │
│   • vselect: Vector conditional select                               │
│   • add_imm: Add immediate (fast offset calculation)                 │
│   • pause: Pause for debugging                                       │
│                                                                       │
│ Format Examples:                                                      │
│   ("flow", ("select", dest, cond, if_true, if_false))                │
│         # dest = if_true if cond != 0 else if_false                  │
│   ("flow", ("add_imm", dest, src, 5))                                │
│         # dest = src + 5                                              │
│                                                                       │
│ • Only 1 flow operation per cycle                                    │
│ • Useful for branch-free conditional logic                           │
└───────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│ DEBUG ENGINE (Development/Testing only)                  Unlimited    │
├───────────────────────────────────────────────────────────────────────┤
│ Operations:                                                           │
│   • compare: Compare value against reference kernel                  │
│   • comment: Add annotation to trace                                 │
│                                                                       │
│ • Ignored by submission tests                                        │
│ • Used for correctness verification during development               │
└───────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
                              MEMORY HIERARCHY
═══════════════════════════════════════════════════════════════════════════════
