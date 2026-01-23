# Claude's Guide to Performance Optimization

## Role
I am your expert guide for this VLIW-SIMD architecture. I understand the simulated processor, instruction set, and optimization opportunities. Ask me questions about the architecture, and we'll optimize the kernel together step by step.

---

## Architecture Summary

### Key Specifications
- **Cores:** 1 (N_CORES = 1, multicore disabled intentionally)
- **Vector Length:** 8 elements (VLEN = 8)
- **Scratch Memory:** 1536 words (32-bit each)
- **Word Size:** 32-bit integers (all operations mod 2^32)

### Execution Engines (Per Cycle Limits)
| Engine | Slots/Cycle | Purpose |
|--------|-------------|---------|
| ALU    | 12          | Scalar arithmetic/logic (+, -, *, //, ^, &, \|, <<, >>, %, <, ==) |
| VALU   | 6           | Vector arithmetic/logic (8-wide SIMD operations) |
| LOAD   | 2           | Memory → Scratch (load, vload, const) |
| STORE  | 2           | Scratch → Memory (store, vstore) |
| FLOW   | 1           | Control flow (select, vselect, add_imm, pause) |
| DEBUG  | ∞           | Development only (ignored in submission) |

### Key Constraint
**All operations within a cycle execute in PARALLEL.** Effects take place at END of cycle.

---

## Problem Description

### What the Kernel Does
The kernel performs a **parallel random walk through binary trees**:

1. **Input:** 256 batch items, each with:
   - `idx`: Current position in tree
   - `val`: Accumulated hash value

2. **Process (16 rounds):**
   For each batch item:
   - Read tree node value at current index
   - XOR with accumulated value
   - Apply multi-stage hash function
   - Decide next direction (left/right) based on hash parity
   - Navigate: `idx = 2*idx + (1 if hash%2==0 else 2)`
   - Wrap if out of bounds
   - Store updated idx and val

3. **Output:** Final hash values after 16 rounds

### Performance Target
- **Baseline:** 147,734 cycles
- **Goal:** Minimize cycles as much as possible
- **Test:** `python tests/submission_tests.py` (thresholds for different optimization levels)

---

## Optimizations Implemented So Far

### ✅ Optimization #1: Parallelized Initialization (Lines 111-130)

**What we changed:**
- Original: Sequential initialization (14 cycles)
- Optimized: Parallel initialization using 2 load slots per cycle (8 cycles)

**How it works:**
```python
# Allocate extra temp register
tmp_init = self.alloc_scratch("tmp_init")

# Process variables in pairs
for i in range(0, len(init_vars), 2):
    if i + 1 < len(init_vars):
        # Cycle N: Load two constants in parallel
        self.instrs.append({"load": [
            ("const", tmp1, i),
            ("const", tmp_init, i + 1)
        ]})
        # Cycle N+1: Load two values in parallel
        self.instrs.append({"load": [
            ("load", self.scratch[init_vars[i]], tmp1),
            ("load", self.scratch[init_vars[i + 1]], tmp_init)
        ]})
```

**Results:**
- **Cycles saved:** 6 cycles (14 → 8)
- **New total:** 147,728 cycles
- **Speedup:** 1.00004x (0.004%)

**Why small improvement?**
Initialization runs once, but the main loop runs 16 × 256 = 4,096 times.
To get significant speedup, we need to optimize the main loop body.

---

## Development Principles

### IMPORTANT: One Change at a Time
- Make **minimal, focused changes**
- Test after **each** optimization
- Don't try to optimize everything at once
- If confused, revert and restart with smaller scope

### Two Ways to Create Instructions

**Method 1: Using `body` list + `build()`** (Current approach for main loop)
```python
body = []
body.append(("alu", ("+", dest, a, b)))
body.append(("load", ("load", dest, addr)))
# ... at end:
body_instrs = self.build(body)  # Converts to sequential instructions
self.instrs.extend(body_instrs)
```

**Method 2: Direct `self.instrs.append()`** (What we used for init optimization)
```python
# Create parallel instruction bundle manually
self.instrs.append({
    "alu": [("+", dest1, a1, b1)],
    "load": [("load", dest2, addr)]
})  # Both execute in 1 cycle!
```

**Method 3: Using `self.add()`** (Helper for single instructions)
```python
self.add("alu", ("+", dest, a, b))
# Equivalent to: self.instrs.append({"alu": [("+", dest, a, b)]})
```

---

## Next Optimization Opportunities

I will talk to you, you explain things guide me and we optimize small things one at a time dont do too much.

---

## Important Constraints

### ❌ Do NOT:
- Modify `problem.py` (especially N_CORES, VLEN, SLOT_LIMITS)
- Modify anything in `tests/` folder
- Change the algorithm (must match reference kernel behavior)
- only allowed to edit the `build_kernel()` function dont touch anything

### ✅ DO:
- Optimize instruction packing in `build_kernel()`
- Add extra scratch registers as needed
- Reorder operations (respecting dependencies)
- Use vectorization (VALU, vload, vstore)
- Modify the `build()` method for better VLIW packing

---

## Questions?

Ask me anything about:
- How specific instructions work
- Dependency analysis for parallelization
- Vectorization strategies
- Architecture capabilities and limits
- Debugging techniques
- Performance analysis

Let's optimize this kernel together!
