# Task

We will optimize the kernel in the `perf_takehome.py` together. Important thing is to remember the flow, I ask you question you answer the question dont implement anything and we decide on the optimization. Once we finalize that I will ask you to optimize.

We can only use 1 CORE, the only thing we are allowed to touch is build_kernel function for optimization not machine settings or anything.

## Optimization Opportunities

### 1. Initial Variable Loading (Lines 110-112)
**Location:** `perf_takehome.py:110-112`

**Current implementation:**
```python
for i, v in enumerate(init_vars):
    self.add("load", ("const", tmp1, i))
    self.add("load", ("load", self.scratch[v], tmp1))
```
- 7 iterations × 2 instructions = **14 cycles**
- Loads mem[0:7] into 7 consecutive scratch locations

**Optimization:**
Use `vload` to load 8 contiguous memory values at once:
```python
self.add("load", ("const", tmp1, 0))
self.add("load", ("vload", self.scratch["rounds"], tmp1))
```
- 1 const + 1 vload = **2 cycles**
- Loads mem[0:8] into 8 consecutive scratch locations
- Loads one extra value (mem[7] = extra_room pointer) but uses negligible extra scratch space

**Savings:** 12 cycles

**Rationale:**
- `init_vars` are allocated consecutively in scratch space
- Memory addresses 0-7 are consecutive in `build_mem_image`
- vload perfectly suited for this pattern

### 2. Address Recomputation in Loop (Lines 138, 142, 165, 168)
**Location:** `perf_takehome.py:138,142,165,168`

**Current implementation:**
```python
# Load idx
body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))  # Line 138
body.append(("load", ("load", tmp_idx, tmp_addr)))

# Load val
body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))   # Line 142
body.append(("load", ("load", tmp_val, tmp_addr)))

# ... hash and navigation logic ...

# Store idx - RECOMPUTES the same address!
body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))  # Line 165
body.append(("store", ("store", tmp_addr, tmp_idx)))

# Store val - RECOMPUTES the same address!
body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))   # Line 168
body.append(("store", ("store", tmp_addr, tmp_val)))
```
- Recomputes `inp_indices_p + i` twice (lines 138 and 165)
- Recomputes `inp_values_p + i` twice (lines 142 and 168)
- Wastes 2 ALU operations per inner loop iteration

**Optimization:**
Save computed addresses and reuse them:
```python
# Allocate scratch for addresses (outside loop)
tmp_idx_addr = self.alloc_scratch("tmp_idx_addr")
tmp_val_addr = self.alloc_scratch("tmp_val_addr")

# Inside loop:
# Load idx
body.append(("alu", ("+", tmp_idx_addr, self.scratch["inp_indices_p"], i_const)))
body.append(("load", ("load", tmp_idx, tmp_idx_addr)))

# Load val
body.append(("alu", ("+", tmp_val_addr, self.scratch["inp_values_p"], i_const)))
body.append(("load", ("load", tmp_val, tmp_val_addr)))

# ... hash and navigation logic ...

# Store idx - REUSE saved address!
body.append(("store", ("store", tmp_idx_addr, tmp_idx)))

# Store val - REUSE saved address!
body.append(("store", ("store", tmp_val_addr, tmp_val)))
```

**Savings:** 2 cycles per inner loop iteration = `2 × rounds × batch_size` cycles

**Trade-offs:**
- Uses 2 additional scratch slots (negligible - we have 1536 available)
- Addresses remain live from load to store (no conflicts)

**Rationale:**
- Addresses don't change during iteration
- No writes to base pointers (`inp_indices_p`, `inp_values_p`)
- Simple register reuse optimization