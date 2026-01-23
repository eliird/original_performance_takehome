# Kernel Analysis

## Reference Kernel Overview
Location: [problem.py:467-484](../problem.py#L467-L484)

The kernel performs **random tree traversal** on a batch of independent inputs over multiple rounds.

## Algorithm Breakdown

### Input Data
- **Tree**: Implicit perfect binary tree with values at each node
- **Input**: Batch of (index, value) pairs
  - `indices[i]`: Current node position in tree (starts at 0 = root)
  - `values[i]`: Current accumulated value
- **rounds**: Number of traversal iterations

### Per-Round, Per-Input Operations

```python
for h in range(inp.rounds):           # Outer loop: rounds
    for i in range(len(inp.indices)): # Inner loop: batch elements
        idx = inp.indices[i]           # 1. Load current index
        val = inp.values[i]            # 2. Load current value

        val = myhash(val ^ t.values[idx])  # 3. Hash: XOR with node value, then hash

        idx = 2 * idx + (1 if val % 2 == 0 else 2)  # 4. Navigate: left child if even, right if odd
        idx = 0 if idx >= len(t.values) else idx     # 5. Wrap: reset to root if out of bounds

        inp.values[i] = val            # 6. Store updated value
        inp.indices[i] = idx           # 7. Store updated index
```

### Step-by-Step Execution

**Step 1-2**: Load current state
- `idx`: Current tree node position
- `val`: Current accumulated value

**Step 3**: Compute new value
- XOR current value with tree node value
- Apply `myhash()` (6-stage hash function using +, ^, <<, >>)

**Step 4**: Tree navigation
- Calculate child node: `2 * idx + 1` (left) or `2 * idx + 2` (right)
- Direction based on hash result parity: even ’ left, odd ’ right

**Step 5**: Boundary check
- If index exceeds tree size, wrap back to root (idx = 0)

**Step 6-7**: Store results
- Update `values[i]` and `indices[i]` for next round

## Hash Function Details
`myhash()` at [problem.py:449-464](../problem.py#L449-L464):
- 6 sequential stages
- Each stage: `a = (a op1 constant) op2 (a op3 shift_amount)`
- Operations: +, ^, <<, >>
- All results are 32-bit (mod 2^32)
- **Cannot be parallelized** - each stage depends on previous result

## Parallelization Opportunities

###  Parallelizable (Independent)

**Across batch elements** (inner loop iterations):
- Each `i` in the batch is completely independent
- No data dependencies between different batch elements
- Different elements may traverse different tree paths
- **Key insight**: Process multiple batch elements simultaneously using SIMD

**Within each iteration**:
- Step 1-2 (loads) are independent
- Step 6-7 (stores) are independent
- Hash function stages can use pipelined VALU operations

### L NOT Parallelizable (Dependent)

**Across rounds** (outer loop):
- Round h+1 depends on results from round h
- Must complete all batch elements in round h before starting h+1

**Within each batch element**:
- Hash stages are sequential (stage N depends on stage N-1)
- New `idx` depends on new `val` (step 4 needs step 3)
- Can't traverse multiple tree levels simultaneously

**Memory dependencies**:
- Tree values are read-only (good for caching)
- Index/value arrays have read-modify-write pattern

## Optimization Strategy

1. **Vectorize across batch**: Process 8 elements per cycle using VALU
2. **Memory access**: Load tree values efficiently (they're random access)
3. **Hash computation**: Pipeline the 6 hash stages across VALU slots
4. **Minimize branches**: Use conditional select instead of jumps
5. **Loop unrolling**: Reduce loop overhead for inner/outer loops
