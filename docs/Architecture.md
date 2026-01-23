# Architecture

## Machine Overview
The machine in [problem.py](../problem.py) is a **VLIW (Very Large Instruction Word)** single-core simulator with SIMD support.

## Core Components
- **scratch**: Local memory (1536 words of 32-bit values), acts as registers and manually-managed cache
- **trace_buf**: Debug buffer (can be ignored for optimization)
- **pc**: Program counter
- **state**: Running/Paused/Stopped flag

## Execution Model
Single core with multiple parallel execution engines. Each cycle executes one instruction bundle containing slots for different engines. All reads happen before any writes (end-of-cycle semantics).

## Execution Engines (Slots per Cycle)

### ALU (12 slots)
Scalar arithmetic and logic operations:
- Arithmetic: `+`, `-`, `*`, `//`, `cdiv` (ceiling division)
- Bitwise: `^`, `&`, `|`, `<<`, `>>`
- Comparison: `<`, `==`
- Modulo: `%`

### VALU (6 slots)
Vector operations on VLEN=8 elements:
- All ALU operations (applied element-wise)
- `vbroadcast`: Copy scalar to all 8 vector elements
- `multiply_add`: Fused multiply-add operation (a * b + c)

### LOAD (2 slots)
Memory to scratch transfers:
- `load`: Load single value from memory
- `load_offset`: Load with offset (useful for vectorized loads)
- `vload`: Load 8 contiguous memory values to vector
- `const`: Load immediate constant

### STORE (2 slots)
Scratch to memory transfers:
- `store`: Store single value to memory
- `vstore`: Store 8 contiguous vector values to memory

### FLOW (1 slot)
Control flow and conditional operations:
- `select`: Conditional scalar select (ternary operator)
- `vselect`: Conditional vector select (element-wise)
- `add_imm`: Add immediate constant to value
- `jump`, `jump_indirect`: Unconditional jumps
- `cond_jump`, `cond_jump_rel`: Conditional jumps (absolute/relative)
- `halt`, `pause`: Stop/pause execution

## Key Constraints
- **1 core only** - no thread-level parallelism
- **VLIW parallelism** - maximize instruction-level parallelism within slot limits
- **SIMD width** - 8 elements per vector operation
- **End-of-cycle writes** - all reads complete before any writes take effect
