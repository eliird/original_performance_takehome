"""
Instruction dump tool - visualizes instruction bundles with fixed-width columns
to show resource usage and overlap across cycles.
"""

# Column widths
COL_WIDTH = 15
COLUMNS = ["LOAD", "STORE", "ALU", "VALU", "FLOW"]


class InstructionDumper:
    """Tracks and dumps instructions with iteration/round boundaries."""

    def __init__(self, output_file="instructions.txt"):
        self.output_file = output_file
        self.f = open(output_file, "w")
        self.cycle = 0
        self.iter_start = 0
        self.round_start = 0
        self.iter_num = 0
        self.round_num = 0

        # Track resource usage
        self.total_usage = {col: 0 for col in COLUMNS}
        self.iter_usage = {col: 0 for col in COLUMNS}
        self.round_usage = {col: 0 for col in COLUMNS}

        # Write header
        self._write_header()

    def _write_header(self):
        self.f.write(f"{'CYC':>5} | ")
        self.f.write(" | ".join(col.center(COL_WIDTH) for col in COLUMNS))
        self.f.write("\n")
        self.f.write("-" * (7 + (COL_WIDTH + 3) * len(COLUMNS)))
        self.f.write("\n")

    def add_instruction(self, instr):
        """Add a single instruction bundle."""
        # Convert tuple format to dict
        if isinstance(instr, tuple):
            engine, slot = instr
            instr = {engine: [slot]}

        # Skip debug-only instructions
        if all(k == "debug" for k in instr.keys()):
            return

        # Format each column
        cols = []
        for col in COLUMNS:
            key = col.lower()
            if key in instr:
                slots = instr[key]
                if slots:
                    op = slots[0][0] if slots else ""
                    count = len(slots)
                    text = f"{op}x{count}" if count > 1 else str(op)
                    cols.append(text[:COL_WIDTH].center(COL_WIDTH))
                    self.total_usage[col] += 1
                    self.iter_usage[col] += 1
                    self.round_usage[col] += 1
                else:
                    cols.append(" " * COL_WIDTH)
            else:
                cols.append(" " * COL_WIDTH)

        self.f.write(f"{self.cycle:>5} | ")
        self.f.write(" | ".join(cols))
        self.f.write("\n")
        self.cycle += 1

    def add_instructions(self, instrs):
        """Add multiple instructions."""
        for instr in instrs:
            self.add_instruction(instr)

    def end_iteration(self, batch_idx):
        """Mark end of an iteration within a round."""
        iter_cycles = self.cycle - self.iter_start
        self.f.write(f"\n--- Iter {self.iter_num} (batch {batch_idx}): {iter_cycles} cycles | ")
        for col in COLUMNS:
            pct = (self.iter_usage[col] / iter_cycles * 100) if iter_cycles > 0 else 0
            if self.iter_usage[col] > 0:
                self.f.write(f"{col}:{pct:.0f}% ")
        self.f.write("---\n\n")

        self.iter_start = self.cycle
        self.iter_num += 1
        self.iter_usage = {col: 0 for col in COLUMNS}

    def end_round(self):
        """Mark end of a round."""
        round_cycles = self.cycle - self.round_start
        self.f.write("\n")
        self.f.write("=" * 80)
        self.f.write(f"\n=== ROUND {self.round_num} COMPLETE: {round_cycles} cycles ===\n")
        self.f.write("Resource usage:\n")
        for col in COLUMNS:
            pct = (self.round_usage[col] / round_cycles * 100) if round_cycles > 0 else 0
            self.f.write(f"  {col}: {self.round_usage[col]:>4} cycles ({pct:>5.1f}%)\n")
        self.f.write("=" * 80)
        self.f.write("\n\n")

        self.round_start = self.cycle
        self.round_num += 1
        self.round_usage = {col: 0 for col in COLUMNS}

    def finalize(self):
        """Write final stats and close file."""
        self.f.write("\n")
        self.f.write("=" * 80)
        self.f.write(f"\nTOTAL CYCLES: {self.cycle}\n")
        self.f.write("Overall resource usage:\n")
        for col in COLUMNS:
            pct = (self.total_usage[col] / self.cycle * 100) if self.cycle > 0 else 0
            self.f.write(f"  {col}: {self.total_usage[col]:>4} cycles ({pct:>5.1f}%)\n")
        self.f.write("=" * 80)
        self.f.write("\n")
        self.f.close()


def dump_instructions(body, output_file="instructions.txt"):
    """
    Simple dump without iteration tracking.
    """
    dumper = InstructionDumper(output_file)
    dumper.add_instructions(body)
    dumper.finalize()


def analyze_body(body):
    """
    Analyzes instruction body and returns stats without writing to file.
    """
    stats = {
        "total_cycles": 0,
        "load_cycles": 0,
        "store_cycles": 0,
        "alu_cycles": 0,
        "valu_cycles": 0,
        "flow_cycles": 0,
    }

    for instr in body:
        if isinstance(instr, tuple):
            engine, slot = instr
            instr = {engine: [slot]}

        if all(k == "debug" for k in instr.keys()):
            continue

        stats["total_cycles"] += 1
        if "load" in instr:
            stats["load_cycles"] += 1
        if "store" in instr:
            stats["store_cycles"] += 1
        if "alu" in instr:
            stats["alu_cycles"] += 1
        if "valu" in instr:
            stats["valu_cycles"] += 1
        if "flow" in instr:
            stats["flow_cycles"] += 1

    return stats