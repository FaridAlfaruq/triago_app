import sys
import re
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

text = Path("scratch/gemastik/logs/proposal_extracted_text.txt").read_text(encoding="utf-8", errors="ignore")

# Find decimal numbers with dots and commas: e.g. 4.90 vs 4,90, 99.15% vs 99,15%
# Let's search for patterns: \b\d+\.\d+\b vs \b\d+,\d+\b

dot_decimals = list(re.finditer(r"\b\d+\.\d+\b", text))
comma_decimals = list(re.finditer(r"\b\d+,\d+\b", text))

print(f"Total numbers with dots: {len(dot_decimals)}")
print(f"Total numbers with commas: {len(comma_decimals)}")

# Filter out chapter/section numbers (e.g. 1.1, 4.3.2) or equation numbers or dates
real_dot_decimals = []
for m in dot_decimals:
    val = m.group()
    # check context
    start = max(0, m.start() - 25)
    end = min(len(text), m.end() + 25)
    snippet = text[start:end].replace("\n", " ")
    # ignore if preceded by Bab, Gambar, Tabel, Persamaan, or if it's section numbering (e.g. at line start)
    if any(k in snippet for k in ["BAB", "Gambar", "Tabel", "Persamaan", "vol.", "no.", "pp.", "doi:"]):
        continue
    # check if it's a section heading like "1.1. Latar Belakang"
    if re.search(r"^\s*\d+\.\d+", text[m.start()-5:m.end()+10]):
        continue
    real_dot_decimals.append((val, snippet))

real_comma_decimals = []
for m in comma_decimals:
    val = m.group()
    start = max(0, m.start() - 25)
    end = min(len(text), m.end() + 25)
    snippet = text[start:end].replace("\n", " ")
    real_comma_decimals.append((val, snippet))

print(f"\nReal dot decimals in prose/tables: {len(real_dot_decimals)}")
for val, snip in real_dot_decimals[:15]:
    print(f"  Dot: {val:10s} in: ...{snip}...")

print(f"\nReal comma decimals in prose/tables: {len(real_comma_decimals)}")
for val, snip in real_comma_decimals[:15]:
    print(f"  Comma: {val:10s} in: ...{snip}...")
