from pathlib import Path
import re

text = Path("scratch/gemastik/logs/proposal_extracted_text.txt").read_text(encoding="utf-8")

# Let's extract all equations or lines with "Persamaan" or "(" with numbers
eq_lines = []
for line in text.splitlines():
    if "persamaan" in line.lower() or re.search(r"\([1-9]\.[0-9]+\)", line):
        eq_lines.append(line.strip())

print("=== EQUATION REFERENCES ===")
for l in eq_lines[:30]:
    print(l)

# Let's search for "Suhu" or temperature equations and descriptions
print("\n=== SUHU / TEMPERATURE SECTIONS ===")
suhu_matches = [m.start() for m in re.finditer(r"suhu", text, re.IGNORECASE)]
for idx in suhu_matches[:5]:
    start = max(0, idx - 100)
    end = min(len(text), idx + 200)
    print("--- SNIPPET ---")
    print(text[start:end])
