import sys
import re
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

text = Path("scratch/gemastik/logs/proposal_extracted_text.txt").read_text(encoding="utf-8")
pages = text.split("--- PAGE ")

print(f"Total pages parsed: {len(pages)-1}")

# Search for equations
print("\n--- EQUATIONS FOUND ---")
eq_pattern = re.compile(r"Persamaan\s*\(([0-9]+\.[0-9]+)\)", re.IGNORECASE)
found_eqs = set(eq_pattern.findall(text))
print(f"Equation numbers found: {sorted(found_eqs)}")

# For each equation, extract its surrounding context
for eq in sorted(found_eqs):
    pos = text.find(f"({eq})")
    if pos != -1:
        start = max(0, pos - 250)
        end = min(len(text), pos + 250)
        print(f"\n>>> CONTEXT FOR PERSAMAAN ({eq}) <<<")
        print(text[start:end].strip())

