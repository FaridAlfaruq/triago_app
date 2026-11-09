import sys
import re
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

text = Path("scratch/gemastik/logs/proposal_extracted_text.txt").read_text(encoding="utf-8", errors="ignore")

# Find all occurrences of Gambar and Tabel
print("=== CAPTIONS / CALLOUTS OF GAMBAR ===")
gambar_matches = re.findall(r"(Gambar\s+\d+\.\d+[^\n]*)", text)
for g in gambar_matches:
    print(g)

print("\n=== CAPTIONS / CALLOUTS OF TABEL ===")
tabel_matches = re.findall(r"(Tabel\s+\d+\.\d+[^\n]*)", text)
for t in tabel_matches:
    print(t)
