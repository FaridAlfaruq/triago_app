import sys
import re
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

text = Path("scratch/gemastik/logs/proposal_extracted_text.txt").read_text(encoding="utf-8", errors="ignore")

print("--- CHAPTER 4 FIG & TABLE DETAILS ---")
ch4 = Path("scratch/gemastik/logs/bab4_new.txt").read_text(encoding="utf-8", errors="ignore")

# Let's inspect where Tabel appears in Bab 4
for line_no, line in enumerate(ch4.splitlines(), 1):
    if "Tabel" in line or "Gambar" in line:
        print(f"L{line_no:04d}: {line}")
