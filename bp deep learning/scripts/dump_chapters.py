import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
text = Path("scratch/gemastik/logs/proposal_extracted_text.txt").read_text(encoding="utf-8")

def find_between(start_str, end_str):
    p1 = text.find(start_str)
    if p1 == -1:
        return f"NOT FOUND: {start_str}"
    if end_str:
        p2 = text.find(end_str, p1)
        if p2 == -1:
            return text[p1:p1+4000]
        return text[p1:p2]
    return text[p1:]

# Dump BAB II
Path("scratch/gemastik/logs/bab2_extracted.txt").write_text(find_between("BAB II", "BAB III"), encoding="utf-8")
# Dump BAB III
Path("scratch/gemastik/logs/bab3_extracted.txt").write_text(find_between("BAB III", "BAB IV"), encoding="utf-8")
# Dump BAB IV
Path("scratch/gemastik/logs/bab4_extracted.txt").write_text(find_between("BAB IV", "BAB V"), encoding="utf-8")
# Dump BAB V & Daftar Pustaka
Path("scratch/gemastik/logs/bab5_extracted.txt").write_text(find_between("BAB V", ""), encoding="utf-8")

print("Dumped BAB II, III, IV, V to scratch/gemastik/logs/")
