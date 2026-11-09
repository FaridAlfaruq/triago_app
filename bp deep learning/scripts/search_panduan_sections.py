import re
from pathlib import Path

text = Path("scratch/gemastik/logs/panduan_extracted_text.txt").read_text(encoding="utf-8")
pages = text.split("--- PAGE ")

# Search for divisions
print("=== DIVISI IN PANDUAN ===")
div_matches = re.findall(r"(?:Divisi\s+[IVX]+[^\n]+|DIVISI\s+[IVX]+[^\n]+)", text)
for d in set(div_matches):
    print(d)

# Search for "Piranti Cerdas" or "Sistem Benam" or "IoT"
print("\n=== SEARCHING FOR PIRANTI CERDAS / SISTEM BENAM / IOT SECTIONS ===")
for idx, page in enumerate(pages):
    if "piranti cerdas" in page.lower() or "sistem benam" in page.lower():
        lines = page.splitlines()
        for l in lines:
            if any(k in l.lower() for k in ["bab", "divisi", "sistematika", "laporan", "final", "penilaian", "tahap"]):
                print(f"Page {idx}: {l}")

