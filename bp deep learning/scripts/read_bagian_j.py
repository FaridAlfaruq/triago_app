import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
text = Path("scratch/gemastik/logs/panduan_extracted_text.txt").read_text(encoding="utf-8")
pages = text.split("--- PAGE ")

for p in range(16, 21):
    if p < len(pages):
        print(f"=== PAGE {p} ===")
        print(pages[p][:1500])
