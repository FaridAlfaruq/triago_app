import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

text = Path("scratch/gemastik/logs/proposal_extracted_text.txt").read_text(encoding="utf-8", errors="ignore")
pages = text.split("--- PAGE ")
print("=== PAGE 2 (ABSTRAK) ===")
print(pages[2])
