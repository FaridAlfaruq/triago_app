import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
text = Path("scratch/gemastik/logs/panduan_extracted_text.txt").read_text(encoding="utf-8")
pages = text.split("--- PAGE ")

print(pages[16])
