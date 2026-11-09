from pathlib import Path
import re

text = Path("scratch/gemastik/logs/proposal_extracted_text.txt").read_text(encoding="utf-8")
pages = text.split("--- PAGE ")
print("=== PROPOSAL COVER & FRONTMATTER ===")
for i in range(1, 4):
    if i < len(pages):
        print(f"--- PAGE {i} ---")
        print(pages[i][:1000])
