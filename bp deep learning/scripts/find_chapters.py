import re
from pathlib import Path

text = Path("scratch/gemastik/logs/proposal_extracted_text.txt").read_text(encoding="utf-8")
pages = text.split("--- PAGE ")

print("=== FINDING CHAPTER STARTS (Excluding TOC) ===")
# Page 1 is index 1
for idx in range(1, len(pages)):
    p_text = pages[idx]
    for m in re.finditer(r"(BAB\s+[IVX]+[^\n]*)", p_text):
        # Ignore if "....." in line (table of contents)
        line = m.group(1)
        if "..." not in p_text[m.start():m.start()+80]:
            print(f"Page {idx}: {line.strip()}")
