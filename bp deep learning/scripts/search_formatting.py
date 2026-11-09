import re
from pathlib import Path

text = Path("scratch/gemastik/logs/panduan_extracted_text.txt").read_text(encoding="utf-8")

keywords = ["font", "margin", "spasi", "times new roman", "arial", "a4", "laporan akhir", "makalah", "template"]

print("=== SEARCH FORMATTING RULES ===")
for kw in keywords:
    matches = [m.start() for m in re.finditer(re.escape(kw), text, re.IGNORECASE)]
    print(f"Keyword '{kw}': {len(matches)} occurrences")
    for idx in matches[:5]:
        start = max(0, idx - 100)
        end = min(len(text), idx + 200)
        print(f"[{kw}] snippet: ...{text[start:end].replace(chr(10), ' ')}...")
