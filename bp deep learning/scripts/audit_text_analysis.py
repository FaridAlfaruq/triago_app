import sys
import re
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

text = Path("scratch/gemastik/logs/proposal_extracted_text.txt").read_text(encoding="utf-8", errors="ignore")

print("--- SEARCH DAFTAR PUSTAKA ---")
matches = [m.start() for m in re.finditer(r"DAFTAR PUSTAKA", text, re.IGNORECASE)]
for idx in matches:
    print(text[max(0, idx-50):min(len(text), idx+300)])
    print("="*50)

print("\n--- SEARCH PAGES ---")
pages = re.findall(r"--- PAGE (\d+) ---", text)
print(f"Total pages: {len(pages)}")

# Print pages 27, 28, 29
pages_split = text.split("--- PAGE ")
for i in range(len(pages_split)):
    if i in [1, 2, 3, 4, 27, 28, 29]:
        print(f"\n================ PAGE {i} ================")
        print(pages_split[i][:400])
