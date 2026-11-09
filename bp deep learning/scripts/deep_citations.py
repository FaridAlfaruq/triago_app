import sys
import re
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

text = Path("scratch/gemastik/logs/proposal_extracted_text.txt").read_text(encoding="utf-8", errors="ignore")

dp_idx = text.find("DAFTAR PUSTAKA \n[1]")
if dp_idx == -1:
    dp_idx = text.find("DAFTAR PUSTAKA")

body_text = text[:dp_idx]
ref_text = text[dp_idx:]

print("=== CITATION CONTEXTS IN BODY ===")
for m in re.finditer(r"\[\d+\]", body_text):
    start = max(0, m.start() - 100)
    end = min(len(body_text), m.end() + 100)
    snippet = body_text[start:end].replace("\n", " ")
    print(f"{m.group()}: {snippet}\n")

print("\n=== FULL DAFTAR PUSTAKA TEXT ===")
print(ref_text)
