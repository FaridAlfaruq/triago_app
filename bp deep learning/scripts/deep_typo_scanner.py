import sys
import re
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

text = Path("scratch/gemastik/logs/proposal_extracted_text.txt").read_text(encoding="utf-8", errors="ignore")

# Let's inspect potential typos using character n-grams or regular expressions
# Common Indonesian typo patterns:
# 1. Missing letters: dihilangan (dihilangkan), melakuakan (melakukan), meskipul (meskipun), prakti (praktis)
# 2. Repeated letters: performma (performa), perforrmma
# 3. Inconsistencies: di- vs kata kerja, ke- vs kata tempat

print("=== SEARCHING COMMON TYPOS ===")
checks = [
    r"\bdihilangan\b",
    r"\bmelakuakan\b",
    r"\bmeskipul\b",
    r"\bprakti\b",
    r"\bperformma\b",
    r"\bperforrmma\b",
    r"\bdimana\b",
    r"\bdisamping\b",
    r"\bmerubah\b",
    r"\bmerata-rata\b",
    r"\brata-\s*rata\b",
    r"\bmeminimalisir\b",
    r"\bmeminimalisasi\b",
    r"\bmengoptimalkan\b",
    r"\bmengoptimalisasi\b",
    r"\bantar\s+komponen\b",
    r"\bantar-komponen\b",
    r"\bantarkomponen\b",
    r"\bantar\s+petugas\b",
    r"\bantar-petugas\b",
    r"\bantardetak\b",
    r"\bantar\s+detak\b",
    r"\bstandarisasi\b",
    r"\bstandardisasi\b",
    r"\bpasien\b",
    r"\bpasin\b",
    r"\bprosentase\b",
    r"\bpersentase\b",
    r"\bsekaligus\b",
    r"\bsebagaimana\b",
    r"\bberikut\b",
]

for pat in checks:
    matches = list(re.finditer(pat, text, re.IGNORECASE))
    if matches:
        print(f"Pattern '{pat}': {len(matches)} matches")
        for m in matches[:3]:
            s = max(0, m.start() - 30)
            e = min(len(text), m.end() + 30)
            print(f"   [{m.start()}]: ...{text[s:e].replace(chr(10), ' ')}...")
