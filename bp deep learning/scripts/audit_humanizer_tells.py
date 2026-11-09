import sys
import re
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

text = Path("scratch/gemastik/logs/proposal_extracted_text.txt").read_text(encoding="utf-8", errors="ignore")

tells = {
    "not_x_but_y": [
        r"tidak hanya\s+[^,\.]+,?\s*(tetapi juga|melainkan juga|namun juga)",
        r"bukan (hanya|sekadar)\s+[^,\.]+,?\s*(tetapi|melainkan)",
    ],
    "inflated_claims_sdgs": [
        r"Sustainable Development Goals",
        r"SDGs?",
        r"revolusioner",
        r"luar biasa",
        r"sangat krusial",
        r"bersifat krusial",
        r"hadir sebagai solusi",
        r"solusi komprehensif",
        r"berperan penting",
        r"memainkan peran kunci",
        r"tonggak penting",
        r"bukti nyata",
    ],
    "passive_avoiding_is": [
        r"berfungsi sebagai",
        r"hadir sebagai",
        r"merupakan suatu sistem yang",
    ],
    "forced_triads_or_staged": [
        r"tidak hanya\s+[^,\.]+\s*,\s*[^,\.]+\s*,\s*tetapi juga",
    ],
    "em_en_dashes": [
        r"—",
        r"–",
        r" -- ",
    ]
}

print("=== HUMANIZER & AI TELLS AUDIT ===")
for category, patterns in tells.items():
    print(f"\n--- Category: {category} ---")
    for pat in patterns:
        matches = list(re.finditer(pat, text, re.IGNORECASE))
        if matches:
            print(f"Pattern '{pat}': {len(matches)} matches")
            for m in matches[:5]:
                start = max(0, m.start() - 40)
                end = min(len(text), m.end() + 40)
                snippet = text[start:end].replace("\n", " ")
                print(f"   [{m.start()}]: ...{snippet}...")
