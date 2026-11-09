import sys
import re
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

text = Path("scratch/gemastik/logs/proposal_extracted_text.txt").read_text(encoding="utf-8", errors="ignore")

# Find where DAFTAR PUSTAKA begins
dp_idx = text.find("DAFTAR PUSTAKA \n[1]")
if dp_idx == -1:
    dp_idx = text.find("DAFTAR PUSTAKA")
    # find next [1]
    dp_idx = text.find("[1]", dp_idx)

body_text = text[:dp_idx]
ref_text = text[dp_idx:]

# Find citations in body
# Find bracket numbers like [1], [1, 2], [1]-[3], [1], [2], etc.
citations_raw = re.findall(r"\[([0-9\s,\-]+)\]", body_text)
print("Raw citation strings in body:", citations_raw)

# Parse individual numbers
cited_nums = set()
for c in citations_raw:
    # Handle ranges like 1-3 or lists like 1, 2
    parts = c.split(",")
    for p in parts:
        p = p.strip()
        if "-" in p:
            subparts = p.split("-")
            if len(subparts) == 2 and subparts[0].isdigit() and subparts[1].isdigit():
                for num in range(int(subparts[0]), int(subparts[1]) + 1):
                    cited_nums.add(num)
        elif p.isdigit():
            cited_nums.add(int(p))

print(f"\nAll cited numbers in body ({len(cited_nums)} unique):", sorted(cited_nums))

# Parse DAFTAR PUSTAKA entries
ref_entries = re.findall(r"\[(\d+)\]\s*([^\[]+)", ref_text)
print(f"\nTotal references defined in DAFTAR PUSTAKA: {len(ref_entries)}")
ref_dict = {}
for num_str, entry in ref_entries:
    num = int(num_str)
    ref_dict[num] = entry.strip()

print("Reference keys in DAFTAR PUSTAKA:", sorted(ref_dict.keys()))

# Check missing citations (cited in body but not in DP)
missing_in_dp = [n for n in cited_nums if n not in ref_dict]
print("Cited in body but missing in DAFTAR PUSTAKA:", missing_in_dp)

# Check unused citations (in DP but not cited in body)
unused_in_body = [n for n in ref_dict if n not in cited_nums]
print("Defined in DAFTAR PUSTAKA but not cited in body:", unused_in_body)

# Print all references
print("\n=== DAFTAR PUSTAKA ENTRIES ===")
for k in sorted(ref_dict.keys()):
    print(f"[{k}] {ref_dict[k][:100]}...")
