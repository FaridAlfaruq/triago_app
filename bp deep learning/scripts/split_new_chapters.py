import sys
import difflib
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

text = Path("scratch/gemastik/logs/proposal_extracted_text.txt").read_text(encoding="utf-8")
pages = text.split("--- PAGE ")

def get_pages(start_p, end_p):
    out = []
    for p in range(start_p, end_p + 1):
        if p < len(pages):
            out.append(f"=== PAGE {p} ===\n" + pages[p])
    return "\n".join(out)

new_ch = {
    1: get_pages(5, 6),
    2: get_pages(7, 11),
    3: get_pages(12, 16),
    4: get_pages(17, 26),
    5: get_pages(27, 29),
}

for ch_num, content in new_ch.items():
    old_file = Path(f"scratch/gemastik/logs/bab{ch_num}_actual.txt")
    new_file = Path(f"scratch/gemastik/logs/bab{ch_num}_new.txt")
    new_file.write_text(content, encoding="utf-8")
    
    if old_file.exists():
        old_lines = old_file.read_text(encoding="utf-8").splitlines()
        new_lines = content.splitlines()
        diff = list(difflib.unified_diff(old_lines, new_lines, fromfile=f"bab{ch_num}_old", tofile=f"bab{ch_num}_new", lineterm=""))
        diff_file = Path(f"scratch/gemastik/logs/diff_bab{ch_num}.txt")
        diff_file.write_text("\n".join(diff), encoding="utf-8")
        print(f"BAB {ch_num}: Diff size = {len(diff)} lines")
    else:
        print(f"BAB {ch_num}: No old file to diff against")

