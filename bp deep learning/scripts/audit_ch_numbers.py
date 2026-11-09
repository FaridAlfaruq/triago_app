import sys
import re
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

for ch in range(1, 6):
    ch_text = Path(f"scratch/gemastik/logs/bab{ch}_new.txt").read_text(encoding="utf-8", errors="ignore")
    print(f"\n================ CHAPTER {ch} ================")
    
    # Find all decimal numbers with dots: e.g. 4.90, 2.3, 0.237, etc.
    # Exclude section headers like 4.3.1, or Figure/Table numbers like Gambar 4.1
    # Exclude dots that are thousands separators: e.g. 98.000, 20.000
    
    lines = ch_text.splitlines()
    for line_idx, line in enumerate(lines, 1):
        # find numbers with dots
        dot_nums = re.findall(r"(?<![A-Za-z0-9_])(\d+\.\d+)(?![A-Za-z0-9_\.])", line)
        comma_nums = re.findall(r"(?<![A-Za-z0-9_])(\d+,\d+)(?![A-Za-z0-9_\,])", line)
        
        # Filter out figure/table/equation/section refs
        clean_dots = []
        for d in dot_nums:
            # If line starts with "4.1." or "Gambar" or "Tabel" or "Persamaan"
            if re.search(r"^\s*" + re.escape(d), line):
                continue
            if f"Gambar {d}" in line or f"Tabel {d}" in line or f"Persamaan ({d})" in line or f"Persamaan {d}" in line:
                continue
            if f"({d})" in line:
                continue
            clean_dots.append(d)
            
        if clean_dots:
            print(f"L{line_idx:04d} [DOT ]: {clean_dots} | Line: {line.strip()[:100]}")
        if comma_nums:
            print(f"L{line_idx:04d} [COMMA]: {comma_nums} | Line: {line.strip()[:100]}")
