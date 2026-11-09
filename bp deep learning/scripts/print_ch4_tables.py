import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ch4 = Path("scratch/gemastik/logs/bab4_new.txt").read_text(encoding="utf-8", errors="ignore")
lines = ch4.splitlines()

# Let's print lines 180 to 410 to see all tables and their contents
for i in range(180, len(lines)):
    print(f"L{i+1:04d}: {lines[i]}")
