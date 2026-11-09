import hashlib
from pathlib import Path

ppg_dir = Path(r"G:\My Drive\TriaGO_Project\Dataset\PPG")
ecg_dir = Path(r"G:\My Drive\TriaGO_Project\Dataset\ECG")

for s in ["subject056", "subject094", "subject122"]:
    p_p = ppg_dir / f"{s}_ppg.csv"
    e_p = ecg_dir / f"{s}_ecg.csv"
    print(f"{s}:")
    print(f"  PPG size={p_p.stat().st_size}, sha256={hashlib.sha256(p_p.read_bytes()).hexdigest()}")
    print(f"  ECG size={e_p.stat().st_size}, sha256={hashlib.sha256(e_p.read_bytes()).hexdigest()}")
