import h5py
from pathlib import Path

data_dir = Path("/kaggle/input/pulsedb-balanced-training-and-testing")
print("Files in input:")
for f in sorted(data_dir.glob("*")):
    print(f" - {f.name} ({f.stat().st_size} bytes)")

for name in ["VitalDB_AAMI_Test_Subset.mat", "VitalDB_CalFree_Test_Subset.mat"]:
    fpath = data_dir / name
    if fpath.exists():
        print(f"\n=== Inspecting {name} ===")
        with h5py.File(fpath, "r") as hf:
            print("Keys:", list(hf.keys()))
            for k in hf.keys():
                if k == "#refs#":
                    continue
                item = hf[k]
                if hasattr(item, "shape"):
                    print(f"  {k}: shape={item.shape}, dtype={item.dtype}")
                elif hasattr(item, "keys"):
                    print(f"  {k} (group with {len(item.keys())} keys): {list(item.keys())[:15]}")
                    for subk in list(item.keys())[:5]:
                        subitem = item[subk]
                        if hasattr(subitem, "shape"):
                            print(f"    {subk}: shape={subitem.shape}, dtype={subitem.dtype}")

print("\nDone probe.")
