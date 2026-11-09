import json
from pathlib import Path

nb_path = Path(r"C:\Users\Adyty\Documents\Farid ITS\TriaGo\model\bp.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb.get('cells', []))}")
for i, cell in enumerate(nb.get("cells", [])):
    src = "".join(cell.get("source", []))
    lower_src = src.lower()
    if any(k in lower_src for k in ["train_test_split", "kfold", "groupkfold", "subject", "split", "leakage"]):
        print(f"=== Cell {i} ({cell.get('cell_type')}) ===")
        lines = [l for l in src.split("\n") if any(k in l.lower() for k in ["split", "train", "test", "val", "subject", "kfold", "group"])]
        for line in lines[:8]:
            print(f"  {line}")
