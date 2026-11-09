import json
import sys
from pathlib import Path

nb_path = Path(r"C:\Users\Adyty\Documents\Farid ITS\TriaGo\model\bp.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

with open(r"scratch\gemastik\bp_cells_dump.txt", "w", encoding="utf-8") as out:
    for i, cell in enumerate(nb.get("cells", [])):
        cell_type = cell.get("cell_type")
        src = "".join(cell.get("source", []))
        out.write(f"\n{'='*40}\nCELL {i} [{cell_type}]\n{'='*40}\n")
        out.write(src + "\n")

print(f"Dumped {len(nb.get('cells', []))} cells to scratch/gemastik/bp_cells_dump.txt")
