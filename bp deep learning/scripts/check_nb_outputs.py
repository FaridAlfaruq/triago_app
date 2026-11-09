import json
from pathlib import Path

nb_path = Path(r"C:\Users\Adyty\Documents\Farid ITS\TriaGo\model\bp.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb.get("cells", [])):
    if cell.get("cell_type") == "code":
        outs = cell.get("outputs", [])
        if outs:
            print(f"Cell {i} has {len(outs)} outputs:")
            for o in outs:
                text = "".join(o.get("text", []))
                if text:
                    for line in text.split("\n")[:5]:
                        if line.strip():
                            print(f"   {line}")
