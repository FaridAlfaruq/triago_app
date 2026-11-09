import json
from pathlib import Path

nb_path = Path(r"C:\Users\Adyty\Documents\Farid ITS\TriaGo\model\bp.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

with open(r"scratch\gemastik\nb_outputs_dump.txt", "w", encoding="utf-8") as out:
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            outs = cell.get("outputs", [])
            if outs:
                out.write(f"\n{'='*40}\nCELL {i} OUTPUTS\n{'='*40}\n")
                for o in outs:
                    text = "".join(o.get("text", []))
                    if text:
                        out.write(text + "\n")
print("Dumped outputs to nb_outputs_dump.txt")
