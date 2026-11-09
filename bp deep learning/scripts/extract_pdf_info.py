import fitz  # PyMuPDF
from pathlib import Path

pdf_path = Path("scratch/gemastik/Gemastik para pencari berkah.pdf")
doc = fitz.open(pdf_path)

print(f"Total pages: {len(doc)}")
full_text = []
for idx, page in enumerate(doc):
    text = page.get_text()
    full_text.append(f"--- PAGE {idx + 1} ---\n{text}\n")

out_path = Path("scratch/gemastik/logs/proposal_extracted_text.txt")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("".join(full_text), encoding="utf-8")
print(f"Extracted to {out_path}, size: {out_path.stat().st_size} bytes")
