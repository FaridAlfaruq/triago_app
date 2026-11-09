import fitz # PyMuPDF
import sys
import re

sys.stdout.reconfigure(encoding="utf-8")

doc = fitz.open("scratch/gemastik/Gemastik para pencari berkah.pdf")
print(f"Total pages in PDF: {len(doc)}")

foreign_terms = [
    "edge computing", "ground", "real-time", "reverse proxy", "throughput",
    "cuffless", "backend", "database", "dashboard", "publisher", "broker",
    "cloud", "over-triage", "under-triage", "downsample", "detrending",
    "filtering", "machine learning", "deep learning", "bandpass", "notch",
    "baseline", "confusion matrix", "cross-validation", "pipeline"
]

results = {term: [] for term in foreign_terms}

for page_idx in range(len(doc)):
    page = doc[page_idx]
    # get text spans with font info
    blocks = page.get_text("dict")["blocks"]
    for b in blocks:
        if "lines" in b:
            for line in b["lines"]:
                for span in line["spans"]:
                    text = span["text"]
                    font = span["font"]
                    flags = span["flags"] # bit 1 is italic: flags & 2
                    is_italic = bool(flags & 2) or ("Italic" in font) or ("Oblique" in font)
                    
                    for term in foreign_terms:
                        if re.search(r"\b" + re.escape(term) + r"\b", text, re.IGNORECASE):
                            results[term].append({
                                "page": page_idx + 1,
                                "text": text,
                                "font": font,
                                "italic": is_italic
                            })

print("=== FONT STYLING AUDIT OF FOREIGN TERMS IN PDF ===")
for term, occurrences in results.items():
    if occurrences:
        italic_count = sum(1 for o in occurrences if o["italic"])
        non_italic_count = len(occurrences) - italic_count
        print(f"\nTerm: '{term}' (Total: {len(occurrences)}, Italic: {italic_count}, Upright: {non_italic_count})")
        for o in occurrences[:3]:
            status = "ITALIC" if o["italic"] else "UPRIGHT (NOT ITALIC)"
            print(f"  Page {o['page']}: [{status}] font={o['font']} text='{o['text'].strip()}'")
