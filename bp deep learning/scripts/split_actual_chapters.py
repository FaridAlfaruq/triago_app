from pathlib import Path

text = Path("scratch/gemastik/logs/proposal_extracted_text.txt").read_text(encoding="utf-8")
pages = text.split("--- PAGE ")

def get_pages(start_p, end_p):
    out = []
    for p in range(start_p, end_p + 1):
        if p < len(pages):
            out.append(f"=== PAGE {p} ===\n" + pages[p])
    return "\n".join(out)

Path("scratch/gemastik/logs/bab1_actual.txt").write_text(get_pages(5, 6), encoding="utf-8")
Path("scratch/gemastik/logs/bab2_actual.txt").write_text(get_pages(7, 11), encoding="utf-8")
Path("scratch/gemastik/logs/bab3_actual.txt").write_text(get_pages(12, 16), encoding="utf-8")
Path("scratch/gemastik/logs/bab4_actual.txt").write_text(get_pages(17, 24), encoding="utf-8")
Path("scratch/gemastik/logs/bab5_actual.txt").write_text(get_pages(25, 28), encoding="utf-8")

print("Dumped actual chapters 1-5.")
