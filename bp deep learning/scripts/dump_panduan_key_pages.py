from pathlib import Path

text = Path("scratch/gemastik/logs/panduan_extracted_text.txt").read_text(encoding="utf-8")
pages = text.split("--- PAGE ")

def get_text_pages(start_p, end_p):
    return "\n".join([f"=== PAGE {p} ===\n" + pages[p] for p in range(start_p, end_p + 1) if p < len(pages)])

# Dump Divisi Piranti Cerdas, Sistem Benam, dan IoT
Path("scratch/gemastik/logs/panduan_iot_section.txt").write_text(get_text_pages(75, 82), encoding="utf-8")

# Dump Bab II Bagian B (Persyaratan Umum / Format Dokumen)
Path("scratch/gemastik/logs/panduan_bab2_umum.txt").write_text(get_text_pages(9, 16), encoding="utf-8")

print("Dumped specific IoT section and General Bab II section.")
