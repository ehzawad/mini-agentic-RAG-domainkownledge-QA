from pathlib import Path
from shutil import copy2
from pypdf import PdfReader, PdfWriter

# Original full PDFs (30 of them)
source_dir = Path("nist_publications_30pdfs")

# New folder: each PDF truncated to <= 10 pages
target_dir = Path("nist_publications_truncated10")
target_dir.mkdir(parents=True, exist_ok=True)

MAX_PAGES = 10

for pdf_path in source_dir.glob("*.pdf"):
    reader = PdfReader(str(pdf_path))
    num_pages = len(reader.pages)

    if num_pages <= MAX_PAGES:
        out_path = target_dir / pdf_path.name
        copy2(pdf_path, out_path)
        print(f"COPIED   ({num_pages:3d} pages): {pdf_path.name} -> {out_path.name}")
    else:
        writer = PdfWriter()
        pages_to_keep = min(MAX_PAGES, num_pages)
        for i in range(pages_to_keep):
            writer.add_page(reader.pages[i])

        out_name = pdf_path.stem + f"_first{MAX_PAGES}.pdf"
        out_path = target_dir / out_name

        with open(out_path, "wb") as f:
            writer.write(f)

        print(f"TRUNCATE ({num_pages:3d} pages -> {pages_to_keep}): {pdf_path.name} -> {out_name}")

print(f"\nDone. ≤{MAX_PAGES}-page PDFs are in: {target_dir.resolve()}")
