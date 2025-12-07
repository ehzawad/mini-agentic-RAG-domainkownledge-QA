from pathlib import Path
from shutil import copy2
from pypdf import PdfReader, PdfWriter

# Folder that currently has your 30 PDFs
source_dir = Path("nist_publications_30pdfs")   # change if needed

# Folder where the truncated PDFs will be written
target_dir = Path("nist_publications_truncated5")
target_dir.mkdir(parents=True, exist_ok=True)

MAX_PAGES = 5

for pdf_path in source_dir.glob("*.pdf"):
    reader = PdfReader(str(pdf_path))
    num_pages = len(reader.pages)

    if num_pages <= MAX_PAGES:
        # Copy short PDFs as-is
        out_path = target_dir / pdf_path.name
        copy2(pdf_path, out_path)
        print(f"COPIED   ({num_pages:3d} pages): {pdf_path.name} -> {out_path.name}")
    else:
        # Create a new PDF with only the first MAX_PAGES pages
        writer = PdfWriter()
        for i in range(MAX_PAGES):
            writer.add_page(reader.pages[i])

        out_name = pdf_path.stem + f"_first{MAX_PAGES}.pdf"
        out_path = target_dir / out_name

        with open(out_path, "wb") as f:
            writer.write(f)

        print(f"TRUNCATE ({num_pages:3d} pages -> {MAX_PAGES}): {pdf_path.name} -> {out_name}")

print(f"\nDone. Truncated PDFs stored in: {target_dir.resolve()}")
