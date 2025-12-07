from pathlib import Path
from shutil import move
from pypdf import PdfReader

source_dir = Path("nist_publications_truncated10")
scanned_dir = Path("nist_publications_scanned_10")
clean_dir = Path("nist_publications_clean_10")

scanned_dir.mkdir(parents=True, exist_ok=True)
clean_dir.mkdir(parents=True, exist_ok=True)

MAX_PAGES_CHECK = 10
MIN_CHARS_THRESHOLD = 200  # tweak if needed

def is_scanned_pdf(pdf_path: Path) -> bool:
    reader = PdfReader(str(pdf_path))
    num_pages = len(reader.pages)
    pages_to_check = min(num_pages, MAX_PAGES_CHECK)

    total_text = ""
    for i in range(pages_to_check):
        page = reader.pages[i]
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        total_text += text

    total_text = total_text.strip()
    return len(total_text) < MIN_CHARS_THRESHOLD

for pdf_path in source_dir.glob("*.pdf"):
    if is_scanned_pdf(pdf_path):
        dest = scanned_dir / pdf_path.name
        move(str(pdf_path), dest)
        print(f"SCANNED (moved): {pdf_path.name}")
    else:
        dest = clean_dir / pdf_path.name
        move(str(pdf_path), dest)
        print(f"TEXT    (kept):  {pdf_path.name}")

print("\nDone.")
print(f"Text PDFs are in:    {clean_dir.resolve()}")
print(f"Scanned PDFs are in: {scanned_dir.resolve()}")
