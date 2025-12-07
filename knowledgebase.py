from huggingface_hub import list_repo_files, hf_hub_download
from pathlib import Path

repo_id = "ethanolivertroy/nist-publications-raw"

# 1. List all files in the dataset repo
all_files = list_repo_files(repo_id=repo_id, repo_type="dataset")

# 2. Keep only PDFs and sort for determinism
pdf_files = sorted([f for f in all_files if f.lower().endswith(".pdf")])

# 3. Take the first 30
pdf_files_30 = pdf_files[:30]
print(f"Found {len(pdf_files)} PDFs in repo, downloading first {len(pdf_files_30)}")

# 4. Download those 30 into a local folder
target_dir = Path("nist_publications_30pdfs")
target_dir.mkdir(parents=True, exist_ok=True)

local_paths = []
for fname in pdf_files_30:
    local_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=fname,
        local_dir=target_dir,
        force_download=False,
    )
    local_paths.append(local_path)
    print(f"Downloaded: {local_path}")

print(f"\nDone. Saved {len(local_paths)} PDFs under: {target_dir.resolve()}")
