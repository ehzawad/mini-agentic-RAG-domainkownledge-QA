import json
from pathlib import Path
from typing import Generator

import fitz  # pymupdf
import tiktoken


# Chunking configuration
CHUNK_SIZE = 750  # Target tokens per chunk (500-1000 range)
CHUNK_OVERLAP = 100  # Token overlap between chunks
ENCODING_NAME = "cl100k_base"  # Encoder for text-embedding-ada-002

# Initialize tokenizer
enc = tiktoken.get_encoding(ENCODING_NAME)


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    doc = fitz.open(str(pdf_path))
    pages = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        
        if text.strip():  # Only include pages with text
            pages.append({
                "filename": pdf_path.name,
                "page_num": page_num + 1,  # 1-indexed
                "text": text.strip()
            })
    
    doc.close()
    return pages


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    tokens = enc.encode(text)
    
    if len(tokens) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        # Move start forward by (chunk_size - overlap)
        start += chunk_size - overlap
        
        if start >= len(tokens):
            break
    
    return chunks


def process_pdf(pdf_path: Path) -> list[dict]:
    pages = extract_text_from_pdf(pdf_path)
    chunks = []
    chunk_id = 0
    
    for page in pages:
        page_chunks = chunk_text(page["text"])
        
        for chunk_text_content in page_chunks:
            chunks.append({
                "chunk_id": f"{pdf_path.stem}_chunk_{chunk_id}",
                "filename": page["filename"],
                "page_num": page["page_num"],
                "text": chunk_text_content,
                "token_count": count_tokens(chunk_text_content)
            })
            chunk_id += 1
    
    return chunks


def process_all_pdfs(pdf_dir: Path) -> list[dict]:
    all_chunks = []
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    
    print(f"Processing {len(pdf_files)} PDFs from {pdf_dir}")
    
    for i, pdf_path in enumerate(pdf_files):
        try:
            chunks = process_pdf(pdf_path)
            all_chunks.extend(chunks)
            print(f"  [{i+1}/{len(pdf_files)}] {pdf_path.name}: {len(chunks)} chunks")
        except Exception as e:
            print(f"  [ERROR] {pdf_path.name}: {e}")
    
    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks


def save_chunks(chunks: list[dict], output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(chunks)} chunks to {output_path}")


def load_chunks(input_path: Path) -> list[dict]:
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    # Process PDFs from clean_10 directory (text-extractable, first 10 pages)
    pdf_dir = Path("nist_publications_clean_10")
    output_path = Path("chunks.json")
    
    if not pdf_dir.exists():
        # Fall back to full PDFs if clean dir doesn't exist
        pdf_dir = Path("nist_publications_30pdfs")
    
    chunks = process_all_pdfs(pdf_dir)
    save_chunks(chunks, output_path)
    
    # Print sample chunk
    if chunks:
        print("\n--- Sample Chunk ---")
        sample = chunks[0]
        print(f"ID: {sample['chunk_id']}")
        print(f"Source: {sample['filename']} (Page {sample['page_num']})")
        print(f"Tokens: {sample['token_count']}")
        print(f"Text preview: {sample['text'][:200]}...")
