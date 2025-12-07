import json
import sys
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

# Import from local modules
from minirag import azure_embed
from document_processor import load_chunks, process_all_pdfs, save_chunks


# Paths
CHUNKS_PATH = Path("chunks.json")
INDEX_PATH = Path("faiss_index.bin")
METADATA_PATH = Path("chunk_metadata.json")

# Embedding config
EMBEDDING_DIM = 1536  # text-embedding-ada-002 dimension
BATCH_DELAY = 0.1  # Delay between API calls to avoid rate limits


def embed_with_retry(text: str, max_retries: int = 3) -> Optional[list[float]]:
    for attempt in range(max_retries):
        try:
            return azure_embed(text)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 1.0  # Exponential backoff
                print(f"  Retry {attempt + 1}/{max_retries} in {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return None


# IndexFlatL2 is good for exact matching and high-quiality semantic search, even tho it can be slower for retrival sometimes
def build_index(chunks: list[dict]) -> tuple[faiss.IndexFlatL2, list[dict]]:
    embeddings = []
    metadata = []
    
    print(f"Embedding {len(chunks)} chunks...")
    
    for i, chunk in enumerate(chunks):
        embedding = embed_with_retry(chunk["text"])
        
        if embedding is None:
            print(f"  [SKIP] Chunk {i}: embedding failed")
            continue
        
        embeddings.append(embedding)
        metadata.append({
            "chunk_id": chunk["chunk_id"],
            "filename": chunk["filename"],
            "page_num": chunk["page_num"],
            "text": chunk["text"]
        })
        
        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(chunks)}] embedded")
        
        time.sleep(BATCH_DELAY)  # Rate limit protection
    
    # Create FAISS index
    embeddings_np = np.array(embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings_np)
    
    print(f"\nBuilt index with {index.ntotal} vectors")
    return index, metadata


def save_index(index: faiss.IndexFlatL2, metadata: list[dict]):
    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved index to {INDEX_PATH} and metadata to {METADATA_PATH}")


def load_index() -> tuple[faiss.IndexFlatL2, list[dict]]:
    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Loaded index with {index.ntotal} vectors")
    return index, metadata


# top_k is number of similar chunks to retrieve, here i kept it 2 so that agent has some room to think about, increasing it may increase the hallucinations
def retrieve(query: str, top_k: int = 2, index: faiss.IndexFlatL2 = None, 
             metadata: list[dict] = None) -> list[dict]:
    if index is None or metadata is None:
        index, metadata = load_index()
    
    # Embed query
    query_embedding = azure_embed(query)
    query_np = np.array([query_embedding], dtype=np.float32)
    
    # Search
    distances, indices = index.search(query_np, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        
        chunk_meta = metadata[idx]
        results.append({
            "chunk_id": chunk_meta["chunk_id"],
            "filename": chunk_meta["filename"],
            "page_num": chunk_meta["page_num"],
            "text": chunk_meta["text"],
            "score": float(distances[0][i]),
            "citation": f"[{chunk_meta['filename']}, Page {chunk_meta['page_num']}]"
        })
    
    return results


def format_context(results: list[dict]) -> str:
    if not results:
        return "No relevant information found."
    
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(
            f"[Source {i}] {r['citation']}\n{r['text']}\n"
        )
    
    return "\n---\n".join(context_parts)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--build":
        # Build index from chunks
        pdf_dir = Path("nist_publications_clean_10")
        if not pdf_dir.exists():
            print(f"Error: PDF directory not found: {pdf_dir.resolve()}")
            sys.exit(1)
        
        # Process PDFs if chunks don't exist
        if not CHUNKS_PATH.exists():
            chunks = process_all_pdfs(pdf_dir)
            save_chunks(chunks, CHUNKS_PATH)
        else:
            chunks = load_chunks(CHUNKS_PATH)
        
        # Build and save index
        index, metadata = build_index(chunks)
        save_index(index, metadata)
        
    elif len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Test retrieval
        query = sys.argv[2] if len(sys.argv) > 2 else "What is zero trust architecture?"
        print(f"\nQuery: {query}\n")
        
        results = retrieve(query, top_k=3)
        print(format_context(results))
        
    else:
        print("Usage:")
        print("  python vector_store.py --build     # Build index from PDFs")
        print("  python vector_store.py --test <query>  # Test retrieval")
