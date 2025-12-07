import sys
from agentic_rag import agentic_rag
from vector_store import load_index

def main():
    print("=" * 70)
    print("Mini Agentic RAG - Interactive Chat")
    print("=" * 70)
    print("Loading FAISS index...")
    
    try:
        index, metadata = load_index()
        print(f"Loaded index with {len(metadata)} chunks")
    except FileNotFoundError:
        print("Error: FAISS index not found. Please run:")
        print("  python vector_store.py --build")
        sys.exit(1)
    
    print("\nType your questions about NIST cybersecurity (or 'quit'/'exit' to stop)")
    print("-" * 70)
    
    while True:
        try:
            # Get user input
            user_input = input("\n[You]: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            # Skip empty input
            if not user_input:
                continue
            
            # Process query with agentic RAG
            print("\n[Agent]: Processing...\n")
            answer = agentic_rag(user_input)
            print(f"[Agent]: {answer}")
            print("-" * 70)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()
