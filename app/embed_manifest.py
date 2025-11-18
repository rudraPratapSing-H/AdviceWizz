import json
import os
from langchain_ollama import OllamaEmbeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize embeddings model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

LIB_FILE = os.path.join(BASE_DIR, "library_manifest.json")
def embed_library_manifest():
    """Read library_manifest.json and add embedding vectors for each book description."""
    
    # Read the manifest file
    with open(LIB_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    print(f"Processing {len(manifest)} books...")
    
    # Add embeddings for each book description
    for book in manifest:
        description = book.get("description", "")
        if description:
            print(f"Embedding description for: {book['title']}")
            # Generate embedding vector for the description
            embedding_vector = embeddings.embed_query(description)
            book["embedding"] = embedding_vector
            print(f"  ✓ Generated {len(embedding_vector)}-dimensional vector")
        else:
            print(f"  ⚠ No description found for {book['title']}")
    
    # Save the updated manifest with embeddings
    with open(LIB_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✅ Successfully embedded all descriptions!")
    print(f"Updated {LIB_FILE} with embedding vectors")

if __name__ == "__main__":
    embed_library_manifest()
