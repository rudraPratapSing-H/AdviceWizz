import os
import json
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from app.retrieval.reranker import get_local_reranker
from app.retrieval.context_expander import expand_context
from numpy import dot
from numpy.linalg import norm

# Setup paths (BASE_DIR is the app folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index_markdown")
LIBRARY_MANIFEST_FILE = os.path.join(BASE_DIR, "Json", "library_manifest.json")

print("Initializing Search Engine (Hybrid Search: FAISS + BM25)...")

# Initialize embeddings and FAISS
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

try:
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("FAISS vectorstore loaded successfully.")
    
    # Initialize BM25 from FAISS documents
    docs = list(vectorstore.docstore._dict.values())
    print(f"Building BM25 Retriever with {len(docs)} documents... (This takes a moment)")
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 100
    
    faiss_retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.60, "k": 100}
    )
    
    # Create the Hybrid Search Retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.3, 0.7] # 30% keyword, 70% vector similarity
    )
    print("Ensemble Retriever initialized.")
    
    # Wrap it in the Local Cross-Encoder Reranker
    local_compressor = get_local_reranker(top_n=10)
    final_retriever = ContextualCompressionRetriever(
        base_compressor=local_compressor,
        base_retriever=ensemble_retriever
    )
    print("Local Cross-Encoder Reranker initialized and wrapping Ensemble Retriever.")
    
except Exception as e:
    print(f"Warning: Could not load FAISS index from {FAISS_INDEX_PATH}. {e}")
    vectorstore = None
    final_retriever = None
    ensemble_retriever = None
    local_compressor = None

def load_library_manifest():
    if not os.path.exists(LIBRARY_MANIFEST_FILE):
        return []
    with open(LIBRARY_MANIFEST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def _retrieve_chunks_internal(query: str, target_chunks: int = 7, blacklisted_books: list = None, all_docs: list = None) -> list:
    """
    Internal recursive function to gather chunks by book relevance.
    """
    if not final_retriever:
        print("Error: Retriever not initialized.")
        return []
        
    if blacklisted_books is None:
        blacklisted_books = []
    
    # Fetch all docs once at the start using Hybrid Search + Local Re-ranker
    if all_docs is None:
        print(f"Performing Search & Rerank for: '{query}'")
        all_docs = final_retriever.invoke(query)
    
    # Route to the most relevant book (excluding blacklisted ones)
    library_manifest = load_library_manifest()
    
    if not library_manifest:
        print("No library manifest found, returning all docs")
        return all_docs[:target_chunks]
    
    # Get query embedding
    query_embedding = embeddings.embed_query(query)
    
    similarities = []
    
    for book in library_manifest:
        if "embedding" not in book or book["id"] in blacklisted_books:
            continue
        
        book_embedding = book["embedding"]
        similarity = dot(query_embedding, book_embedding) / (norm(query_embedding) * norm(book_embedding))
        similarities.append((book["id"], similarity))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    if not similarities:
        print("No more books to search, returning collected docs")
        return all_docs[:target_chunks]
    
    best_book = similarities[0][0]
    print(f"Searching in book: {best_book} (similarity: {similarities[0][1]:.4f})")
    
    # Get chunks from this book
    docs = [doc for doc in all_docs if doc.metadata.get("book") == best_book][:target_chunks]
    print(f"Found {len(docs)} chunks from {best_book}")
    
    # If we have enough chunks, return them
    if len(docs) >= target_chunks:
        return docs
    
    # Otherwise, recursively search the next book
    remaining_needed = target_chunks - len(docs)
    print(f"Need {remaining_needed} more chunks, recursing...")
    
    blacklisted_books.append(best_book)
    remaining_docs = _retrieve_chunks_internal(query, remaining_needed, blacklisted_books, all_docs)
    
    # Combine and return
    docs.extend(remaining_docs)
    return docs[:target_chunks]

def retrieve_chunks_recursive(query: str, target_chunks: int = 7) -> list:
    """
    The main entry point for the endpoint.
    Retrieves the best chunks and then magically expands them into their full parent Headings!
    """
    raw_chunks = _retrieve_chunks_internal(query, target_chunks)
    
    # Expand the chunks into full Headings!
    print("Expanding retrieved chunks into their parent Headings...")
    expanded_chunks = expand_context(raw_chunks, vectorstore)
    
    print(f"Successfully expanded into {len(expanded_chunks)} unified 'Super-Chunks'.")
    return expanded_chunks

def retrieve_for_keywords(keywords: list, target_chunks: int = 10) -> list:
    """
    Takes an array of keyword strings, searches ALL of them via Hybrid Search,
    combines and deduplicates the raw chunks, Re-ranks the massive pile using the Cross-Encoder,
    and then Expands the absolute best top 10 chunks into full Headings.
    """
    if not ensemble_retriever:
        print("Error: Retriever not initialized.")
        return []
        
    all_raw_docs = []
    seen_content = set()
    
    # 1. Fetch from Ensemble (BM25 + FAISS) for all keywords
    for kw in keywords:
        print(f"Fetching raw chunks for keyword: '{kw}'")
        docs = ensemble_retriever.invoke(kw)
        for d in docs:
            if d.page_content not in seen_content:
                seen_content.add(d.page_content)
                all_raw_docs.append(d)
                
    # 2. Re-rank the massive combined pile
    print(f"Re-ranking {len(all_raw_docs)} combined chunks...")
    combined_query = " ".join(keywords)
    
    if local_compressor:
        reranked_docs = local_compressor.compress_documents(documents=all_raw_docs, query=combined_query)
        best_chunks = list(reranked_docs)[:target_chunks]
    else:
        best_chunks = all_raw_docs[:target_chunks]
        
    # 3. Expand Context!
    print("Expanding best chunks into full Headings...")
    expanded_chunks = expand_context(best_chunks, vectorstore)
    
    print(f"Successfully expanded into {len(expanded_chunks)} unified 'Super-Chunks'.")
    return expanded_chunks
