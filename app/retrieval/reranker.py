from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

def get_local_reranker(top_n=10):
    """
    Initializes a local HuggingFace Cross-Encoder for re-ranking.
    Downloads the model to the local machine on first run (~90MB).
    """
    print("Loading local Cross-Encoder model (ms-marco-MiniLM-L-6-v2)...")
    
    # This is a tiny, lightning-fast cross-encoder model
    model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Wrap it in Langchain's Reranker interface
    compressor = CrossEncoderReranker(model=model, top_n=top_n)
    
    return compressor
