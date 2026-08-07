from langchain_core.documents import Document

def get_parent_key(doc: Document) -> tuple:
    """
    Identifies the parent of a chunk. 
    Prioritizes 'heading', falls back to 'chapter'.
    """
    book = doc.metadata.get('book')
    
    heading = doc.metadata.get('heading')
    if heading:
        return (book, 'heading', heading)
        
    chapter = doc.metadata.get('chapter')
    if chapter:
        return (book, 'chapter', chapter)
        
    return (book, 'none', None)

def expand_context(top_chunks: list, vectorstore) -> list:
    """
    Takes a list of retrieved chunks and expands them into their full 
    parent sections (Headings) by querying the FAISS docstore.
    Ensures ZERO redundancy.
    """
    if not vectorstore:
        return top_chunks
        
    seen_parents = set()
    expanded_docs = []
    
    # We grab all documents from the FAISS docstore. 
    # Because they were inserted in order, iterating over them preserves reading order!
    all_docs = list(vectorstore.docstore._dict.values())
    
    for chunk in top_chunks:
        parent_key = get_parent_key(chunk)
        
        # If the chunk has no parent header/chapter, just return it as is to be safe
        if parent_key[1] == 'none':
            if chunk.page_content not in [d.page_content for d in expanded_docs]:
                expanded_docs.append(chunk)
            continue
            
        # If we already expanded this parent for a previous chunk, skip it! (Zero Redundancy)
        if parent_key in seen_parents:
            continue
            
        seen_parents.add(parent_key)
        
        # Extract all sibling chunks from the docstore that share this exact parent
        sibling_chunks = []
        for doc in all_docs:
            if get_parent_key(doc) == parent_key:
                sibling_chunks.append(doc.page_content)
                
        # Combine them into one large "Super-Chunk"
        combined_text = "\n\n".join(sibling_chunks)
        
        # Create the new document
        super_doc = Document(
            page_content=combined_text,
            metadata={
                "book": chunk.metadata.get("book"),
                "chapter": chunk.metadata.get("chapter"),
                "heading": chunk.metadata.get("heading"),
                "expanded": True
            }
        )
        expanded_docs.append(super_doc)
        
    return expanded_docs
