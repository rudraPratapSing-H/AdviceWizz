# create_index.py

import os
import re
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

# --- Configuration ---
# 1. Set the path to your folder containing the PDF files
BOOK_FOLDER_PATH = "books" # Example path

# 2. Set the path where you want to save the FAISS index
FAISS_INDEX_PATH = "faiss_index2"

# 3. Choose your embedding model (uncomment one)
# --- Option A: Local & Private (Ollama) ---
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# --- Option B: Cloud & Powerful (OpenAI) ---
# embeddings = OpenAIEmbeddings()
# ---------------------

def create_vector_store(append):
    """Loads PDF docs, splits them, creates embeddings, and saves them to FAISS."""
    print("Starting the index creation process...")

    # Load all .pdf files from the specified directory
    loader = PyPDFDirectoryLoader(BOOK_FOLDER_PATH) # <-- CHANGE: Using PyPDFDirectoryLoader
    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")

    # Split the documents into smaller, manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = []

    page_regex = re.compile(r'Page\s*[:#]?\s*(\d{1,5})', re.IGNORECASE)

    for doc in docs:
        # Get book name from source metadata
        source = doc.metadata.get("source", "")
        book_name = os.path.basename(source) if source else "unknown.pdf"

        # Try to get page number from metadata, else use regex on content
        page_number = (
            doc.metadata.get("page")
            or doc.metadata.get("page_number")
            or doc.metadata.get("pageno")
            or None
        )
        if page_number is None:
            match = page_regex.search(doc.page_content)
            if match:
                try:
                    page_number = int(match.group(1))
                except Exception:
                    page_number = None

        # Split the document into chunks
        chunks = text_splitter.split_documents([doc])
        for chunk in chunks:
            meta = chunk.metadata or {}
            meta["book"] = book_name
            meta["page"] = page_number
            chunk.metadata = meta
            final_documents.append(chunk)

    print(f"Split documents into {len(final_documents)} chunks (with book/page metadata).")
    
    if append and os.path.exists(FAISS_INDEX_PATH):
        print("Loading existing FAISS index to append new vectors...")
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(final_documents)
    else:
        print("Creating new FAISS index from documents...")
        vectorstore = FAISS.from_documents(final_documents, embeddings)

    # Save the created index to disk
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"FAISS index created and saved successfully at: {FAISS_INDEX_PATH}")




if __name__ == "__main__":
    if not os.path.exists(BOOK_FOLDER_PATH):
        # CHANGE: Updated error message for PDFs
        print(f"Error: The folder '{BOOK_FOLDER_PATH}' does not exist. Please create it and add your PDF files.")
    else:
        create_vector_store(append=False)