# create_index.py

import os
# The DirectoryLoader is replaced with PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
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

    # Process each PDF file and each page individually, attach book/page metadata to every chunk
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = []

    for filename in sorted(os.listdir(BOOK_FOLDER_PATH)):
        if not filename.lower().endswith(".pdf"):
            continue
        filepath = os.path.join(BOOK_FOLDER_PATH, filename)
        print(f"Processing file: {filepath}")

        # Load page-level documents for this PDF
        loader = PyPDFLoader(filepath)
        pages = loader.load()  # returns a Document per page (usually)
        print(f"  -> Loaded {len(pages)} pages from {filename}")

        # For each page, split into chunks and annotate with book/page metadata
        for page_index, page_doc in enumerate(pages, start=1):
            # split_documents returns a list of Document chunks for this page
            page_chunks = text_splitter.split_documents([page_doc])
            for chunk in page_chunks:
                meta = chunk.metadata or {}
                meta["book"] = filename
                meta["page"] = page_index
                meta["source"] = filepath
                chunk.metadata = meta
                final_documents.append(chunk)

    print(f"Split documents into {len(final_documents)} chunks (book/page metadata added).")
    
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