import os
from pathlib import Path
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
# Get the absolute path of the directory containing this script (markdownParser)
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Resolve paths relative to the script's location, so it works no matter where you run it from!
BOOK_FOLDER_PATH = (SCRIPT_DIR / ".." / "books").resolve()
FAISS_INDEX_PATH = str((SCRIPT_DIR / ".." / "faiss_index_markdown").resolve())
DEBUG_FILE_PATH = str((SCRIPT_DIR / "markdown_chunks_debug.txt").resolve())

# Initialize embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

def create_markdown_vector_store(append=False):
    print("Starting Markdown-based index creation process...")

    if not os.path.exists(BOOK_FOLDER_PATH):
        print(f"Error: The folder '{BOOK_FOLDER_PATH}' does not exist.")
        return

    pdf_files = [f for f in os.listdir(BOOK_FOLDER_PATH) if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files.")

    # We split by Markdown headers to keep sections together
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # We use a character splitter as a fallback for massive sections
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    final_documents = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(BOOK_FOLDER_PATH, pdf_file)
        book_name = pdf_file
        
        print(f"\nProcessing: {book_name}")
        
        # State variables to track headers across pages!
        current_chapter = None
        current_heading = None
        current_sub_heading = None
        
        try:
            # page_chunks=True returns a list of dictionaries per page
            # This allows us to preserve page numbers easily!
            pages_data = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
            
            for page_data in pages_data:
                page_text = page_data.get('text', '')
                page_num = page_data.get('metadata', {}).get('page', 0)
                
                if not page_text.strip():
                    continue
                    
                # 1. Split the page text by Markdown headers
                md_splits = markdown_splitter.split_text(page_text)
                
                # 2. Add book and page metadata to each split, and track headers across pages
                for split in md_splits:
                    split.metadata['book'] = book_name
                    split.metadata['page'] = page_num
                    
                    # Extract local headers from this specific chunk
                    local_h1 = split.metadata.pop('Header 1', None)
                    local_h2 = split.metadata.pop('Header 2', None)
                    local_h3 = split.metadata.pop('Header 3', None)
                    
                    # Update our global state. If a higher-level header appears, reset the lower levels!
                    if local_h1:
                        current_chapter = local_h1
                        current_heading = None
                        current_sub_heading = None
                    if local_h2:
                        current_heading = local_h2
                        current_sub_heading = None
                    if local_h3:
                        current_sub_heading = local_h3
                        
                    # Apply the active global state to the chunk's metadata
                    if current_chapter:
                        split.metadata['chapter'] = current_chapter
                    if current_heading:
                        split.metadata['heading'] = current_heading
                    if current_sub_heading:
                        split.metadata['sub_heading'] = current_sub_heading
                
                # 3. Apply the fallback character splitter to ensure chunks aren't too large
                chunks = text_splitter.split_documents(md_splits)
                final_documents.extend(chunks)
                
            print(f"  ✓ Processed {len(pages_data)} pages from {book_name}")
            
        except Exception as e:
            print(f"  ❌ Error processing {book_name}: {e}")

    print(f"\nTotal chunks created: {len(final_documents)}")
    
    # Save chunks for debugging
    with open(DEBUG_FILE_PATH, "w", encoding="utf-8") as f:
        f.write("="*100 + "\n")
        f.write("MARKDOWN CHUNKS DEBUG FILE\n")
        f.write("="*100 + "\n\n")
        for i, doc in enumerate(final_documents, 1):
            f.write(f"\n{'='*100}\nCHUNK #{i}\n{'='*100}\n")
            f.write(f"Metadata: {doc.metadata}\n")
            f.write(f"{'-'*100}\nCONTENT:\n{'-'*100}\n")
            f.write(doc.page_content)
            f.write(f"\n\n")
            
    print(f"✅ Debug file saved: {DEBUG_FILE_PATH}")

    if not final_documents:
        print("No valid documents found to index. Exiting.")
        return

    # Create or update FAISS index
    if append and os.path.exists(FAISS_INDEX_PATH):
        print("\nLoading existing FAISS index to append new vectors...")
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(final_documents)
    else:
        print("\nCreating new FAISS index from documents...")
        vectorstore = FAISS.from_documents(final_documents, embeddings)

    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"\n✅ FAISS index created and saved successfully at: {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    create_markdown_vector_store(append=False)
