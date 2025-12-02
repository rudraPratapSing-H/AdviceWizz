# create_index.py

import os
import re
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings  # Changed this line
from langchain_core.documents import Document
from dotenv import load_dotenv
import fitz  # PyMuPDF
from collections import Counter
import os
from pathlib import Path



# Load environment variables
load_dotenv()

# --- Configuration ---
# 1. Set the path to your folder containing the PDF files
BOOK_FOLDER_PATH = Path("books")
# 2. Set the path where you want to save the FAISS index
FAISS_INDEX_PATH = "faiss_index_gemini"

# 3. Using local Ollama embeddings
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
# ---------------------
class PDFStructureExtractor:
    def __init__(self, pdf_path):
        self.doc = fitz.open(pdf_path)
        self.body_font_size = 0

    def analyze_fonts(self):
        font_sizes = []
        for page_num in range(min(5, len(self.doc))):
            blocks = self.doc[page_num].get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_sizes.append(round(span["size"]))
        
        if font_sizes:
            self.body_font_size = Counter(font_sizes).most_common(1)[0][0]
            print(f"Detected Body Font Size: {self.body_font_size}pt")

    def extract_structured_text(self, page_number):
        """Extract text with structure markers (chapter, heading) from a page"""
        page = self.doc[page_number]
        blocks = page.get_text("dict")["blocks"]
        page_content = []
        
        for block in blocks:
            if block['type'] == 1: 
                page_content.append({"type": "image", "text": "[Visual: Image Present]"})
                continue
            
            if "lines" not in block:
                continue

            for line in block["lines"]:
                line_text = ""
                max_size = 0
                
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    
                    max_size = max(max_size, span["size"])
                    line_text += text + " "
                
                line_text = line_text.strip()
                if not line_text:
                    continue
                
                # Determine type based on font size
                if max_size > self.body_font_size + 2:
                    if "chapter" in line_text.lower():
                        page_content.append({"type": "chapter", "text": line_text})
                    else:
                        page_content.append({"type": "heading", "text": line_text})
                else:
                    page_content.append({"type": "body", "text": line_text})
        
        return page_content

    def process_book(self):
        """Process entire book and return structured content with metadata"""
        self.analyze_fonts()
        all_pages = []
        
        print(f"Processing {len(self.doc)} pages...")
        
        for page_num in range(len(self.doc)):
            page_content = self.extract_structured_text(page_num)
            all_pages.append({
                "page_number": page_num + 1,
                "content": page_content
            })
        
        return all_pages
    
def create_vector_store(append):
    """Loads PDF docs, extracts structure, creates embeddings, and saves them to FAISS."""
    print("Starting the structured index creation process...")

    if not os.path.exists(BOOK_FOLDER_PATH):
        print(f"Error: The folder '{BOOK_FOLDER_PATH}' does not exist.")
        return

    # Get all PDF files
    pdf_files = [f for f in os.listdir(BOOK_FOLDER_PATH) if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(BOOK_FOLDER_PATH, pdf_file)
        book_name = pdf_file
        
        print(f"\nProcessing: {book_name}")
        
        # Extract structured content
        extractor = PDFStructureExtractor(pdf_path)
        pages_data = extractor.process_book()
        
        # Track current chapter and heading
        current_chapter = None
        current_heading = None
        
        for page_data in pages_data:
            page_number = page_data["page_number"]
            page_text = ""
            
            for item in page_data["content"]:
                if item["type"] == "chapter":
                    # New chapter detected - reset heading
                    current_chapter = item["text"]
                    current_heading = None
                    page_text += f"\n## {item['text']}\n\n"
                    
                elif item["type"] == "heading":
                    # New heading detected
                    current_heading = item["text"]
                    page_text += f"\n### {item['text']}\n\n"
                    
                else:  # body or image
                    page_text += item["text"] + " "
            
            # Split page text into chunks
            if page_text.strip():
                doc = Document(page_content=page_text, metadata={
                    "book": book_name,
                    "page": page_number,
                    "chapter": current_chapter,
                    "heading": current_heading
                })
                
                chunks = text_splitter.split_documents([doc])
                
                # Preserve metadata in all chunks
                for chunk in chunks:
                    chunk.metadata = {
                        "book": book_name,
                        "page": page_number,
                        "chapter": current_chapter,
                        "heading": current_heading
                    }
                    final_documents.append(chunk)
        
        print(f"  ✓ Extracted {len([p for p in pages_data])} pages from {book_name}")

    print(f"\nTotal chunks created: {len(final_documents)}")
    print("Sample metadata:")
    if final_documents:
        print(f"  {final_documents[0].metadata}")
    
    # Save chunks with metadata to text file for debugging
    debug_file = "chunks_debug.txt"
    with open(debug_file, "w", encoding="utf-8") as f:
        f.write("="*100 + "\n")
        f.write("CHUNKS DEBUG FILE - All Chunks with Metadata\n")
        f.write("="*100 + "\n\n")
        
        for i, doc in enumerate(final_documents, 1):
            f.write(f"\n{'='*100}\n")
            f.write(f"CHUNK #{i}\n")
            f.write(f"{'='*100}\n")
            f.write(f"Book: {doc.metadata.get('book', 'N/A')}\n")
            f.write(f"Page: {doc.metadata.get('page', 'N/A')}\n")
            f.write(f"Chapter: {doc.metadata.get('chapter', 'N/A')}\n")
            f.write(f"Heading: {doc.metadata.get('heading', 'N/A')}\n")
            f.write(f"{'-'*100}\n")
            f.write(f"CONTENT:\n")
            f.write(f"{'-'*100}\n")
            f.write(doc.page_content)
            f.write(f"\n\n")
    
    print(f"✅ Debug file saved: {debug_file}")
    
    if append and os.path.exists(FAISS_INDEX_PATH):
        print("\nLoading existing FAISS index to append new vectors...")
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(final_documents)
    else:
        print("\nCreating new FAISS index from documents...")
        vectorstore = FAISS.from_documents(final_documents, embeddings)

    # Save the created index to disk
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"\n✅ FAISS index created and saved successfully at: {FAISS_INDEX_PATH}")




if __name__ == "__main__":
    if not os.path.exists(BOOK_FOLDER_PATH):
        # CHANGE: Updated error message for PDFs
        print(f"Error: The folder '{BOOK_FOLDER_PATH}' does not exist. Please create it and add your PDF files.")
    else:
        create_vector_store(append=False)