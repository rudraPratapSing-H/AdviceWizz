import json
from http import HTTPStatus
from datetime import datetime
import os

from fastapi import APIRouter
from pydantic import BaseModel

from starlette.responses import Response, FileResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# Get the directory where this file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MEMORY_FILE = os.path.join(BASE_DIR, "conversation_memory.json")
STYLE_FILE = os.path.join(BASE_DIR, "response_style.json")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index2")
MEMORY_FAISS_INDEX_PATH = os.path.join(BASE_DIR, "memory_faiss_index")
LIBRARY_MANIFEST_FILE = os.path.join(BASE_DIR, "library_manifest.json")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = FAISS.load_local(
    FAISS_INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

router = APIRouter()
llm = OllamaLLM(model="llama3:8b")

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
    
def load_style():
    if not os.path.exists(STYLE_FILE):
        return {}
    with open(STYLE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def load_library_manifest():
    if not os.path.exists(LIBRARY_MANIFEST_FILE):
        return []
    with open(LIBRARY_MANIFEST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

class EventSchema(BaseModel):
    user_id: str
    query: str  
    style_type_id: str

def summarize_and_store_memory(user_id: str, user_history: list):
    """
    Summarizes the user's conversation history and stores it in FAISS
    for semantic memory retrieval.
    """
    if not user_history:
        return
    
    conversation_text = "\n".join(
        [f"User: {m['query']}\nAssistant: {m['response']}" for m in user_history]
    )
    
    summary_prompt = f"""Summarize the following conversation concisely, 
    highlighting key topics, emotions, and important details:
    
    {conversation_text}
    """
    summary = llm.invoke(summary_prompt)
    
    os.makedirs(MEMORY_FAISS_INDEX_PATH, exist_ok=True)
    
    index_path = os.path.join(MEMORY_FAISS_INDEX_PATH, user_id)
    
    doc = Document(
        page_content=summary,
        metadata={"user_id": user_id, "timestamp": datetime.now().isoformat()}
    )
    
    if os.path.exists(index_path):
        memory_vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        memory_vectorstore.add_documents([doc])
    else:
        memory_vectorstore = FAISS.from_documents([doc], embeddings)
    
    memory_vectorstore.save_local(index_path)
    print(f"Memory summary stored for user {user_id}")

def route_to_book(query: str) -> str:
    """
    Routes the query to the most relevant book by comparing query embedding
    with book description embeddings.
    Returns the book ID (filename) of the most relevant book.
    """
    library_manifest = load_library_manifest()
    
    if not library_manifest:
        print("No library manifest found, using all books")
        return None
    
    # Get query embedding
    query_embedding = embeddings.embed_query(query)
    
    # Calculate cosine similarity with each book description
    from numpy import dot
    from numpy.linalg import norm
    
    best_match = None
    best_similarity = -1
    
    for book in library_manifest:
        if "embedding" not in book:
            print(f"No embedding found for {book['title']}, skipping")
            continue
        
        book_embedding = book["embedding"]
        
        # Cosine similarity
        similarity = dot(query_embedding, book_embedding) / (norm(query_embedding) * norm(book_embedding))
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = book["id"]
    
    print(f"Routed to book: {best_match} (similarity: {best_similarity:.4f})")
    return best_match

def retrieve_semantic_memory(user_id: str, query: str, k: int = 3):
    """
    Retrieves relevant semantic memories for a user based on the query.
    """
    index_path = os.path.join(MEMORY_FAISS_INDEX_PATH, user_id)
    
    if not os.path.exists(index_path):
        return []
    
    try:
        memory_vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        docs = memory_vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    except Exception as e:
        print(f"Error retrieving semantic memory: {e}")
        return []


@router.post("/", dependencies=[])
def handle_event(data: EventSchema) -> Response:
    print(data)
    
    # Step 1: Route query to most relevant book
    target_book_id = route_to_book(data.query)
    
    # Step 2: Retrieve relevant context from FAISS, filtered by book
    if target_book_id:
        # Filter search to only the routed book
        all_docs = vectorstore.similarity_search(data.query, k=50)  # Get more docs to filter
        docs = [doc for doc in all_docs if doc.metadata.get("book") == target_book_id][:7]
        print(f"Found {len(docs)} chunks from {target_book_id}")
    else:
        # No routing, search all books
        docs = vectorstore.similarity_search(data.query, k=7)
    
    retrieved_contexts = []
    retrieved_sources = []
    for doc in docs:
        retrieved_contexts.append(doc.page_content)
        meta = doc.metadata or {}
        retrieved_sources.append({
            "book": meta.get("book", "Unknown"),
            "page": meta.get("page", "N/A"),
            "chunk": doc.page_content
        })
    
    retrieved_context = "\n\n".join(retrieved_contexts)
    
    memory = load_memory()
    style = load_style()

    user_history = memory.get(data.user_id, [])
    user_style = style.get(data.style_type_id, "You are a therapist")

    history_text = "\n".join(
        [f"User: {m['query']}\nAssistant: {m['response']}" for m in user_history]
    )
    full_context = f"{retrieved_context}\n\nPrevious conversation:\n{history_text}"
    
    semantic_memories = retrieve_semantic_memory(data.user_id, data.query, k=2)
    if semantic_memories:
        print("Semantic memories found")
        memories = f"Relevant past discussions:\n{chr(10).join(semantic_memories)}"
    else:
        print("No semantic memories found")
        memories = "No previous relevant discussions found."
    
    # Detect emotion
    emotion_prompt = f"""Based on this message, identify the user's emotional state:

    "{data.query}"

    Respond with: emotion name, intensity (1-10), and brief explanation."""
    emotion = llm.invoke(emotion_prompt)

    # Generate response
    prompt = ChatPromptTemplate.from_template("""
{style}
Keep it like a conversation between two humans.. the user might ask you questions or they might give you answer of your question and you have to follow up.
maximum 100 words and keep the size of your response concise dont bluff. 
Use these memories to optimize the response:- {memories}
<context>
{context}
</context>
User emotion state:- {emotion}. Now you give response addressing user's emotion.

Question: {query}
""")
    prompt_str = prompt.format(
        style=user_style,
        context=full_context,
        query=data.query,
        emotion=emotion,
        memories=memories
    )
    response = llm.invoke(prompt_str)

    # Update memory
    user_history.append({
        "query": data.query, 
        "response": response,
        "therapist": data.style_type_id
    })
    memory[data.user_id] = user_history[-10:]
    save_memory(memory)
    
    # Summarize and store in FAISS every 5 exchanges
    if len(user_history) % 5 == 0:
        print(f"Summarizing memory for user {data.user_id} (total: {len(user_history)} exchanges)")
        summarize_and_store_memory(data.user_id, user_history)
    
    return Response(
        content=json.dumps({
            "message": "Data received!",
            "response": response,
            "memories": memories,
            "emotion": emotion,
            "retrieved_sources": retrieved_sources,
            "history": memory,
            "style": user_style
        }), 
        status_code=HTTPStatus.ACCEPTED,
    )

@router.get("/history/{user_id}", dependencies=[])
def get_chat_history(user_id: str) -> Response:
    """
    Get chat history for a specific user.
    """
    memory = load_memory()
    user_history = memory.get(user_id, [])
    
    return Response(
        content=json.dumps({
            user_id: user_history
        }),
        status_code=HTTPStatus.OK,
        media_type="application/json"
    )
