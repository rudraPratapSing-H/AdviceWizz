import json
from http import HTTPStatus
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

from starlette.responses import Response
from langchain_core.prompts import ChatPromptTemplate
import os

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.schema import Document


MEMORY_FILE = "conversation_memory.json"
STYLE_FILE = "response_style.json"
FAISS_INDEX_PATH = "faiss_index2"
MEMORY_FAISS_INDEX_PATH = "memory_faiss_index"

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
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)
        return {}
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
    
def load_style():
    if not os.path.exists(STYLE_FILE):
        default = {"default": "You are a therapist"}
        with open(STYLE_FILE, "w", encoding="utf-8") as f:
            json.dump(default, f, ensure_ascii=False, indent=2)
        return default
    with open(STYLE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

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
        print("No user history to summarize")
        return
    
    conversation_text = "\n".join(
        [f"User: {m['query']}\nAssistant: {m['response']}" for m in user_history]
    )
    
    summary_prompt = f"""Summarize the following conversation in 2-3 sentences, 
    capturing the main topics and emotional tone:
    
    {conversation_text}
    
    Summary:"""
    
    summary = llm.invoke(summary_prompt)
    
    memory_doc = Document(
        page_content=summary,
        metadata={
            "user_id": user_id,
            "timestamp": str(datetime.now()),
            "original_length": len(user_history),
            "type": "conversation_summary"
        }
    )
    
    if os.path.exists(MEMORY_FAISS_INDEX_PATH):
        memory_vectorstore = FAISS.load_local(
            MEMORY_FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        memory_vectorstore.add_documents([memory_doc])
    else:
        memory_vectorstore = FAISS.from_documents([memory_doc], embeddings)
    
    memory_vectorstore.save_local(MEMORY_FAISS_INDEX_PATH)
    print(f"Memory summarized and stored for user: {user_id}")

def retrieve_semantic_memory(user_id: str, query: str, k: int = 3):
    """
    Retrieves relevant past conversation summaries using semantic search.
    """
    if not os.path.exists(MEMORY_FAISS_INDEX_PATH):
        print("No semantic memory index found")
        return []
    
    memory_vectorstore = FAISS.load_local(
        MEMORY_FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    docs = memory_vectorstore.similarity_search(query, k=k)
    filtered_docs = [doc for doc in docs if doc.metadata.get("user_id") == user_id]
    
    return [doc.page_content for doc in filtered_docs]

@router.post("/", dependencies=[])
def handle_event(data: EventSchema) -> Response:
    print(data)
    
    # Retrieve relevant context from FAISS
    docs = vectorstore.similarity_search(data.query, k=7)
    retrieved_contexts = []
    retrieved_meta = []
    for doc in docs:
        retrieved_contexts.append(doc.page_content)
        meta = doc.metadata or {}
        retrieved_meta.append(meta)
    
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
        full_context = f"Relevant past discussions:\n{chr(10).join(semantic_memories)}\n\n{full_context}"
    else:
        print("No semantic memories found")
    
    # Detect emotion
    emotion_prompt = f"""Based on this message, identify the user's emotional state:

    "{data.query}"

    Respond with: emotion name, intensity (1-10), and brief explanation."""
    emotion = llm.invoke(emotion_prompt)

    # Generate response
    prompt = ChatPromptTemplate.from_template("""
{style}
Keep it like a conversation between two humans.. the user might ask you questions or they might give you answer of your question and you have to follow up.
maximum 100 words.
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
        emotion=emotion
    )
    response = llm.invoke(prompt_str)

    # Update memory
    user_history.append({"query": data.query, "response": response})
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
            "memories": semantic_memories,
            "emotion": emotion,
            "retrieved_meta": retrieved_meta,
            "history": memory,
            "style": user_style
        }), 
        status_code=HTTPStatus.ACCEPTED,
    )
