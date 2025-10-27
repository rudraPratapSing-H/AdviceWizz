# AI Therapist Chatbot - System Design

A retrieval-augmented generation (RAG) chatbot with emotion detection, semantic memory, and multi-persona support for therapeutic conversations.

**[📊 View Interactive System Design on Canva](https://www.canva.com/design/DAG2-viFGjE/ff0dIOGQKadJdCUGZjcXRA/edit?utm_content=DAG2-viFGjE&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)**

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                             │
│  (Web UI / Mobile App / API Client)                             │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP/REST
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API GATEWAY LAYER                           │
│  FastAPI Server (main.py + router.py)                           │
│  - Authentication / Rate Limiting                                │
│  - Request Validation (Pydantic schemas)                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BUSINESS LOGIC LAYER                          │
│  endpoint.py - Chat Handler                                      │
│  ├─ Conversation Memory Management                               │
│  ├─ Emotion Detection                                            │
│  ├─ Context Retrieval (RAG)                                      │
│  ├─ Response Generation (LLM)                                    │
│  └─ Memory Summarization                                         │
└────────────┬──────────────────────┬─────────────────────────────┘
             │                      │
             ▼                      ▼
┌─────────────────────┐  ┌──────────────────────────────────────┐
│  STORAGE LAYER      │  │     AI/ML LAYER                      │
│                     │  │                                      │
│ ┌─────────────────┐ │  │ ┌──────────────────────────────────┐│
│ │ JSON Files      │ │  │ │ LLM (Ollama - llama3:8b)         ││
│ │ - conversation_ │ │  │ │ - Response Generation            ││
│ │   memory.json   │ │  │ │ - Emotion Detection              ││
│ │ - response_     │ │  │ │ - Memory Summarization           ││
│ │   style.json    │ │  │ └──────────────────────────────────┘│
│ └─────────────────┘ │  │                                      │
│                     │  │ ┌──────────────────────────────────┐│
│ ┌─────────────────┐ │  │ │ Embeddings (OllamaEmbeddings)    ││
│ │ Vector Stores   │ │  │ │ - mxbai-embed-large              ││
│ │ - faiss_index2/ │◄─┼──┤ │ - Text-to-vector conversion      ││
│ │   (Knowledge)   │ │  │ └──────────────────────────────────┘│
│ │ - memory_faiss_ │ │  │                                      │
│ │   index/        │ │  │ ┌──────────────────────────────────┐│
│ │   (Summaries)   │ │  │ │ FAISS Vector Search              ││
│ └─────────────────┘ │  │ │ - Semantic retrieval (k=7)       ││
│                     │  │ │ - Memory retrieval (k=2)         ││
│ ┌─────────────────┐ │  │ └──────────────────────────────────┘│
│ │ Document Store  │ │  └──────────────────────────────────────┘
│ │ - books/ (PDFs) │ │
│ └─────────────────┘ │
└─────────────────────┘
```

---

## 📊 Data Flow Diagram

```
User Query (user_id, query, style_type_id)
    │
    ▼
┌────────────────────────────────────┐
│  1. API Gateway (FastAPI)          │
│     - Validate EventSchema         │
└────────────┬───────────────────────┘
             ▼
┌────────────────────────────────────┐
│  2. Load Resources                 │
│     - conversation_memory.json     │
│     - response_style.json          │
└────────────┬───────────────────────┘
             ▼
┌────────────────────────────────────┐
│  3. Emotion Detection              │
│     - Analyze user's emotional     │
│       state via LLM                │
└────────────┬───────────────────────┘
             ▼
┌────────────────────────────────────┐
│  4. Context Retrieval (RAG)        │
│     - FAISS similarity search      │
│     - Retrieve top 7 chunks        │
│     - Extract metadata (book/page) │
└────────────┬───────────────────────┘
             ▼
┌────────────────────────────────────┐
│  5. Semantic Memory Retrieval      │
│     - Search conversation summaries│
│     - Retrieve top 2 relevant      │
│       past discussions             │
└────────────┬───────────────────────┘
             ▼
┌────────────────────────────────────┐
│  6. Build Context                  │
│     - Combine: style + knowledge + │
│       history + semantic memory +  │
│       emotion                      │
└────────────┬───────────────────────┘
             ▼
┌────────────────────────────────────┐
│  7. Generate Response              │
│     - LLM invocation with prompt   │
│     - Max 100 words                │
└────────────┬───────────────────────┘
             ▼
┌────────────────────────────────────┐
│  8. Update Memory                  │
│     - Append to user history       │
│     - Keep last 10 exchanges       │
│     - Save to JSON                 │
└────────────┬───────────────────────┘
             ▼
┌────────────────────────────────────┐
│  9. Conditional Summarization      │
│     - Every 5 exchanges:           │
│       * Summarize conversation     │
│       * Store in memory_faiss_index│
└────────────┬───────────────────────┘
             ▼
┌────────────────────────────────────┐
│ 10. Return Response                │
│     - response text                │
│     - emotion analysis             │
│     - semantic memories            │
│     - metadata (book/page)         │
│     - conversation history         │
└────────────────────────────────────┘
```

---

## 🧩 Component Details

### **1. API Gateway Layer**
**Files:** `main.py`, `router.py`

- **FastAPI Application** - HTTP server
- **Router** - Routes `/chat/` POST requests
- **Pydantic Validation** - `EventSchema` (user_id, query, style_type_id)

### **2. Business Logic Layer**
**File:** `endpoint.py`

| Function | Purpose |
|----------|---------|
| `load_memory()` | Load user conversation history from JSON |
| `save_memory()` | Persist updated conversation history |
| `load_style()` | Load AI persona definitions |
| `summarize_and_store_memory()` | Create summaries and store in FAISS |
| `retrieve_semantic_memory()` | Retrieve past conversation summaries |
| `handle_event()` | Main endpoint handler |

### **3. Storage Layer**

#### JSON Files
- **`conversation_memory.json`** - Recent chat history (last 10 exchanges per user)
- **`response_style.json`** - AI persona definitions (hannibal, kneeting, ANDY, etc.)

#### Vector Stores (FAISS)
- **`faiss_index2/`** - Knowledge base from PDF books
  - Contains: Text chunks + metadata (book name, page number)
  - Purpose: RAG context retrieval
  
- **`memory_faiss_index/`** - Conversation summaries
  - Contains: Summarized conversations + metadata (user_id, timestamp)
  - Purpose: Long-term semantic memory

#### Document Store
- **`books/`** - Source PDF files
- Indexed by `create_index2.py`

### **4. AI/ML Layer**

#### LLM (Ollama - llama3:8b)
- Response generation
- Emotion detection
- Memory summarization

#### Embeddings (OllamaEmbeddings - mxbai-embed-large)
- Convert text to vectors
- Enable semantic search in FAISS

#### FAISS Vector Search
- Similarity search in knowledge base (k=7)
- Semantic memory retrieval (k=2)

---

## 🚀 Features

### ✅ Multi-Persona Support
Choose from different AI personalities:
- `hannibal` - Dr. Hannibal Lecter (analytical, unsettling)
- `kneeting` - John Keating (inspirational, poetic)
- `ANDY` - Dr. Victor Blaine (brutally honest)
- `default` - Standard therapist

### ✅ Emotion Detection
Analyzes user's emotional state:
- Emotion name (e.g., anxious, sad, hopeful)
- Intensity (1-10 scale)
- Brief explanation

### ✅ RAG (Retrieval-Augmented Generation)
- Retrieves relevant knowledge from PDF books
- Returns top 7 chunks with metadata (book name, page number)
- Grounds responses in factual content

### ✅ Semantic Memory
- Stores conversation summaries every 5 exchanges
- Retrieves relevant past discussions using semantic search
- Maintains long-term context across sessions

### ✅ Short-Term Memory
- Keeps last 10 exchanges per user
- Included in every prompt for context continuity

---

## 📁 Project Structure

```
fastapi-tutorial/
├── app/
│   ├── main.py                      # FastAPI app entry point
│   ├── router.py                    # Route definitions
│   ├── endpoint.py                  # Main business logic
│   ├── create_index2.py             # PDF indexing script
│   ├── conversation_memory.json     # Chat history storage
│   ├── response_style.json          # AI persona definitions
│   ├── books/                       # Source PDF files
│   ├── faiss_index2/                # Knowledge base vectors
│   │   └── index.faiss
│   └── memory_faiss_index/          # Conversation summaries
│       └── index.faiss
├── pyproject.toml                   # Dependencies
└── README.md                        # This file
```

---

## 🔧 Setup & Installation

### Prerequisites
- Python 3.9+
- Ollama with llama3:8b and mxbai-embed-large models

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd fastapi-tutorial

# Install dependencies
pip install -r requirements.txt
# or
poetry install

# Start Ollama
ollama serve

# Pull required models
ollama pull llama3:8b
ollama pull mxbai-embed-large

# Index your PDF books
python app/create_index2.py

# Run the server
uvicorn app.main:app --reload
```

---

## 🌐 API Usage

### Endpoint: `POST /chat/`

**Request Body:**
```json
{
  "user_id": "user123",
  "query": "I'm feeling overwhelmed with work",
  "style_type_id": "hannibal"
}
```

**Response:**
```json
{
  "message": "Data received!",
  "response": "Ah, the familiar weight of expectation...",
  "emotion": "anxious, intensity: 7/10 - User expresses stress...",
  "memories": [
    "Summary of past conversation about work stress..."
  ],
  "retrieved_meta": [
    {"book": "psychology.pdf", "page": 42}
  ],
  "history": {
    "user123": [
      {"query": "...", "response": "..."}
    ]
  },
  "style": "You are Dr. Hannibal Lecter..."
}
```

---

## 🔮 Future Enhancements

### Scalability
- [ ] Replace JSON with PostgreSQL/MongoDB
- [ ] Add Redis caching layer
- [ ] Implement async LLM calls
- [ ] Add Celery for background tasks
- [ ] Microservices architecture

### Security
- [ ] JWT authentication
- [ ] Rate limiting
- [ ] Data encryption
- [ ] Input sanitization

### Features
- [ ] Multi-language support
- [ ] Voice input/output
- [ ] Session management
- [ ] Analytics dashboard
- [ ] Export conversation history

### Monitoring
- [ ] Logging (Sentry)
- [ ] Performance metrics (Prometheus)
- [ ] LLM token usage tracking

---

## 📄 License

[Your License Here]

---

## 👥 Contributors

[Your Name]

---

## 🙏 Acknowledgments

- LangChain for RAG framework
- Ollama for local LLM deployment
- FAISS for vector search
- FastAPI for web framework