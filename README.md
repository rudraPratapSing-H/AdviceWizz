# AdviceWizz — AI Therapist Chatbot

A retrieval-augmented generation (RAG) chatbot with emotion detection, semantic memory, multi-persona support, and an advanced hybrid search pipeline for therapeutic conversations.

**[📊 View Interactive System Design on Canva](https://www.canva.com/design/DAG2-viFGjE/ff0dIOGQKadJdCUGZjcXRA/edit?utm_content=DAG2-viFGjE&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)**

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                               │
│  (Web UI / Mobile App / API Client)                                 │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ HTTP/REST
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        API GATEWAY LAYER                             │
│  FastAPI (main.py + router.py)                                      │
│  - CORS Middleware (localhost:3000)                                  │
│  - Request Validation (Pydantic — EventSchema)                      │
│  - GET /chat/history/{user_id}                                      │
│  - POST /chat/                                                      │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      BUSINESS LOGIC LAYER                            │
│  endpoint.py — Chat Handler                                         │
│  ├─ NLP: Query Optimization (nlp/query_optimizer.py)                │
│  ├─ Retrieval: Hybrid Search + Re-rank + Context Expansion          │
│  │    (retrieval/search_engine.py)                                  │
│  ├─ Semantic Memory Retrieval (memory_faiss_index/)                 │
│  ├─ Response Generation + Emotion Detection                         │
│  │    (generation/response_generator.py)                            │
│  ├─ Short-Term Memory Management (Json/conversation_memory.json)    │
│  └─ Long-Term Memory Summarization → FAISS                          │
└────────────┬───────────────────────────────┬────────────────────────┘
             │                               │
             ▼                               ▼
┌────────────────────────┐   ┌───────────────────────────────────────┐
│     STORAGE LAYER      │   │             AI / ML LAYER             │
│                        │   │                                       │
│ ┌────────────────────┐ │   │ ┌───────────────────────────────────┐ │
│ │ JSON Files         │ │   │ │ LLM (Ollama — llama3.1:8b)        │ │
│ │ - conversation_    │ │   │ │ - Response Generation             │ │
│ │   memory.json      │ │   │ │ - Emotion Detection               │ │
│ │ - response_        │ │   │ │ - Memory Summarization            │ │
│ │   style.json       │ │   │ │ - Query Optimization              │ │
│ │ - library_         │ │   │ └───────────────────────────────────┘ │
│ │   manifest.json    │ │   │                                       │
│ └────────────────────┘ │   │ ┌───────────────────────────────────┐ │
│                        │   │ │ Embeddings (mxbai-embed-large)    │ │
│ ┌────────────────────┐ │   │ │ - Text-to-vector conversion       │ │
│ │ Vector Stores      │ │   │ │ - Semantic routing via cosine sim │ │
│ │ - faiss_index_     │◄┼───┤ └───────────────────────────────────┘ │
│ │   markdown/        │ │   │                                       │
│ │   (Knowledge Base) │ │   │ ┌───────────────────────────────────┐ │
│ │ - memory_faiss_    │ │   │ │ Hybrid Search (EnsembleRetriever) │ │
│ │   index/{user_id}/ │ │   │ │ - BM25 Keyword Search   (30%)    │ │
│ │   (Summaries)      │ │   │ │ - FAISS Vector Search   (70%)    │ │
│ └────────────────────┘ │   │ │ - Score Threshold: 0.60, k=100   │ │
│                        │   │ └───────────────────────────────────┘ │
│ ┌────────────────────┐ │   │                                       │
│ │ Document Store     │ │   │ ┌───────────────────────────────────┐ │
│ │ - books/ (PDFs)    │ │   │ │ Cross-Encoder Reranker            │ │
│ │ - Indexed via      │ │   │ │ (ms-marco-MiniLM-L-6-v2)         │ │
│ │   markdown_        │ │   │ │ - Re-ranks top candidates         │ │
│ │   indexer.py       │ │   │ │ - Selects top_n=10 chunks        │ │
│ └────────────────────┘ │   │ └───────────────────────────────────┘ │
└────────────────────────┘   └───────────────────────────────────────┘
```

---

## 📊 Data Flow Diagram

```
User Query  (user_id, query, style_type_id)
    │
    ▼
┌──────────────────────────────────────────┐
│  1. API Gateway (FastAPI)                │
│     - Validate EventSchema              │
│     - Route POST /chat/                 │
└─────────────────┬────────────────────────┘
                  ▼
┌──────────────────────────────────────────┐
│  2. Query Optimization (NLP)             │
│     - LLM extracts core search concepts  │
│     - Returns array of keyword strings   │
│     e.g. ["anxiety", "work stress"]      │
└─────────────────┬────────────────────────┘
                  ▼
┌──────────────────────────────────────────┐
│  3. Batched Hybrid Retrieval             │
│     - Run BM25 + FAISS for EACH keyword  │
│     - Combine & deduplicate all results  │
│     - Score threshold: 0.60, k=100       │
└─────────────────┬────────────────────────┘
                  ▼
┌──────────────────────────────────────────┐
│  4. Cross-Encoder Re-Ranking             │
│     - Combined query = join(keywords)    │
│     - ms-marco-MiniLM-L-6-v2 scores     │
│       every candidate chunk              │
│     - Keeps top 10 most relevant         │
└─────────────────┬────────────────────────┘
                  ▼
┌──────────────────────────────────────────┐
│  5. Context Expansion (Super-Chunks)     │
│     - Map each top chunk to parent       │
│       Heading or Chapter in FAISS        │
│     - Merge all sibling chunks into one  │
│       unified "Super-Chunk"              │
│     - Zero redundancy (dedup by parent)  │
└─────────────────┬────────────────────────┘
                  ▼
┌──────────────────────────────────────────┐
│  6. Load User Context                    │
│     - conversation_memory.json (history) │
│     - response_style.json (persona)      │
└─────────────────┬────────────────────────┘
                  ▼
┌──────────────────────────────────────────┐
│  7. Semantic Memory Retrieval            │
│     - Search user's memory_faiss_index   │
│     - Retrieve top 2 relevant past       │
│       conversation summaries             │
└─────────────────┬────────────────────────┘
                  ▼
┌──────────────────────────────────────────┐
│  8. Emotion Detection                    │
│     - LLM detects emotion, intensity     │
│       (1-10), and brief explanation      │
└─────────────────┬────────────────────────┘
                  ▼
┌──────────────────────────────────────────┐
│  9. Response Generation                  │
│     - Build full prompt:                 │
│       style + Super-Chunks + history +   │
│       semantic memories + emotion        │
│     - LLM generates max 100-word reply   │
└─────────────────┬────────────────────────┘
                  ▼
┌──────────────────────────────────────────┐
│ 10. Update Short-Term Memory             │
│     - Append {query, response, therapist}│
│     - Retain last 10 exchanges           │
│     - Save to conversation_memory.json   │
└─────────────────┬────────────────────────┘
                  ▼
┌──────────────────────────────────────────┐
│ 11. Conditional Long-Term Summarization  │
│     - Every 5 exchanges:                 │
│       * LLM summarizes full conversation │
│       * Store summary in                 │
│         memory_faiss_index/{user_id}/    │
└─────────────────┬────────────────────────┘
                  ▼
┌──────────────────────────────────────────┐
│ 12. Return Response                      │
│     - response (text)                    │
│     - emotion (name, intensity, reason)  │
│     - memories (semantic memory strings) │
│     - retrieved_sources (book/chunk)     │
│     - history (full user history)        │
│     - style (active persona prompt)      │
└──────────────────────────────────────────┘
```

---

## 🧩 Component Details

### **1. API Gateway Layer**
**Files:** `main.py`, `router.py`

- **FastAPI Application** — HTTP server with CORS configured for `localhost:3000`
- **Router** — Mounts `/chat/` prefix; handles POST and GET routes
- **Pydantic Validation** — `EventSchema` (user_id, query, style_type_id)

---

### **2. Business Logic Layer**
**File:** `endpoint.py`

| Function | Purpose |
|---|---|
| `load_memory()` | Load user conversation history from JSON |
| `save_memory()` | Persist updated conversation history |
| `load_style()` | Load AI persona definitions |
| `load_library_manifest()` | Load book descriptions with embeddings |
| `route_to_book()` | Route query to most relevant book (legacy, not in main path) |
| `summarize_and_store_memory()` | Summarize history & store in per-user FAISS index |
| `retrieve_semantic_memory()` | Retrieve top-k relevant past summaries from FAISS |
| `handle_event()` | Main chat endpoint handler |
| `get_chat_history()` | GET `/history/{user_id}` endpoint |

---

### **3. NLP Layer**
**File:** `nlp/query_optimizer.py`

Powered by **llama3.1:8b**. Converts the raw user query into an array of focused search concepts by stripping conversational filler. Falls back to the original query if LLM parsing fails.

**Example:**
```
Input:  "I've been feeling so anxious about my job lately, I can't sleep"
Output: ["anxiety", "work stress", "insomnia"]
```

---

### **4. Retrieval Layer**
**Files:** `retrieval/search_engine.py`, `retrieval/reranker.py`, `retrieval/context_expander.py`

#### Hybrid Search — `EnsembleRetriever`
Combines two complementary retrieval methods:

| Retriever | Type | Weight | Config |
|---|---|---|---|
| `BM25Retriever` | Keyword (sparse) | 30% | k=100 |
| `FAISS` | Vector (dense) | 70% | score_threshold=0.60, k=100 |

#### Cross-Encoder Reranker — `reranker.py`
Uses **`cross-encoder/ms-marco-MiniLM-L-6-v2`** (HuggingFace, ~90MB local model) to re-score every retrieved candidate against the combined query. Selects `top_n=10` highest-scoring chunks.

#### Context Expander — `context_expander.py`
After reranking, each top chunk is mapped back to its **parent Heading** (or Chapter) in the FAISS docstore. All sibling chunks under that heading are merged into a single **"Super-Chunk"**. This ensures the LLM receives full, coherent sections rather than isolated fragments. Guarantees **zero redundancy** by deduplicating on parent key.

#### `retrieve_for_keywords()` — Main Pipeline
```
keywords[] → Ensemble(BM25 + FAISS) per keyword
           → Deduplicate
           → Cross-Encoder Re-rank on combined query
           → Take top 10
           → Context Expansion → Super-Chunks
```

---

### **5. Generation Layer**
**File:** `generation/response_generator.py`

Powered by **llama3.1:8b** via Ollama. Two sequential LLM calls per request:

1. **Emotion Detection** — identifies emotion, intensity (1–10), and brief explanation from raw query
2. **Response Generation** — builds a full `ChatPromptTemplate` from:
   - Persona style prompt
   - Expanded knowledge Super-Chunks
   - Conversation history
   - Semantic memories
   - Detected emotion
   - Max **100 words** output constraint

---

### **6. Storage Layer**

#### JSON Files (`app/Json/`)
| File | Contents |
|---|---|
| `conversation_memory.json` | Recent chat history — last 10 exchanges per user |
| `response_style.json` | AI persona prompts (hannibal, kneeting, ANDY, default) |
| `library_manifest.json` | Book titles, descriptions, and embeddings for semantic routing |

#### Vector Stores (FAISS)
| Store | Path | Contents | Purpose |
|---|---|---|---|
| Knowledge Base | `faiss_index_markdown/` | Markdown-parsed book chunks + metadata (book, chapter, heading) | RAG context retrieval |
| Memory (per user) | `memory_faiss_index/{user_id}/` | Summarized conversations + timestamp | Long-term semantic memory |

#### Document Store
- **`books/`** — Source PDF files
- Indexed by **`markdownParser/markdown_indexer.py`** (converts PDFs → structured Markdown chunks, preserving heading hierarchy)

---

## 🚀 Features

### ✅ Advanced Hybrid Search
Combines **BM25 keyword search (30%)** and **FAISS vector search (70%)** via `EnsembleRetriever`. For each optimized keyword, both retrievers run across the `faiss_index_markdown` store. Results are pooled and deduplicated before re-ranking.

### ✅ Local Cross-Encoder Re-Ranking
Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` — a compact HuggingFace model that runs fully locally. Re-ranks all retrieved candidates with precise query-document pair scoring. Much more accurate than vector similarity alone.

### ✅ Context Expansion (Super-Chunks)
Retrieved chunks are "zoomed out" to their full parent section. Instead of feeding the LLM isolated paragraphs, it receives entire Heading sections — preserving narrative flow and structure.

### ✅ Query Optimization (NLP Pre-Processing)
Raw user messages are transformed by the LLM into compact, high-signal search keywords before hitting the retrieval pipeline. This dramatically improves retrieval precision.

### ✅ Multi-Persona Support
| ID | Persona |
|---|---|
| `hannibal` | Dr. Hannibal Lecter (analytical, unsettling) |
| `kneeting` | John Keating (inspirational, poetic) |
| `ANDY` | Dr. Victor Blaine (brutally honest) |
| `default` | Standard therapist |

### ✅ Emotion Detection
Every query is analyzed for:
- Emotion name (e.g., anxious, sad, hopeful)
- Intensity (1–10 scale)
- Brief explanation

### ✅ Semantic Memory (Long-Term)
- Conversation summaries are stored per user in a dedicated FAISS index every 5 exchanges
- Relevant past discussions are retrieved semantically and injected into every new prompt (top-2)

### ✅ Short-Term Memory
- Last 10 exchanges per user stored in `conversation_memory.json`
- Fully included in every prompt for context continuity

### ✅ Markdown-Aware Indexing
Books are processed by `markdown_indexer.py`, which converts PDFs into structured Markdown and chunks them by heading hierarchy — preserving chapter and section metadata used by the context expander.

---

## 📁 Project Structure

```
AdviceWizz/
├── app/
│   ├── main.py                        # FastAPI app entry point + CORS
│   ├── router.py                      # Route definitions
│   ├── endpoint.py                    # Main business logic & chat handler
│   │
│   ├── nlp/
│   │   └── query_optimizer.py         # LLM-powered query → keywords
│   │
│   ├── retrieval/
│   │   ├── search_engine.py           # Hybrid BM25+FAISS, rerank, expand pipeline
│   │   ├── reranker.py                # Local Cross-Encoder (ms-marco-MiniLM-L-6-v2)
│   │   └── context_expander.py        # Super-Chunk context expansion
│   │
│   ├── generation/
│   │   └── response_generator.py      # Emotion detection + LLM response generation
│   │
│   ├── markdownParser/
│   │   └── markdown_indexer.py        # PDF → Markdown → FAISS indexing
│   │
│   ├── Json/
│   │   ├── conversation_memory.json   # Short-term chat history (per user)
│   │   ├── response_style.json        # AI persona definitions
│   │   └── library_manifest.json      # Book metadata + embeddings
│   │
│   ├── books/                         # Source PDF files
│   ├── faiss_index_markdown/          # Knowledge base vector store
│   └── memory_faiss_index/            # Per-user long-term memory stores
│
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Project metadata
└── README.md                          # This file
```

---

## 🔧 Setup & Installation

### Prerequisites
- Python 3.9+
- Ollama with `llama3.1:8b` and `mxbai-embed-large` models

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd AdviceWizz

# Install dependencies
pip install -r requirements.txt

# Start Ollama
ollama serve

# Pull required models
ollama pull llama3.1:8b
ollama pull mxbai-embed-large

# Index your knowledge base (place PDFs in app/books/ first)
python app/markdownParser/markdown_indexer.py

# Run the server
uvicorn app.main:app --reload
```

---

## 🌐 API Reference

### `POST /chat/`

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
  "emotion": "anxious, intensity: 7/10 - User expresses stress and overwhelm...",
  "memories": "Relevant past discussions:\nSummary of prior work-stress conversation...",
  "retrieved_sources": [
    {
      "book": "psychology.pdf",
      "page": "42",
      "chunk": "...full expanded section text..."
    }
  ],
  "history": {
    "user123": [
      { "query": "...", "response": "...", "therapist": "hannibal" }
    ]
  },
  "style": "You are Dr. Hannibal Lecter..."
}
```

---

### `GET /chat/history/{user_id}`

Returns the full conversation history for a given user.

**Response:**
```json
{
  "user123": [
    { "query": "...", "response": "...", "therapist": "hannibal" }
  ]
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

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | FastAPI + Uvicorn |
| LLM | Ollama — llama3.1:8b |
| Embeddings | Ollama — mxbai-embed-large |
| Vector Store | FAISS (CPU) |
| Hybrid Search | LangChain EnsembleRetriever (BM25 + FAISS) |
| Re-Ranking | HuggingFace cross-encoder/ms-marco-MiniLM-L-6-v2 |
| RAG Framework | LangChain / LangChain-Classic |
| PDF Parsing | pymupdf4llm, pdfplumber, PyPDF2 |
| Data Validation | Pydantic |

---

## 📄 License

[Your License Here]

---

## 👥 Contributors

[Your Name]

---

## 🙏 Acknowledgments

- LangChain for the RAG and retrieval framework
- Ollama for local LLM deployment
- FAISS (Meta AI) for vector search
- HuggingFace for the cross-encoder reranker model
- FastAPI for the web framework
