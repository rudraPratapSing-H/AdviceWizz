# RAG Comparison Guide

## 🚀 Quick Start

### Step 1: Make sure your FastAPI server is running
```powershell
# In terminal 1
cd C:\Projects\AdviceWizz\AdviceWizz
uvicorn app.endpoint:app --reload
```

### Step 2: Run the comparison script to collect data
```powershell
# In terminal 2
cd C:\Projects\AdviceWizz\AdviceWizz
python app/compare_rag.py
```

This will generate `rag_comparison_results.json` with responses from:
- 🔴 Llama3 (Local, no RAG)
- 🔵 Gemini API (Cloud, no RAG)
- 🟢 RAG (Semantic routing + FAISS)

### Step 3: Launch the visualization dashboard
```powershell
# In terminal 3
cd C:\Projects\AdviceWizz\AdviceWizz\app
streamlit run visualize_comparison.py
```

Your browser will open automatically with the interactive dashboard!

## 📊 Dashboard Features

- **Performance Metrics**: Average response times for all models
- **Response Time Comparison**: Bar chart comparing speed across queries
- **Response Length Analysis**: Character count comparison
- **Speed Trends**: Line chart showing performance over queries
- **Detailed Comparison**: Side-by-side response viewer
- **Source Distribution**: Pie chart of books used by RAG
- **Performance Summary**: Table with success rates

## 🔑 Environment Setup

Make sure your `.env` file has:
```
GEMINI_API_KEY=your-api-key-here
```

Get your key from: https://aistudio.google.com/app/apikey

## 📝 Test Queries

The script tests 10 queries across all 5 books:
1. 48 Laws of Power (3 queries)
2. Atomic Habits (3 queries)
3. The Subtle Art of Not Giving a F*ck (2 queries)
4. The Art of Being ALONE (1 query)
5. The Art of Loving (1 query)

## 🎯 What to Look For

- **Speed**: Which model responds fastest?
- **Quality**: Which responses are most relevant?
- **Consistency**: Which model maintains performance?
- **RAG Value**: Does semantic routing improve responses?
- **Source Relevance**: Are the retrieved books appropriate?

## 🔄 Re-running Comparisons

To collect new data:
```powershell
# Delete old results
del rag_comparison_results.json

# Run comparison again
python app/compare_rag.py

# Dashboard auto-refreshes!
```
