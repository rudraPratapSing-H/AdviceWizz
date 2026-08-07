"""
RAG vs Non-RAG Comparison Script

This script compares the performance of your RAG application against 
baselines: Llama3 without RAG and Google Gemini API.
"""

import json
import time
import requests
import os
from langchain_ollama import OllamaLLM
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize LLM
llm = OllamaLLM(model="llama3:8b")

# Initialize Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
else:
    gemini_model = None
    print("⚠️  GEMINI_API_KEY not found. Gemini comparison will be skipped.")

# FastAPI endpoint
ENDPOINT_URL = "http://localhost:8000/chat/"

def load_response_style(style_type_id: str = "default") -> str:
    """Load the response style prompt from response_style.json"""
    try:
        with open("../Json/response_style.json", "r", encoding="utf-8") as f:
            styles = json.load(f)
            return styles.get(style_type_id, styles.get("default", "You are a helpful assistant."))
    except FileNotFoundError:
        return "You are a helpful assistant."

def non_rag_response(query: str, style_type_id: str = "default") -> dict:
    """Generate response WITHOUT RAG using Llama3 (baseline)"""
    user_style = load_response_style(style_type_id)
    
    prompt = f"""{user_style}

Question: {query}
"""
    
    start_time = time.time()
    response = llm.invoke(prompt)
    end_time = time.time()
    
    return {
        "response": response,
        "time_taken": end_time - start_time,
        "method": "Non-RAG (Llama3 Baseline)",
        "style": style_type_id
    }

def gemini_response(query: str, style_type_id: str = "default") -> dict:
    """Generate response using Google Gemini API"""
    if not gemini_model:
        return {
            "response": "ERROR: Gemini API key not configured",
            "time_taken": 0,
            "method": "Gemini API",
            "style": style_type_id,
            "error": "GEMINI_API_KEY environment variable not set"
        }
    
    user_style = load_response_style(style_type_id)
    
    prompt = f"""{user_style}

Question: {query}
"""
    
    start_time = time.time()
    try:
        response = gemini_model.generate_content(prompt)
        end_time = time.time()
        
        return {
            "response": response.text,
            "time_taken": end_time - start_time,
            "method": "Gemini API",
            "style": style_type_id
        }
    except Exception as e:
        end_time = time.time()
        return {
            "response": f"ERROR: {str(e)}",
            "time_taken": end_time - start_time,
            "method": "Gemini API",
            "style": style_type_id,
            "error": str(e)
        }

def rag_response_from_endpoint(query: str, user_id: str, style_type_id: str = "default") -> dict:
    """
    Generate response WITH RAG by calling your FastAPI endpoint
    """
    start_time = time.time()
    
    try:
        payload = {
            "user_id": user_id,
            "query": query,
            "style_type_id": style_type_id
        }
        
        response = requests.post(ENDPOINT_URL, json=payload, timeout=60)
        end_time = time.time()
        
        if response.status_code == 202:
            data = response.json()
            return {
                "response": data.get("response", "No response"),
                "time_taken": end_time - start_time,
                "method": "RAG (via Endpoint)",
                "style": style_type_id,
                "retrieved_sources": data.get("retrieved_sources", []),
                "emotion": data.get("emotion", ""),
                "memories": data.get("memories", "")
            }
        else:
            return {
                "response": f"Error: {response.status_code}",
                "time_taken": end_time - start_time,
                "method": "RAG (via Endpoint)",
                "style": style_type_id,
                "error": response.text
            }
    
    except requests.exceptions.ConnectionError:
        return {
            "response": "ERROR: Could not connect to endpoint",
            "time_taken": 0,
            "method": "RAG (via Endpoint)",
            "style": style_type_id,
            "error": "Connection failed - ensure FastAPI server is running on localhost:8000"
        }
    except Exception as e:
        return {
            "response": f"ERROR: {str(e)}",
            "time_taken": 0,
            "method": "RAG (via Endpoint)",
            "style": style_type_id,
            "error": str(e)
        }

def rag_response(query: str, retrieved_context: str, style_type_id: str = "default") -> dict:
    """Generate response WITHOUT RAG"""
    user_style = load_response_style(style_type_id)
    
    prompt = f"""{user_style}

<context>
{retrieved_context}
</context>

Question: {query}
"""
    
    start_time = time.time()
    response = llm.invoke(prompt)
    end_time = time.time()
    
    return {
        "response": response,
        "time_taken": end_time - start_time,
        "method": "WITHOUT RAG",
        "style": style_type_id
    }

def compare_responses(query: str, retrieved_context: str = None, style_type_id: str = "default", user_id: str = "test_user", use_endpoint: bool = True) -> dict:
    """
    Collect RAG, Non-RAG (Llama3), and Gemini responses for comparison
    
    Args:
        query: The query to test
        retrieved_context: Optional static context (if not using endpoint)
        style_type_id: The persona style to use
        user_id: User ID for endpoint requests
        use_endpoint: If True, use FastAPI endpoint; if False, use static context
    """
    print(f"\nProcessing: {query[:60]}...")
    
    print("  [1/3] Generating Non-RAG response (Llama3)...")
    non_rag_result = non_rag_response(query, style_type_id)
    
    print("  [2/3] Generating Gemini response...")
    gemini_result = gemini_response(query, style_type_id)
    
    print("  [3/3] Generating RAG response...")
    if use_endpoint:
        rag_result = rag_response_from_endpoint(query, user_id, style_type_id)
    else:
        rag_result = rag_response(query, retrieved_context or "", style_type_id)
    
    return {
        "query": query,
        "style": style_type_id,
        "user_id": user_id,
        "non_rag_llama3": non_rag_result,
        "gemini": gemini_result,
        "rag": rag_result,
        "timestamp": time.time()
    }

def test_with_sample_context():
    """Test with sample context"""
    sample_context = """
    Power is not about your position or title, but about your ability to understand 
    the psychological needs and desires of others. Those who master the art of 
    observation gain tremendous influence. The key is to listen more than you speak, 
    watch people's reactions carefully, and use their own desires against them subtly.
    """
    
    test_queries = [
        "How can I gain more influence at work?",
        "What's the best way to handle a manipulative colleague?",
    ]
    
    results = []
    for query in test_queries:
        result = compare_responses(query, sample_context, style_type_id="default")
        results.append(result)
    
    with open("../Json/rag_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("Results saved to: rag_comparison_results.json")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("RAG vs Non-RAG Comparison - 3-Model Response Collection")
    print("="*80)
    print(f"FastAPI Endpoint: {ENDPOINT_URL}")
    print("Models: Llama3 (Local) | Gemini 1.5 Flash (Cloud) | RAG (Semantic Routing)")
    print("Make sure your FastAPI server is running!\n")
    
    # Test queries covering all books
    test_queries = [
        # 48 Laws of Power queries
        "How can I gain more influence at work?",
        "What's the best way to handle a manipulative colleague?",
        "How do I understand people's hidden motivations?",
        
        # # Atomic Habits queries
        "How can I build better habits in my daily life?",
        "What's the best way to break a bad habit?",
        "How do I stay consistent with my goals?",
        
        # # The Subtle Art of Not Giving a F*ck queries
        "How do I manage anxiety and stress?",
        "What does it mean to find true purpose in life?",
        
        # # The Art of Being ALONE queries
        "How can I embrace solitude without feeling lonely?",
        
        # # The Art of Loving queries
        "What is the true nature of love and relationships?",
    ]
    
    print(f"Collecting responses for {len(test_queries)} queries...\n")
    
    results = []
    for i, query in enumerate(test_queries, 1):
        print(f"[{i}/{len(test_queries)}]", end=" ")
        result = compare_responses(
            query=query,
            style_type_id="default",
            user_id=f"comparison_test_{i}",
            use_endpoint=True
        )
        results.append(result)
    
    # Save results to JSON
    output_file = "../Json/rag_comparison_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"✅ Results saved to: {output_file}")
    print(f"Total responses collected: {len(results)}")
    print(f"{'='*80}\n")