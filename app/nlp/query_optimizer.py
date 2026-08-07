import json
import ast
from langchain_ollama import OllamaLLM

def optimize_query(query: str) -> list:
    """
    Optimizes a search query by extracting core entities using Llama3.1:8b.
    Returns a list of strings representing search concepts.
    """
    llm = OllamaLLM(model="llama3.1:8b")
    
    prompt = f"""You are a search query optimizer. Your job is to extract the core point/s of what the user want to know from the user's raw input which will be used in RAG for vector similarity. Keep the query concise and focused on the main topics. Remove all conversational filler. Return an array of strings representing the most important search concepts. Return ONLY a valid JSON array.

User's input: {query}
"""
    
    try:
        response = llm.invoke(prompt).strip()
        
        # Clean up markdown code blocks if the LLM includes them
        clean_response = response
        if clean_response.startswith("```"):
            lines = clean_response.split('\n')
            if len(lines) >= 2:
                # Remove the first and last line (which contain the backticks)
                clean_response = '\n'.join(lines[1:-1])
            else:
                clean_response = clean_response.strip('`')
        clean_response = clean_response.strip()
        
        # Attempt to parse as JSON
        parsed = json.loads(clean_response)
        
        # Ensure it's a list
        if isinstance(parsed, list):
            return parsed
            
    except Exception as e:
        # If parsing fails or any other error occurs, try ast evaluation as a fallback
        try:
            parsed = ast.literal_eval(response)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
            
    # If all parsing fails or the result isn't a list, return the original query in an array
    return [query]

if __name__ == "__main__":
    # Test cases
    test_queries = [
        "What does the author means bu surrounded by idiots?"
    ]
    
    for q in test_queries:
        print(f"Original: {q}")
        print(f"Optimized: {optimize_query(q)}")
        print("-" * 40)
