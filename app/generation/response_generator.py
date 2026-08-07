from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

llm = OllamaLLM(model="llama3.1:8b")

def generate_response(query: str, retrieved_context: str, memory_context: str, user_style: str, user_history: list) -> tuple:
    """
    Generates a response using Llama3.1:8b based on the provided context.
    Returns a tuple of (response_text, emotion).
    """
    
    # Format the immediate conversation history
    history_text = "\n".join(
        [f"User: {m['query']}\nAssistant: {m['response']}" for m in user_history]
    )
    
    full_context = f"{retrieved_context}\n\nPrevious conversation:\n{history_text}"
    
    # 1. Detect emotion
    emotion_prompt = f"""Based on this message, identify the user's emotional state:

"{query}"

Respond with: emotion name, intensity (1-10), and brief explanation."""

    emotion = llm.invoke(emotion_prompt)

    # 2. Generate final response
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
        query=query,
        emotion=emotion,
        memories=memory_context
    )
    
    print("Generating response with LLM...")
    response = llm.invoke(prompt_str)
    
    return response, emotion
