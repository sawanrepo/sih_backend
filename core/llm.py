from langchain_google_genai import ChatGoogleGenerativeAI

chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

def get_recommendation(message: str, goal: str, lca_context: dict, history: list):
    """
    Generate sustainability recommendation using Gemini + context.
    """
    context_str = f"Goal: {goal}\nContext: {lca_context}"
    conversation = "\n".join([f"{h['role']}: {h['content']}" for h in history])
    
    prompt = f"""
    You are an expert LCA assistant.
    User message: {message}
    {context_str}
    Past conversation:
    {conversation}
    Provide recommendations with reasoning.
    """

    response = chat_model.invoke(prompt)
    return response.content