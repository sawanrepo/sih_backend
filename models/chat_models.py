from pydantic import BaseModel
from typing import List, Dict, Optional

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    goal: str
    lca_context: Dict[str, float]
    conversation_history: List[ChatMessage]

class ChatResponse(BaseModel):
    success: bool
    response: str
    suggested_scenarios: Optional[List[Dict]] = None
    confidence: Optional[float] = None