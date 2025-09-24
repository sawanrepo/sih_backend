from fastapi import APIRouter
from models.chat_models import ChatRequest, ChatResponse
from core.llm import get_recommendation

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    response = get_recommendation(req.message, req.goal, req.lca_context, req.conversation_history)
    return ChatResponse(success=True, response=response, confidence=0.9)