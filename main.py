from fastapi import FastAPI
from routers import predict, chat

app = FastAPI(title="LCA AI Backend")

# Routers
app.include_router(predict.router, prefix="/api/lca", tags=["Prediction"])
app.include_router(chat.router, prefix="/api/lca", tags=["Chat"])