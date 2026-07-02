from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import ChatRequest, ChatResponse, HealthResponse
from app.agent import run_chat, MAX_TURNS

app = FastAPI(title="SHL Assessment Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    if len(request.messages) > MAX_TURNS:
        # Evaluator caps at 8 turns; if it somehow sends more, just use the
        # most recent MAX_TURNS so we don't blow the timeout on a huge prompt.
        request.messages = request.messages[-MAX_TURNS:]
    return run_chat(request.messages)
