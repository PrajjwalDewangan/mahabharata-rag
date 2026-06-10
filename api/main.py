import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
 
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
 
load_dotenv(ROOT / ".env")
 
from generation.generator import MahabharataGenerator

generator: MahabharataGenerator | None = None
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    global generator
    print("Initialising RAG pipeline...")
    generator = MahabharataGenerator(
        chroma_path=str(ROOT / "data" / "chroma"),
        chunks_path=str(ROOT / "data" / "chunks.json"),
    )
    print("RAG pipeline ready.")
    yield
    print("Shutting down.")

app = FastAPI(
    title="Mahabharata RAG API",
    description="Hybrid RAG chatbot over the KMG English translation",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=str(ROOT / "ui")),
    name="static",
)

class ChatRequest(BaseModel):
    query: str
 
class SourceItem(BaseModel):
    parva_name    : str
    section_number: int
    section_heading: str
    preview       : str       
 
class ChatResponse(BaseModel):
    answer : str
    sources: list[SourceItem]
    history_length: int

@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serve the chat UI."""
    return FileResponse(str(ROOT / "ui" / "index.html"))
 
 
@app.get("/health")
async def health():
    return {
        "status"     : "ok",
        "environment": os.getenv("ENVIRONMENT", "production"),
        "model"      : os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
    }
 
 
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
 
    try:
        result = generator.chat(request.query)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    
    seen = set()
    source_items = []
    for chunk in result["chunks"]:
        key = (chunk["parva_name"], chunk["section_number"])
        if key not in seen:
            seen.add(key)
            source_items.append(SourceItem(
                parva_name     = chunk["parva_name"],
                section_number = chunk["section_number"],
                section_heading= chunk["section_heading"],
                preview        = chunk["text"][:200].strip() + "...",
            ))
 
    return ChatResponse(
        answer         = result["answer"],
        sources        = source_items,
        history_length = len(generator.history),
    )
 
 
@app.post("/reset")
async def reset():
    generator.reset()
    return {"status": "ok", "message": "Conversation history cleared."}