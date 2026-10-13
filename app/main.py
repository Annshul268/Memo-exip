"""
FastAPI application entrypoint for Memo-exip.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router

app = FastAPI(
    title="Memo-exip: Adaptive Temporal Memory Management API",
    description=(
        "Research prototype for continual external memory in LLM agents. "
        "Provides bitemporal indexing, version-aware updates, intent-based hybrid retrieval, "
        "and token-budget-aware memory selection."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for Streamlit and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
