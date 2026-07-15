"""
Entrypoint. Loads the prebuilt FAISS index at startup (via retriever.faiss_store's
module-level singleton import) — no heavy model download/inference happens
here beyond loading OpenCLIP itself for query-time encoding.

Run locally:
    uvicorn backend.main:app --reload --port 8000

Render start command:
    uvicorn backend.main:app --host 0.0.0.0 --port $PORT
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import search, upload, feedback

app = FastAPI(
    title="Glance Fashion & Context Retrieval API",
    description="Multimodal semantic image retrieval over a fashion image dataset.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router, tags=["search"])
app.include_router(upload.router, tags=["upload"])
app.include_router(feedback.router, tags=["feedback"])


@app.get("/health")
def health():
    from retriever.faiss_store import store
    return {"status": "ok", "indexed_images": store.index.ntotal}
