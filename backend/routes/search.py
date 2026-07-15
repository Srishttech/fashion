from fastapi import APIRouter

from backend.schemas import SearchRequest
from indexer.encoder import encode_text
from retriever.faiss_store import store
from reranker.hybrid_ranker import rerank
from parser.query_parser import parse_query
from feedback.feedback_store import get_feedback_boost
from utils.config import TOP_K_FAISS

router = APIRouter()

# Optional personality-profile nudge: treated as a soft extra vibe hint,
# never overrides the query's own parsed attributes (brief requirement:
# "natural language queries must remain the primary source of relevance").
PROFILE_VIBE_HINT = {
    "professional": "professional",
    "minimalist": "minimalist",
    "creative": "streetwear",
    "trendy": "party",
    "classic": "vintage",
}


@router.post("/search")
def search(req: SearchRequest):
    query_attributes = parse_query(req.query)

    if req.profile and "vibe" not in query_attributes:
        hint = PROFILE_VIBE_HINT.get(req.profile.lower())
        if hint:
            query_attributes["vibe"] = hint

    query_embedding = encode_text(req.query)[0]
    candidates = store.search(query_embedding, top_k=TOP_K_FAISS)
    ranked = rerank(candidates, query_attributes, top_k=req.top_k)

    for r in ranked:
        r["final_score"] += get_feedback_boost(r["image_id"])

    return {
        "query": req.query,
        "parsed_attributes": query_attributes,
        "results": ranked,
    }
