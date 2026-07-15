from typing import Optional
from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    profile: Optional[str] = None  # optional personality-aware signal:
                                    # "professional" | "minimalist" | "creative"
                                    # | "trendy" | "classic"


class FeedbackRequest(BaseModel):
    query: str
    image_id: str
    relevant: bool
