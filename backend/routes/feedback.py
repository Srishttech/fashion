from fastapi import APIRouter

from backend.schemas import FeedbackRequest
from feedback.feedback_store import record_feedback

router = APIRouter()


@router.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    record_feedback(req.query, req.image_id, req.relevant)
    return {"status": "recorded"}
