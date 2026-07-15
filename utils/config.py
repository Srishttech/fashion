"""
Central configuration. Single source of truth for paths, model names, and
ranking weights so nothing is hardcoded across modules.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")
FEEDBACK_DB_PATH = os.path.join(DATA_DIR, "feedback.db")

# --- Models ---
# OpenCLIP ViT-B-32 (laion2b_s34b_b79k) chosen over the original OpenAI CLIP
# checkpoint: better zero-shot accuracy on fashion-adjacent concepts, same
# embedding dim (512), same CPU inference cost. FashionCLIP is a drop-in
# swap (same encoder.py) if a fashion-specific checkpoint is preferred later.
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
EMBEDDING_DIM = 512

# --- Attribute vocabularies used for zero-shot tagging AND query parsing ---
# Same vocab on both sides is what makes attribute matching exact-string
# (not fuzzy NLP), which is what makes the "matched attributes" explanation
# possible and cheap.
COLORS = ["black", "white", "red", "blue", "green", "yellow", "pink",
          "purple", "brown", "grey", "beige", "orange", "navy", "multicolor"]

CLOTHING_TYPES = ["shirt", "t-shirt", "dress", "jacket", "coat", "suit",
                   "jeans", "trousers", "skirt", "shorts", "sweater",
                   "hoodie", "blazer", "saree", "kurta", "gown"]

ENVIRONMENTS = ["office", "park", "beach", "street", "studio", "home",
                "restaurant", "gym", "outdoor", "indoor", "urban", "nature"]

STYLES = ["formal", "casual", "streetwear", "athletic", "bohemian",
          "vintage", "minimalist", "traditional", "elegant"]

VIBES = ["professional", "weekend casual", "smart casual", "streetwear",
         "minimalist", "vintage", "luxury", "sporty", "travel", "party"]

ATTRIBUTE_GROUPS = {
    "color": COLORS,
    "clothing_type": CLOTHING_TYPES,
    "environment": ENVIRONMENTS,
    "style": STYLES,
    "vibe": VIBES,
}

# --- Hybrid ranking weights ---
# embedding similarity is the primary signal (CLIP already captures a lot),
# attribute match is a hard-ish correction layer, vibe is softer (subjective)
# so it gets the smallest weight. See reranker/hybrid_ranker.py docstring
# for the full justification.
RANK_WEIGHTS = {
    "embedding_similarity": 0.55,
    "attribute_match": 0.30,
    "vibe_match": 0.15,
}

TOP_K_FAISS = 50
TOP_K_FINAL = 5
