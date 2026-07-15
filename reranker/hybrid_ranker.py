"""
Combines CLIP embedding similarity with structured attribute matching.

Formula:
    final_score = w1 * embedding_similarity
                + w2 * attribute_match_ratio
                + w3 * vibe_match

Weight justification (see utils/config.RANK_WEIGHTS):
  - embedding_similarity (0.55): CLIP's joint embedding already encodes a
    lot of the query's meaning (composition, color, scene) even for
    multi-attribute queries, so it stays the dominant signal — we're
    correcting it, not replacing it.
  - attribute_match (0.30): a hard, explainable signal for the fields the
    parser was confident about (color/clothing_type/environment/style).
    Weighted high enough that an image matching 4/4 requested attributes
    can outrank a slightly-higher-cosine image matching 1/4.
  - vibe_match (0.15): vibe is the most subjective/overlapping category
    (e.g. "smart casual" vs "minimalist" often both fit), so it nudges
    ranking rather than dominating it.
  These are starting weights, not fixed constants — see README's
  "future enhancements" section on learning them from feedback data.
"""
from utils.config import RANK_WEIGHTS


def score_result(result, query_attributes):
    """
    result: one dict from FaissStore.search() — has embedding_similarity
            and attributes (the image's tags)
    query_attributes: dict from parser.query_parser.parse_query()
    Returns: (final_score: float, explanation: dict)
    """
    image_attrs = result.get("attributes", {})

    # attribute match ratio: fraction of REQUESTED (non-vibe) attributes
    # this image also has, exact match on shared vocab
    requested = {k: v for k, v in query_attributes.items() if k != "vibe"}
    matched = {k: v for k, v in requested.items()
               if image_attrs.get(k) == v}
    attribute_match_ratio = (len(matched) / len(requested)) if requested else 1.0

    # vibe handled separately: binary match (or neutral 0.5 if query didn't
    # specify a vibe, so it doesn't penalize otherwise-good matches)
    if "vibe" in query_attributes:
        vibe_match = 1.0 if image_attrs.get("vibe") == query_attributes["vibe"] else 0.0
    else:
        vibe_match = 0.5

    embedding_similarity = result["embedding_similarity"]

    final_score = (
        RANK_WEIGHTS["embedding_similarity"] * embedding_similarity
        + RANK_WEIGHTS["attribute_match"] * attribute_match_ratio
        + RANK_WEIGHTS["vibe_match"] * vibe_match
    )

    explanation = {
        "matched_attributes": matched,
        "requested_attributes": requested,
        "matched_count": f"{len(matched)}/{len(requested)}" if requested else "N/A",
        "vibe_requested": query_attributes.get("vibe"),
        "vibe_matched": image_attrs.get("vibe") if vibe_match == 1.0 else None,
        "embedding_similarity": round(embedding_similarity, 4),
        "final_score": round(final_score, 4),
    }
    return final_score, explanation


def rerank(results, query_attributes, top_k):
    scored = []
    for r in results:
        score, explanation = score_result(r, query_attributes)
        scored.append({**r, "final_score": score, "explanation": explanation})
    scored.sort(key=lambda x: x["final_score"], reverse=True)
    return scored[:top_k]
