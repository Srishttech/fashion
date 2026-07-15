"""
Parses a natural language query into the same structured attribute schema
used to tag images (utils.config.ATTRIBUTE_GROUPS), so matching at rank
time is a plain set/dict comparison.

Two-pass approach:
  1. Fast keyword pass: direct substring match against each vocab
     (catches "blue shirt", "office", "formal" instantly, no model call).
  2. CLIP zero-shot fallback pass: for any group the keyword pass didn't
     fill, embed the query text and compare it against that group's
     prompt embeddings (same ones cached in attribute_tagger) — this
     catches paraphrases the keyword pass misses, e.g. "sitting on a bench
     in the outdoors" -> environment: park/outdoor without the literal
     word "park".
"""
from utils.config import ATTRIBUTE_GROUPS
from indexer.attribute_tagger import _get_group_embeddings, CONFIDENCE_THRESHOLD, _softmax
from indexer.encoder import encode_text
import numpy as np


def parse_query(query_text):
    query_lower = query_text.lower()
    attributes = {}

    # Pass 1: keyword match
    for group, vocab in ATTRIBUTE_GROUPS.items():
        for term in vocab:
            if term in query_lower:
                attributes[group] = term
                break

    # Pass 2: zero-shot fallback for unfilled groups only
    missing_groups = [g for g in ATTRIBUTE_GROUPS if g not in attributes]
    if missing_groups:
        query_emb = encode_text(query_text)[0]  # (512,)
        for group in missing_groups:
            text_embs = _get_group_embeddings(group)
            sims = text_embs @ query_emb
            probs = _softmax(sims)
            top_idx = int(np.argmax(probs))
            if probs[top_idx] >= CONFIDENCE_THRESHOLD:
                attributes[group] = ATTRIBUTE_GROUPS[group][top_idx]

    return attributes
