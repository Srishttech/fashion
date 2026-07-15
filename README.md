# Glance — Fashion & Context Image Retrieval System

Multimodal semantic image retrieval over a fashion image dataset. Given a
natural-language query ("someone in a blue shirt sitting on a park bench"),
returns the top-K most relevant *existing* images from the dataset — not
generation, not keyword search, not a shopping recommender.

Dataset used for indexing: `test/` folder from `val_test2020.zip`, 3200
JPEGs (hash-named, no bundled captions/labels).

---

## 1. Why this isn't "vanilla CLIP + FAISS"

Vanilla approach: encode images with CLIP, encode query with CLIP, cosine
similarity, return top-K. Works, but:
- gives you a similarity number with no explanation of *why* something matched
- has no way to weight "this image matches 4/4 things you asked for" over
  "this image is 0.02 cosine higher for reasons you can't see"
- multi-attribute queries ("blue shirt AND park AND casual") get blended
  into one vector, so a strong match on 2/3 attributes can outrank a
  weaker-embedding but 3/3-attribute match

This system adds a **structured metadata layer + hybrid ranking** on top of
CLIP, without adding a second heavy model:

- **No BLIP captioning.** The brief's suggested pipeline (BLIP caption →
  regex/NLP-parse into attributes) means running a second heavy model at
  index time and writing a caption parser that breaks on phrasing it
  hasn't seen. Instead, `indexer/attribute_tagger.py` reuses the CLIP text
  encoder already loaded for embeddings, and does **zero-shot
  classification per attribute group** (color / clothing_type /
  environment / style / vibe) directly against a closed vocabulary. Same
  vocab is reused by the query parser, so an image attribute and a query
  attribute are directly comparable (set intersection, not NLP fuzzy
  matching) — this is also what makes explainability free.
- **Hybrid ranking** (`reranker/hybrid_ranker.py`): `final_score =
  0.55·embedding_similarity + 0.30·attribute_match_ratio + 0.15·vibe_match`.
  See that file's docstring for why those specific weights.
- **Explainability is structural, not decorative**: because attributes are
  a closed vocab shared by images and queries, "matched 3/4 attributes:
  color=blue, environment=park, style=casual" falls out of the scoring
  function itself, not a separate hand-written justification layer.

Trade-off, stated honestly for the interview: closed vocabulary can't
describe things outside it, and the current schema has one `color` field
per image (can't separately track "white shirt, red tie" as two colors).
Both are one-line-vocab-edit / one-field-schema-change extensions — see
Future Enhancements.

---

## 2. Architecture

```
OFFLINE (run once, on a machine with normal internet — laptop/Colab, NOT Render)
  data/images/*.jpg
      -> indexer/encoder.py         (OpenCLIP ViT-B-32, CPU)
      -> indexer/attribute_tagger.py (zero-shot tag: color, clothing_type,
                                       environment, style, vibe)
      -> indexer/build_index.py writes:
           data/faiss.index    (FAISS IndexFlatIP over 512-d normalized vectors)
           data/metadata.json  ({image_id: {path, attributes, faiss_row}})

ONLINE (Render, CPU only, no re-encoding of the dataset)
  user query
      -> parser/query_parser.py     (keyword pass + CLIP zero-shot fallback
                                      -> same attribute schema as images)
      -> indexer/encoder.encode_text (query -> 512-d vector)
      -> retriever/faiss_store.py   (FAISS search, top 50)
      -> reranker/hybrid_ranker.py  (hybrid score + explanation, top 5)
      -> feedback/feedback_store.py (small score nudge from past 👍/👎)
      -> backend/main.py (FastAPI) -> frontend/app.py (Streamlit)
```

## 3. Folder structure

```
backend/       FastAPI app + routes (search, upload, feedback) + schemas
indexer/       encoder.py (OpenCLIP wrapper), attribute_tagger.py (zero-shot
               tagging), build_index.py (offline CLI script)
retriever/     faiss_store.py — index load/search/incremental-add, singleton
reranker/      hybrid_ranker.py — scoring + explanation
parser/        query_parser.py — query -> structured attributes
feedback/      feedback_store.py — SQLite 👍/👎 storage + score boost
utils/         config.py (all constants/weights/vocab), image_utils.py
frontend/      Streamlit demo UI
data/          images/ (sample of 60 for smoke-testing), faiss.index,
               metadata.json, feedback.db (all generated, not hand-written)
```

ML logic (indexer/retriever/reranker/parser/feedback) is fully decoupled
from the API layer (backend/) — the FastAPI routes are thin, they just
call into these modules. This is what "modular" buys you: any of these
five can be unit-tested or swapped independently.

---

## 4. Setup (run locally — needs real internet for model weights)

```bash
git clone <repo>
cd glance-fashion-retrieval
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Put your dataset images in data/images/ (a 60-image sample is
#    already there for smoke-testing the pipeline end to end)

# 2. Build the index (downloads OpenCLIP weights on first run)
python -m indexer.build_index --images-dir data/images --limit 800

# 3. Run the API
uvicorn backend.main:app --reload --port 8000

# 4. Run the UI (separate terminal)
streamlit run frontend/app.py
```

> **Note on this repo's origin**: the code was written and logic-tested
> (FAISS search/add, hybrid ranking, feedback boost, query parsing) using
> dummy embeddings in a network-restricted sandbox that cannot reach
> huggingface.co. Running `build_index.py` for real requires a normal
> internet connection (your laptop, Colab, or Render's build step) to
> download the OpenCLIP checkpoint the first time.

---

## 5. API design

| Endpoint | Method | Body | Returns |
|---|---|---|---|
| `/search` | POST | `{query, top_k, profile?}` | `{query, parsed_attributes, results: [{image_id, path, attributes, embedding_similarity, final_score, explanation}]}` |
| `/upload` | POST | multipart file | `{image_id, attributes, status}` — encodes, tags, appends to FAISS without rebuilding |
| `/feedback` | POST | `{query, image_id, relevant}` | `{status}` |
| `/health` | GET | — | `{status, indexed_images}` |

## 6. Metadata schema (`data/metadata.json`)

```json
{
  "<image_id>": {
    "path": "images/xxx.jpg",
    "attributes": {
      "color": "blue", "clothing_type": "shirt",
      "environment": "park", "style": "casual", "vibe": "weekend casual"
    },
    "faiss_row": 12
  }
}
```
A group key is **omitted** (not null) if zero-shot confidence didn't clear
the threshold — an unfilled field means "unknown", never a guessed default.

## 7. Feedback learning without retraining

1. **Immediate**: `feedback_store.get_feedback_boost()` — small capped
   additive nudge (±0.1 max) to `final_score` from an image's net
   historical votes. Pure lookup, no training loop, applied at request time.
2. **Medium-term**: periodically mine `(query_attributes, image_id, label)`
   rows to re-tune `RANK_WEIGHTS` (e.g. small grid search / logistic
   regression over the 3 scalar weights) — still zero changes to CLIP itself.
3. **Long-term** (not implemented here, out of scope for Render deploy): use
   accumulated feedback as weak supervision for a linear probe on top of
   frozen CLIP embeddings.

## 8. Scalability: 1K → 1M images

| | 1K–10K | 100K | 1M+ |
|---|---|---|---|
| FAISS index | `IndexFlatIP` (exact, what's here) | `IndexIVFFlat` (cluster + probe) | `IndexIVFPQ` (quantized, fits in RAM) |
| Metadata store | JSON file (current) | SQLite | Postgres / a document store |
| Attribute filtering | none needed | pre-filter FAISS candidates by attribute before rerank | metadata-first filter, then FAISS only within filtered set |
| Rerank cost | negligible on 50 candidates | still fine on 50 | still fine — rerank always operates on FAISS's top-K, not the full index, so this step doesn't get more expensive as the dataset grows |

Retrieval complexity stays effectively O(top-K) for reranking regardless
of dataset size; the FAISS index type is what changes to keep the *search*
step itself fast at scale.

## 9. Render deployment architecture

- Build step: `pip install -r requirements.txt` (CPU-only torch via the
  `--extra-index-url` pinned in requirements.txt — keeps the image small,
  no CUDA deps pulled in)
- `data/faiss.index` + `data/metadata.json` are committed to the repo (or
  uploaded to a Render persistent disk) — **built once, offline**, never
  regenerated on the server
- Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- Only two things happen at request time on Render: encode the query text
  (CLIP text tower only, cheap), and a FAISS search over an already-loaded
  in-memory index. No image encoding happens on Render except for the
  optional `/upload` endpoint (also CPU, one image at a time — cheap).

## 10. Future enhancements

- Weather/season awareness, city/location context: extra attribute groups
  in the same `ATTRIBUTE_GROUPS` schema, no architecture change
- Brand recognition: would need a proper detector/logo-classifier, the
  first genuinely new model in the pipeline
- Multilingual search: translate query before `parse_query`/`encode_text`,
  or swap in a multilingual CLIP checkpoint (`encoder.py` is the only file
  that changes)
- Per-garment color (e.g. shirt vs tie separately): extend the metadata
  schema from `color: str` to `colors: {garment: color}` — parser and
  tagger logic already generalize to this once the vocab structure changes
- Personalization: log `profile` + accepted results per user, use as an
  extra soft ranking signal like the existing vibe hint

## 11. Known limitations (say these upfront in the interview)

- Closed attribute vocabulary — can't describe things outside `utils/config.py`'s lists
- One `color`/`clothing_type` field per image — can't yet capture "white shirt + red tie" as two separate colors
- Zero-shot tagging confidence threshold (0.22) was picked as a reasonable
  starting point, not empirically tuned against labeled ground truth
  (there isn't any for this dataset) — first thing to validate with real
  data before trusting it in production
