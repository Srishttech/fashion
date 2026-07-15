"""
Loads the prebuilt FAISS index + metadata.json once at API startup, and
serves search / incremental-add from memory. This is the ONLY thing the
deployed app touches at request time — no model re-encoding of the dataset.
"""
import json
import threading

import faiss
import numpy as np

from utils.config import FAISS_INDEX_PATH, METADATA_PATH, EMBEDDING_DIM


class FaissStore:
    def __init__(self, index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self._lock = threading.Lock()  # guards add_image / persistence
        self._load()

    def _load(self):
        try:
            self.index = faiss.read_index(self.index_path)
        except RuntimeError:
            self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        try:
            with open(self.metadata_path) as f:
                self.metadata = json.load(f)
        except FileNotFoundError:
            self.metadata = {}
        # row -> image_id reverse lookup, rebuilt from metadata on load
        self.row_to_id = {v["faiss_row"]: k for k, v in self.metadata.items()}

    def search(self, query_embedding, top_k):
        """query_embedding: (512,) normalized numpy vector."""
        if self.index.ntotal == 0:
            return []
        scores, rows = self.index.search(query_embedding.reshape(1, -1), top_k)
        results = []
        for score, row in zip(scores[0], rows[0]):
            if row == -1:
                continue
            image_id = self.row_to_id.get(int(row))
            if image_id is None:
                continue
            results.append({
                "image_id": image_id,
                "embedding_similarity": float(score),  # cosine, since IP + normalized
                **self.metadata[image_id],
            })
        return results

    def add_image(self, image_id, embedding, path, attributes):
        """Append one new image without rebuilding the whole index."""
        with self._lock:
            new_row = self.index.ntotal
            self.index.add(embedding.reshape(1, -1))
            self.metadata[image_id] = {
                "path": path,
                "attributes": attributes,
                "faiss_row": new_row,
            }
            self.row_to_id[new_row] = image_id
            self._persist()

    def _persist(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)


# module-level singleton the FastAPI app imports directly
store = FaissStore()
