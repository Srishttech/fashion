"""
Offline indexing pipeline. Run this ONCE (or whenever the dataset changes)
on a machine with normal internet access (laptop / Colab), NOT on Render.

    python -m indexer.build_index --images-dir data/images --limit 800

Output (both committed to the repo / uploaded to Render's disk):
    data/faiss.index   - flat IP index over L2-normalized embeddings
    data/metadata.json - {image_id: {path, attributes}} for every indexed image

The deployed FastAPI app only ever reads these two files. It never calls
encode_image() on the dataset again.
"""
import argparse
import json
import os
import sys

import numpy as np
import faiss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import FAISS_INDEX_PATH, METADATA_PATH, EMBEDDING_DIM
from utils.image_utils import list_images, load_image, image_id_from_path
from indexer.encoder import encode_images_batch
from indexer.attribute_tagger import tag_image


def build_index(images_dir, limit=None, batch_size=32):
    paths = list_images(images_dir)
    if limit:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(f"No images found in {images_dir}")

    print(f"Indexing {len(paths)} images from {images_dir} ...")

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    metadata = {}

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        images = [load_image(p) for p in batch_paths]
        embeddings = encode_images_batch(images, batch_size=batch_size)  # (B, 512)

        index.add(embeddings)

        for path, emb in zip(batch_paths, embeddings):
            image_id = image_id_from_path(path)
            attributes = tag_image(emb)
            metadata[image_id] = {
                "path": os.path.relpath(path, os.path.dirname(images_dir)),
                "attributes": attributes,
                # faiss_row lets us map a search hit back to metadata even
                # after future appends (rows are assigned in insertion order)
                "faiss_row": index.ntotal - len(batch_paths) + list(batch_paths).index(path),
            }

        done = min(start + batch_size, len(paths))
        print(f"  {done}/{len(paths)} encoded + tagged", flush=True)

    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done. Index -> {FAISS_INDEX_PATH} ({index.ntotal} vectors)")
    print(f"Metadata -> {METADATA_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--limit", type=int, default=None,
                         help="Cap number of images (assignment suggests 500-1000)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    build_index(args.images_dir, limit=args.limit, batch_size=args.batch_size)
