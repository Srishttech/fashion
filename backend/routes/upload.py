import os
import uuid

from fastapi import APIRouter, UploadFile, File
from PIL import Image

from indexer.encoder import encode_image
from indexer.attribute_tagger import tag_image
from retriever.faiss_store import store
from utils.config import IMAGES_DIR

router = APIRouter()


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    image_id = uuid.uuid4().hex
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    save_path = os.path.join(IMAGES_DIR, f"{image_id}{ext}")

    os.makedirs(IMAGES_DIR, exist_ok=True)
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    pil_image = Image.open(save_path).convert("RGB")
    embedding = encode_image(pil_image)
    attributes = tag_image(embedding)

    store.add_image(
        image_id=image_id,
        embedding=embedding,
        path=os.path.relpath(save_path, os.path.dirname(IMAGES_DIR)),
        attributes=attributes,
    )

    return {"image_id": image_id, "attributes": attributes, "status": "indexed"}
