"""Small image I/O helpers shared by indexer, retriever, and backend."""
import os
from PIL import Image


def load_image(path):
    """Load an image as RGB, raising a clear error if the file is bad."""
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        raise ValueError(f"Could not load image at {path}: {e}")


def list_images(directory, extensions=(".jpg", ".jpeg", ".png", ".webp")):
    """Return sorted absolute paths of all images in a directory."""
    files = []
    for fname in sorted(os.listdir(directory)):
        if fname.lower().endswith(extensions):
            files.append(os.path.join(directory, fname))
    return files


def image_id_from_path(path):
    """Filename without extension is used as the stable image_id everywhere."""
    return os.path.splitext(os.path.basename(path))[0]
