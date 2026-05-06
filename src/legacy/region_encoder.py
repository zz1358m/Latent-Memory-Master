"""
Image Encoder — Whole-image downsampling
=========================================
Each image is resized to 1/20 × 1/20 of its original dimensions (area → 1/400)
and passed as a single sample to LLaVA's vision tower.

No region splitting. One image → one latent token in the MemoryBank.

Usage
-----
>>> from src.region_encoder import resize_image
>>> from PIL import Image
>>> img = Image.open("path/to/image.jpg").convert("RGB")
>>> small = resize_image(img, scale=1/20)
"""

from PIL import Image


def resize_image(image: Image.Image, scale: float = 1 / 20) -> Image.Image:
    """
    Downscale a PIL Image by `scale` in each dimension.
    Default 1/20 per side → area reduced to 1/400 of original.
    """
    w, h = image.size
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    return image.resize((new_w, new_h), Image.LANCZOS)


def image_doc_id(image_id: str) -> str:
    """Canonical doc_id for an image in the MemoryBank."""
    return str(image_id)
