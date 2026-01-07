"""
clip_encoder.py
----------------

This module provides all CLIP-based image encoding functionality for the project.
It performs the following tasks:

1. Extracts image URLs from a metadata JSONL file.
2. Downloads and resizes images to 224x224.
3. Encodes each item's images using CLIP's ViT-B/32 model.
4. Averages embeddings if an item has multiple images.
5. Returns a dictionary mapping:
       asin -> 512-dimensional CLIP embedding (torch.Tensor)

Used for:
    - CLIP Baseline (image-only retrieval)
    - BLaIR-MM (text + image multimodal fusion model)
"""

import json
import requests
import torch
from transformers import CLIPProcessor, CLIPVisionModel
from PIL import Image
from io import BytesIO
from typing import Dict, List, Optional


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
CLIP_MODEL.eval()
CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def download_and_resize_image(
    url: str, size: tuple = (224, 224)
) -> Optional[Image.Image]:
    """
    Download an image from a URL and resize it.
    :param url: URL of the image.
    :param size: Desired output size for CLIP (224x224).
    :return: PIL Image or None if failed.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = img.resize(size, Image.BICUBIC)
        return img
    except Exception as e:
        print(f"[CLIP] Failed to process {url}: {e}")
        return None


def create_image_lists(
    dataset_path: str, max_items: int = 500000
) -> Dict[str, List[Image.Image]]:
    """
    Extract lists of PIL images (resized) for each item in the dataset.

    :param dataset_path: Path to JSONL metadata file.
    :param max_items: Limit number of datapoints (useful for debugging).
    :return: dict: asin -> list of PIL.Image objects
    """
    item_images = {}

    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_items:
                break

            data = json.loads(line)
            asin = data.get("asin")
            if not asin:
                continue

            images = data.get("images", [])
            image_list = []

            for img_info in images:
                for key in [
                    "hi_res",
                    "large",
                    "thumb",
                    "large_image_url",
                    "medium_image_url",
                    "small_image_url",
                ]:
                    url = img_info.get(key)
                    if url:
                        img = download_and_resize_image(url)
                        if img:
                            image_list.append(img)

            if image_list:
                item_images[asin] = image_list

    return item_images


def encode_images_with_clip(image_list: List[Image.Image]) -> Optional[torch.Tensor]:
    """
    Encode a list of PIL images into a 512-d averaged CLIP image embedding.
    :param image_list: List of PIL.Image objects.
    :return: torch.Tensor of shape [512] or None
    """
    if not image_list:
        return None

    embeddings = []

    for img in image_list:
        try:
            inputs = CLIP_PROCESSOR(images=img, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                outputs = CLIP_MODEL(**inputs)
                # Use pooler_output which is the CLIP image embedding
                emb = outputs.pooler_output  # [1, 512]
                emb = emb / emb.norm(dim=-1, keepdim=True)  # L2 normalize
                embeddings.append(emb.cpu())

        except Exception as e:
            print(f"[CLIP] Encoding failed: {e}")

    if not embeddings:
        return None

    # Average embeddings for this item
    return torch.mean(torch.cat(embeddings, dim=0), dim=0)  # shape: [512]


def encode_images_for_all_items(
    metadata_path: str, max_items: int = 500000
) -> Dict[str, torch.Tensor]:
    """
    High-level wrapper that:
        1. Extracts images for each ASIN
        2. Encodes all images into a single CLIP embedding per ASIN

    :param metadata_path: Path to metadata JSONL file.
    :param max_items: Optional limit for debugging.
    :return: dict: asin -> 512-d CLIP embedding
    """
    print(f"\n[CLIP] Extracting image lists from: {metadata_path}")
    item_images = create_image_lists(metadata_path, max_items=max_items)

    print(f"[CLIP] Found {len(item_images)} items with images. Encoding...")

    clip_embeddings = {}
    for asin, img_list in item_images.items():
        emb = encode_images_with_clip(img_list)
        if emb is not None:
            clip_embeddings[asin] = emb

    print(f"[CLIP] Finished encoding {len(clip_embeddings)} items.\n")
    return clip_embeddings


def main():
    TEST_PATH = "./training_datasets/meta_Appliances.jsonl"

    print("[CLIP] Testing image extraction...")
    item_imgs = create_image_lists(TEST_PATH, max_items=5)
    print(f"Extracted images for {len(item_imgs)} items\n")

    print("[CLIP] Testing CLIP encoding...")
    clip_embs = encode_images_for_all_items(TEST_PATH, max_items=5)

    for asin, emb in clip_embs.items():
        print(f"ASIN: {asin}")
        print(f"Embedding shape: {emb.shape}")
        print(f"Sample values: {emb[:5]}")


if __name__ == "__main__":
    main()
