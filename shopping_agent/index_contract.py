"""Shared corpus ordering and embedding settings for the Kaggle build and local search."""
from __future__ import annotations

import hashlib
from typing import Iterable

from .models import Product


EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_MAX_LENGTH = 128
INDEX_FORMAT = 1


def product_text(product: Product) -> str:
    return f"{product.title} {product.brand} {product.category} {product.description}"


def corpus_signature(products: Iterable[Product], model_name: str = EMBEDDING_MODEL) -> str:
    digest = hashlib.sha256(f"{model_name}:maxlen{EMBEDDING_MAX_LENGTH}:v2".encode())
    for product in products:
        digest.update(f"\n{product.product_id}\t{product.title}\t{product.brand}\t{product.category}\t{product.description}".encode())
    return digest.hexdigest()
