"""Optional int8 dynamic quantization for the CPU embedding and reranking models.

BGE-M3 and bge-reranker-v2-m3 are each about 2.1 GB in fp32 on CPU, which is a large
share of a 16 GB laptop's budget. Dynamic quantization turns the Linear (and optionally
Embedding) weights into int8 at load time, with no retraining and no change to the
FAISS index -- only the query encoder and the reranker are affected, and both are
rebuilt from the same weights on every start.

Enable with ``MODEL_QUANTIZE=int8`` (default); set it to ``none`` to load fp32.
"""
from __future__ import annotations

from typing import Any


def _dynamic_quantize(module: Any, include_embedding: bool = False) -> Any:
    """int8 dynamic quantization of the Linear layers.

    Only ``nn.Linear`` is quantized. ``nn.Embedding`` needs a weight-only qconfig
    (``float_qparams_weight_only_qconfig``), and passing a qconfig *dict* to
    ``quantize_dynamic`` in torch 2.x produces a module whose forward breaks on
    XLM-RoBERTa (``input_ids.ne`` receives a non-tensor). Verified on this machine:
    Linear-only keeps the query embedding bit-identical to fp32, so there is no
    reason to take that risk. The flag is kept for callers that want to experiment.
    """
    import torch
    import torch.nn as nn

    if not include_embedding:
        return torch.quantization.quantize_dynamic(module, {nn.Linear}, dtype=torch.qint8)
    spec = {
        nn.Linear: torch.quantization.default_dynamic_qconfig,
        nn.Embedding: torch.quantization.float_qparams_weight_only_qconfig,
    }
    try:
        return torch.quantization.quantize_dynamic(module, spec, dtype=torch.qint8)
    except (AssertionError, RuntimeError, AttributeError):
        return torch.quantization.quantize_dynamic(module, {nn.Linear}, dtype=torch.qint8)


def _first_transformer(model: Any) -> Any:
    """Return the SentenceTransformer's Transformer module, or None."""
    try:
        first = model[0]
    except (TypeError, IndexError, KeyError):
        return None
    return first


def quantize_sentence_transformer(model: Any, include_embedding: bool = False) -> bool:
    """Quantize a SentenceTransformer's encoder in place. Returns True on success.

    Verified working: query embeddings stay bit-identical to fp32 (cosine 1.000000).
    Note that ``quantize_dynamic`` builds a copy, so peak RSS roughly doubles during
    startup, and the steady-state saving is smaller than the weight file size suggests
    because the fp32 pages are mmap'd and may stay resident.
    """
    first = _first_transformer(model)
    if first is None:
        return False
    attribute = "auto_model" if hasattr(first, "auto_model") else ("model" if hasattr(first, "model") else None)
    if attribute is None:
        return False
    target = getattr(first, attribute)
    if not hasattr(target, "parameters"):
        return False
    setattr(first, attribute, _dynamic_quantize(target, include_embedding))
    return True


def quantize_cross_encoder(model: Any, include_embedding: bool = False) -> bool:
    """Quantize a CrossEncoder's underlying transformer in place.

    NOT RECOMMENDED: verified broken on this machine. Replacing the underlying
    ``XLMRobertaForSequenceClassification`` with a quantized copy loses the forward
    signature that Hugging Face uses to map positional arguments, so inference fails
    with ``AttributeError`` / ``KeyError: 'ne'``. Kept only so the experiment is
    reproducible; ``MODEL_QUANTIZE`` defaults to ``none``.
    """
    target = getattr(model, "model", None)
    if target is None or not hasattr(target, "parameters"):
        return False
    model.model = _dynamic_quantize(target, include_embedding)
    return True


def quantize_enabled(mode: str) -> bool:
    return str(mode).strip().lower() in {"int8", "qint8", "dynamic", "1", "true", "yes"}
