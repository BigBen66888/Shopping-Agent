"""BGE-reranker-v2-m3 cross-encoder on CPU."""
from __future__ import annotations

from .model_paths import resolve_model
from .models import Product
from .quantize import quantize_enabled
from .settings import settings


class MainAgentReranker:
    def __init__(self):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("请安装 sentence-transformers") from exc
        self.name = settings.reranker_model
        local_model = settings.data_dir / "models" / "bge-reranker-v2-m3"
        self.model = CrossEncoder(resolve_model(self.name, local_model), device="cpu", trust_remote_code=True, max_length=256)
        if quantize_enabled(settings.model_quantize):
            from .quantize import quantize_cross_encoder

            quantize_cross_encoder(self.model)

    def rerank(self, query: str, candidates: list[tuple[Product, float]], budget: float | None = None, top_k: int = 8) -> list[tuple[Product, float]]:
        eligible = [(p, score) for p, score in candidates if budget is None or p.price <= budget][:20]
        if not eligible:
            return []
        pairs = [(query, f"{p.title} {p.brand} {p.category} {p.description}") for p, _ in eligible]
        scores = self.model.predict(pairs, batch_size=4, show_progress_bar=False)
        ranked = [(p, float(score)) for (p, _), score in zip(eligible, scores)]
        ranked.sort(key=lambda item: (-item[1], item[0].price))
        return ranked[:top_k]
