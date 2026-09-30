from shopping_agent.models import Product
from shopping_agent.rrf import fuse


def test_bm25_weight_can_promote_lexical_top_result():
    lexical = Product("lexical", "exact model")
    semantic = Product("semantic", "similar item")
    equal = fuse([(lexical, 1.0)], [(semantic, 1.0)], top_k=2)
    weighted = fuse([(lexical, 1.0)], [(semantic, 1.0)], weights=(1.5, 1.0), top_k=2)
    assert {p.product_id for p, _ in equal} == {"lexical", "semantic"}
    assert weighted[0][0].product_id == "lexical"
