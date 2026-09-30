from shopping_agent.agents.search_agent import SearchAgent
from shopping_agent.models import Product


def test_natural_category_phrase_uses_taxonomy_noun():
    keyboard = Product("k1", "Mechanical Keyboard", category="Computer Accessories > Keyboard & Mouse > Gaming Keyboards")
    mouse = Product("m1", "Wireless Mouse", category="Computer Accessories > Keyboard & Mouse > Basic Mice")

    class Retriever:
        def search(self, query, top_k, budget):
            return [(keyboard, 1.0), (mouse, 0.9)], {"rrf_count": 2}

    class Reranker:
        def rerank(self, query, rows, budget, top_k):
            return rows[:top_k]

    agent = SearchAgent(Retriever(), reranker=Reranker())
    keyboard_rows, _ = agent._retrieve("mechanical keyboard", 8, 1500, category="mechanical keyboard")
    mouse_rows, _ = agent._retrieve("wireless mouse", 8, 1500, category="wireless mouse")

    assert [product.product_id for product, _ in keyboard_rows] == ["k1"]
    assert [product.product_id for product, _ in mouse_rows] == ["m1"]


def test_explicit_color_rejects_unverified_color():
    white = Product("w1", "Travel Backpack", description="Available in white or grey", category="Bags > Backpacks")
    black = Product("b1", "Business Backpack", description="Available in black", category="Bags > Backpacks")

    class Retriever:
        def search(self, query, top_k, budget):
            return [(black, 1.0), (white, 0.9)], {"rrf_count": 2}

    class Reranker:
        def rerank(self, query, rows, budget, top_k):
            return rows[:top_k]

    agent = SearchAgent(Retriever(), reranker=Reranker())
    rows, _ = agent._retrieve("white backpack", 8, 1000, color="white")
    assert [product.product_id for product, _ in rows] == ["w1"]
