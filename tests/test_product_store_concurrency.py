from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor

from shopping_agent.product_store import ProductLineStore


def test_parallel_delegates_read_the_correct_product(tmp_path):
    path = tmp_path / "products.jsonl"
    path.write_text("".join(json.dumps({"product_id": str(i), "title": f"Product {i}"}) + "\n"
                            for i in range(1000)), encoding="utf-8")
    store = ProductLineStore(path)
    rng = random.Random(20260928)
    requested = [rng.randrange(1000) for _ in range(10000)]
    with ThreadPoolExecutor(max_workers=16) as pool:
        observed = list(pool.map(lambda i: store[i].product_id, requested))
    assert observed == [str(i) for i in requested]
