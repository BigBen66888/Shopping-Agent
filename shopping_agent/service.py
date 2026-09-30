from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from .agents import MainAgent, SearchAgent, ShopAgent
from .database import Database, utc_now
from .dataset import preprocess
from .memory.context_manager import ContextManager, MemoryExtractor
from .memory.conflicts import memory_slot
from .product_store import ProductLineStore, cache_key
from .hybrid import HybridRetriever
from .reranker import MainAgentReranker
from types import SimpleNamespace
from .settings import settings
from .deepseek import DeepSeekClient


class ShoppingService:
    """应用服务：会话、检索、购物状态和本地模拟交易的唯一入口。"""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or settings.data_dir
        self.processed = self.data_dir / "products.jsonl"
        self.agent = None
        self.main_agent: MainAgent | None = None
        self.store: ProductLineStore | None = None
        self.db = Database(settings.database)
        self.context_manager = ContextManager()
        self.memory_extractor = MemoryExtractor()

    @property
    def bm25_cache(self) -> Path:
        """Derived from ``processed`` rather than frozen in __init__.

        The BM25 cache belongs beside the corpus it was built from, so redirecting
        ``processed`` (as the tests do) also redirects the cache instead of writing
        into the real data directory.
        """
        return self.processed.parent / "bm25_cache.pkl"

    def prepare(self, limit: int | None = None) -> dict:
        if not self.processed.exists():
            result = preprocess(self.data_dir, self.processed, limit)
        else:
            result = {"output": str(self.processed), "cached": True}

        # 懒加载：只记每行的字节偏移，检索需要哪条才解码哪条。
        # 全量 275 万商品若整份载入内存约 6 GB，懒加载后偏移表只占约 22 MB。
        started = time.perf_counter()
        self.store = ProductLineStore(self.processed, with_attributes=settings.load_attributes)
        result["records"] = len(self.store)
        result["offset_scan_seconds"] = round(time.perf_counter() - started, 2)

        hybrid = HybridRetriever(self.store, bm25_cache=self.bm25_cache,
                                 bm25_cache_key=cache_key(self.processed))
        reranker = MainAgentReranker()
        self.agent = SimpleNamespace(hybrid=hybrid, reranker=reranker)
        self.main_agent = MainAgent(SearchAgent(hybrid, self, reranker), ShopAgent(self))
        result["index_ready"] = True
        return result

    def architecture(self) -> dict:
        return {
            "main_agent": {"role": "route, delegate and aggregate", "ready": self.main_agent is not None},
            "search_agent": {"permissions": sorted(self.main_agent.search_agent.permissions) if self.main_agent else [], "tools": self.main_agent.search_agent.registry.describe() if self.main_agent else [], "violations": self.main_agent.search_agent.registry.permission_violations if self.main_agent else 0, "rate_limit_violations": self.main_agent.search_agent.registry.rate_limit_violations if self.main_agent else 0, "failures": self.main_agent.search_agent.registry.failures if self.main_agent else 0},
            "shop_agent": {"permissions": sorted(self.main_agent.shop_agent.permissions) if self.main_agent else [], "tools": self.main_agent.shop_agent.registry.describe() if self.main_agent else [], "violations": self.main_agent.shop_agent.registry.permission_violations if self.main_agent else 0, "rate_limit_violations": self.main_agent.shop_agent.registry.rate_limit_violations if self.main_agent else 0, "failures": self.main_agent.shop_agent.registry.failures if self.main_agent else 0},
        }

    def model_dispatch(self, session_id: str, query: str) -> dict:
        if self.main_agent is None:
            self.prepare()
        assert self.main_agent is not None
        return self.main_agent.dispatch_model(DeepSeekClient(), query, session_id)

    def recommend(self, query: str, top_k: int | None = None, context: dict | None = None):
        session_id = self.create_session()["id"]
        return self.recommend_for_session(session_id, query, top_k)

    def _record_interaction(self, query: str, result) -> None:
        with self.db.connect() as db:
            db.execute("INSERT INTO interactions(query,response) VALUES (?,?)", (query, json.dumps(result.to_dict(), ensure_ascii=False)))

    def create_session(self, title: str = "新购物会话", user_id: str = "demo") -> dict:
        sid = uuid.uuid4().hex[:16]
        now = utc_now()
        with self.db.connect() as db:
            db.execute("INSERT INTO sessions(id,user_id,title,created_at,updated_at) VALUES (?,?,?,?,?)", (sid, user_id, title, now, now))
            db.execute("INSERT INTO shopping_state VALUES (?,?,?)", (sid, Database.dumps({"query": "", "budget": None, "category": "", "candidate_ids": [], "selected_ids": []}), now))
        return {"id": sid, "user_id": user_id, "title": title, "created_at": now, "updated_at": now}

    def list_sessions(self) -> list[dict]:
        with self.db.connect() as db:
            return [dict(row) for row in db.execute("SELECT id,user_id,title,created_at,updated_at FROM sessions ORDER BY updated_at DESC").fetchall()]

    def delete_session(self, session_id: str) -> dict:
        """Delete one conversation while preserving single-user shopping data."""
        with self.db.connect() as db:
            exists = db.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone()
            if not exists:
                raise KeyError("会话不存在")
            list_ids = [row[0] for row in db.execute("SELECT id FROM shopping_lists WHERE session_id=?", (session_id,)).fetchall()]
            for list_id in list_ids:
                db.execute("DELETE FROM shopping_list_items WHERE list_id=?", (list_id,))
            db.execute("DELETE FROM shopping_lists WHERE session_id=?", (session_id,))
            db.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            db.execute("DELETE FROM tool_traces WHERE session_id=?", (session_id,))
            db.execute("DELETE FROM run_metrics WHERE session_id=?", (session_id,))
            db.execute("DELETE FROM shopping_state WHERE session_id=?", (session_id,))
            db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        return {"deleted": session_id}

    def clear_session(self, session_id: str) -> dict:
        """Clear the conversation in place, leaving shared shopping data intact."""
        now = utc_now()
        empty_state = {"query": "", "budget": None, "category": "", "candidate_ids": [], "selected_ids": []}
        with self.db.connect() as db:
            if not db.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone():
                raise KeyError("会话不存在")
            db.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            db.execute("DELETE FROM tool_traces WHERE session_id=?", (session_id,))
            db.execute("DELETE FROM run_metrics WHERE session_id=?", (session_id,))
            db.execute("UPDATE shopping_state SET state=?,updated_at=? WHERE session_id=?", (Database.dumps(empty_state), now, session_id))
            db.execute("UPDATE sessions SET title=?,updated_at=? WHERE id=?", ("新购物会话", now, session_id))
        return {"cleared": session_id}

    def _user_id(self, session_id: str) -> str:
        with self.db.connect() as db:
            row = db.execute("SELECT user_id FROM sessions WHERE id=?", (session_id,)).fetchone()
        return str(row[0]) if row else "demo"

    def _shopping_scope(self, session_id: str) -> str:
        """All carts, favorites and orders belong to the one local user."""
        return f"user:{self._user_id(session_id)}"

    def _state(self, session_id: str) -> dict:
        with self.db.connect() as db:
            row = db.execute("SELECT state FROM shopping_state WHERE session_id=?", (session_id,)).fetchone()
        return Database.loads(row[0], {}) if row else {}

    def session_state(self, session_id: str) -> dict:
        owner = self._shopping_scope(session_id)
        with self.db.connect() as db:
            if not db.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone():
                raise KeyError("会话不存在")
            messages = db.execute("SELECT id,role,content,products,trace,created_at FROM messages WHERE session_id=? ORDER BY id", (session_id,)).fetchall()
            cart = db.execute("SELECT product,quantity FROM cart WHERE session_id=? ORDER BY rowid", (owner,)).fetchall()
            favorites = db.execute("SELECT product FROM favorites WHERE session_id=? ORDER BY created_at DESC", (owner,)).fetchall()
            orders = db.execute("SELECT id,status,items,created_at,confirmed_at FROM orders WHERE session_id=? ORDER BY created_at DESC", (owner,)).fetchall()
            user_id = self._user_id(session_id)
            prefs = db.execute("SELECT id,kind,content,'memory' AS source,status,created_at FROM memory WHERE user_id=? AND status='ACTIVE' ORDER BY created_at DESC,id DESC", (user_id,)).fetchall()
            trace_count = db.execute("SELECT count(*) FROM tool_traces WHERE session_id=?", (session_id,)).fetchone()[0]
            list_rows = db.execute("SELECT i.product,i.quantity FROM shopping_list_items i JOIN shopping_lists l ON l.id=i.list_id WHERE l.session_id=? ORDER BY i.rowid", (session_id,)).fetchall()
        return {
            "messages": [{"id": r[0], "role": r[1], "content": r[2], "products": Database.loads(r[3], []), "trace": Database.loads(r[4], []), "created_at": r[5]} for r in messages],
            "cart": [{**Database.loads(r[0], {}), "quantity": r[1]} for r in cart],
            "favorites": [Database.loads(r[0], {}) for r in favorites],
            "orders": [{"id": r[0], "status": r[1], "items": Database.loads(r[2], []), "created_at": r[3], "confirmed_at": r[4]} for r in orders],
            "preferences": [dict(r) for r in prefs],
            "trace_count": trace_count,
            "shopping_list": [{**Database.loads(r[0], {}), "quantity": r[1]} for r in list_rows],
            "shopping_state": self._state(session_id),
        }

    def _recent_user_query(self, session_id: str) -> str:
        with self.db.connect() as db:
            row = db.execute("SELECT content FROM messages WHERE session_id=? AND role='user' ORDER BY id DESC LIMIT 1", (session_id,)).fetchone()
        return str(row[0]) if row else ""

    def recommend_for_session(self, session_id: str, query: str, top_k: int | None = None):
        result = None
        for event in self.recommend_for_session_stream(session_id, query, top_k):
            if event.get("type") == "done":
                result = event["recommendation"]
        if result is None:
            raise RuntimeError("模型未完成响应")
        return result

    def _session_context(self, session_id: str, query: str, client=None) -> tuple[str, dict, dict]:
        previous = self._recent_user_query(session_id)
        contextual_query = f"{previous}；本轮补充：{query}" if previous and any(word in query for word in ("刚才", "上面", "便宜", "第二", "换一个", "前面")) else query
        state = self._state(session_id)
        compacted_through_id = int(state.get("context_compacted_through_id", 0) or 0)
        with self.db.connect() as db:
            recent = [dict(row) for row in db.execute(
                "SELECT id,role,content FROM messages WHERE session_id=? AND id>? ORDER BY id",
                (session_id, compacted_through_id)).fetchall()]
        long_term = self.long_term_memory(session_id)
        category = str(state.get("category", "")).lower()
        context = self.context_manager.build(
            recent, state, (), state.get("context_summary", ""), long_term=long_term,
            client=client, compacted_through_id=compacted_through_id,
        )
        return contextual_query, state, context

    # ---- 长期记忆（问题 2）----

    def long_term_memory(self, session_id: str, limit: int | None = None) -> list[dict]:
        """读取仍然有效的跨会话记忆；最近写入的优先。"""
        user_id = self._user_id(session_id)
        with self.db.connect() as db:
            rows = db.execute(
                "SELECT id,kind,content,created_at FROM memory WHERE user_id=? AND status='ACTIVE' ORDER BY created_at DESC,id DESC LIMIT ?",
                (user_id, limit or settings.memory_max_injected * 3),
            ).fetchall()
        items = [dict(row) for row in rows]
        return items[: (limit or settings.memory_max_injected)]

    def remember(self, session_id: str, kind: str, content: str, source: str = "agent") -> dict:
        """写入长期记忆；同一商品主题和偏好槽位由最新值覆盖。"""
        text = (content or "").strip()
        if not text:
            return {"saved": False, "reason": "empty"}
        user_id = self._user_id(session_id)
        slot = memory_slot(kind, text)
        with self.db.connect() as db:
            superseded = False
            if slot:
                active = db.execute(
                    "SELECT id,kind,content FROM memory WHERE user_id=? AND status='ACTIVE'",
                    (user_id,),
                ).fetchall()
                for old in active:
                    old_slot = memory_slot(str(old[1]), str(old[2]))
                    if old_slot and old_slot[:2] == slot[:2] and str(old[2]) != text:
                        db.execute("UPDATE memory SET status='SUPERSEDED' WHERE id=?", (old[0],))
                        superseded = True
            existing = db.execute(
                "SELECT id,status FROM memory WHERE user_id=? AND kind=? AND content=?", (user_id, kind, text)
            ).fetchone()
            if existing:
                if existing[1] in {"DELETED", "SUPERSEDED"}:
                    db.execute("UPDATE memory SET status='ACTIVE',source_session=?,created_at=? WHERE id=?", (session_id, utc_now(), existing[0]))
                    return {"saved": True, "id": existing[0], "kind": kind, "content": text}
                if superseded:
                    db.execute("UPDATE memory SET source_session=?,created_at=? WHERE id=?", (session_id, utc_now(), existing[0]))
                    return {"saved": True, "id": existing[0], "kind": kind, "content": text}
                return {"saved": False, "reason": "duplicate", "id": existing[0]}
            db.execute(
                "INSERT INTO memory(user_id,kind,content,source_session,created_at) VALUES (?,?,?,?,?)",
                (user_id, kind, text, session_id, utc_now()),
            )
            row = db.execute("SELECT last_insert_rowid()").fetchone()
        return {"saved": True, "id": row[0], "kind": kind, "content": text}

    def forget(self, session_id: str, memory_id: int) -> dict:
        with self.db.connect() as db:
            db.execute("UPDATE memory SET status='DELETED' WHERE user_id=? AND id=?", (self._user_id(session_id), memory_id))
        return {"deleted": memory_id}

    def clear_memory(self, session_id: str) -> dict:
        with self.db.connect() as db:
            db.execute("UPDATE memory SET status='DELETED' WHERE user_id=?", (self._user_id(session_id),))
        return {"cleared": True}

    def capture_memory(self, session_id: str, query: str, answer: str, client: Any) -> list[dict]:
        """Use the configured LLM on every completed turn; [] means no durable preference."""
        items = self.memory_extractor.extract(query, answer, client)
        saved = []
        for item in items:
            result = self.remember(session_id, item["kind"], item["content"], source="agent")
            if result.get("saved"):
                saved.append(result)
        return saved

    def _recent_products(self, session_id: str) -> list[dict]:
        with self.db.connect() as db:
            row = db.execute("SELECT products FROM messages WHERE session_id=? AND role='assistant' AND products<>'[]' ORDER BY id DESC LIMIT 1", (session_id,)).fetchone()
        return Database.loads(row[0], []) if row else []

    def _save_session_result(self, session_id: str, query: str, state: dict, context: dict, result) -> None:
        now = utc_now()
        run_id = uuid.uuid4().hex[:12]
        payload = [p.to_dict() for p in result.products]
        state.update({"query": query, "budget": result.evidence.get("budget"), "category": (result.products[0].category if result.products else state.get("category", "")), "candidate_ids": [p.product_id for p in result.products] if result.products else state.get("candidate_ids", []), "selected_ids": state.get("selected_ids", [])})
        with self.db.connect() as db:
            # A result finishing after a deletion must not recreate its session.
            if not db.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone():
                return
            db.execute("UPDATE sessions SET title=?,updated_at=? WHERE id=?", (query[:40], now, session_id))
            db.execute("INSERT OR REPLACE INTO shopping_state VALUES (?,?,?)", (session_id, Database.dumps(state), now))
            db.execute("INSERT INTO messages(session_id,role,content,products,trace,created_at) VALUES (?,?,?,?,?,?)", (session_id, "user", query, "[]", "[]", now))
            db.execute("INSERT INTO messages(session_id,role,content,products,trace,created_at) VALUES (?,?,?,?,?,?)", (session_id, "assistant", result.answer, Database.dumps(payload), Database.dumps(result.trace), now))
            state["context_summary"] = context.get("summary", "")
            state["context_compacted_through_id"] = context.get("compacted_through_id", 0)
            db.execute("UPDATE shopping_state SET state=?,updated_at=? WHERE session_id=?", (Database.dumps(state), now, session_id))
            for item in result.trace:
                role = item.get("role", "ai")
                db.execute("INSERT INTO tool_traces(session_id,run_id,tool,args,result_count,elapsed_ms,created_at) VALUES (?,?,?,?,?,?,?)", (session_id, run_id, role, item.get("content", "")[:2000], len(payload) if role == "tool" else 0, 0, now))
            usage = result.evidence.get("usage", {})
            timing = result.evidence.get("timing_ms", {})
            db.execute(
                "INSERT INTO run_metrics(session_id,run_id,input_tokens,output_tokens,api_cost,compression_count,retrieval_ms,rrf_ms,rerank_ms,errors,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    session_id, run_id, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)),
                    float(result.evidence.get("api_cost", 0)), int(result.evidence.get("compression_count", 0)),
                    float(timing.get("lexical", 0)) + float(timing.get("bm25", 0)) + float(timing.get("vector", 0)),
                    float(timing.get("rrf", 0)), float(timing.get("rerank", 0)),
                    int(result.evidence.get("errors", 0)), now,
                ),
            )

    def recommend_for_session_stream(self, session_id: str, query: str, top_k: int | None = None):
        """The API model is required; failures propagate to the stream error event."""
        if self.main_agent is None:
            self.prepare()
        client = DeepSeekClient()
        contextual_query, state, context = self._session_context(session_id, query, client)
        stream = self.main_agent.recommend_stream(
            client, contextual_query, session_id, context, routing_query=query,
        )
        while True:
            try:
                yield next(stream)
            except StopIteration as completed:
                result = completed.value
                try:
                    saved = self.capture_memory(session_id, query, result.answer, client)
                except RuntimeError as exc:
                    # Memory extraction is supplementary. A malformed model reply
                    # must not discard an otherwise completed shopping turn.
                    saved = []
                    result.evidence["memory_capture_error"] = str(exc)[:200]
                for item in saved:
                    content = "Memory · 已记录长期记忆「" + item["content"] + "」"
                    result.trace.append({"role": "event", "content": content, "agent": "Memory"})
                    yield {"type": "trace", "content": content, "agent": "Memory"}
                self._record_interaction(contextual_query, result)
                self._save_session_result(session_id, query, state, context, result)
                yield {"type": "done", "result": result.to_dict(), "recommendation": result}
                return

    def get_memory(self, session_id: str) -> dict:
        items = self.long_term_memory(session_id, limit=1000)
        return {"items": items, "count": len(items)}

    def list_items(self, session_id: str) -> dict:
        with self.db.connect() as db:
            rows = db.execute("SELECT i.product,i.quantity FROM shopping_list_items i JOIN shopping_lists l ON l.id=i.list_id WHERE l.session_id=? ORDER BY i.rowid", (session_id,)).fetchall()
        items = [{**Database.loads(row[0], {}), "quantity": row[1]} for row in rows]
        return {"items": items, "count": len(items), "total_quantity": sum(item["quantity"] for item in items)}

    def get_cart(self, session_id: str) -> dict:
        items = self.session_state(session_id)["cart"]
        return {"items": items, "count": len(items), "total_quantity": sum(item["quantity"] for item in items), "total_price": round(sum(float(item.get("price", 0)) * item["quantity"] for item in items), 2)}

    def get_orders(self, session_id: str) -> dict:
        orders = self.session_state(session_id)["orders"]
        return {"items": orders, "count": len(orders)}

    def get_favorites(self, session_id: str) -> dict:
        items = self.session_state(session_id)["favorites"]
        return {"items": items, "count": len(items)}

    def add_to_list(self, session_id: str, product: dict, name: str = "默认清单") -> dict:
        with self.db.connect() as db:
            db.execute("INSERT OR IGNORE INTO shopping_lists(session_id,name) VALUES (?,?)", (session_id, name))
            row = db.execute("SELECT id FROM shopping_lists WHERE session_id=? AND name=?", (session_id, name)).fetchone()
            db.execute("INSERT INTO shopping_list_items(list_id,product_id,product,quantity) VALUES (?,?,?,1) ON CONFLICT(list_id,product_id) DO UPDATE SET quantity=quantity+1", (row[0], product.get("product_id", ""), Database.dumps(product)))
        return self.list_items(session_id)

    def remove_from_list(self, session_id: str, product_id: str, name: str = "默认清单") -> dict:
        with self.db.connect() as db:
            row = db.execute("SELECT id FROM shopping_lists WHERE session_id=? AND name=?", (session_id, name)).fetchone()
            if row:
                db.execute("DELETE FROM shopping_list_items WHERE list_id=? AND product_id=?", (row[0], product_id))
        return self.list_items(session_id)

    def add_cart(self, session_id: str, product: dict) -> dict:
        owner = self._shopping_scope(session_id)
        with self.db.connect() as db:
            db.execute("INSERT INTO cart(session_id,product_id,product,quantity) VALUES (?,?,?,1) ON CONFLICT(session_id,product_id) DO UPDATE SET quantity=quantity+1", (owner, product.get("product_id", ""), Database.dumps(product)))
        return self.session_state(session_id)

    def remove_cart(self, session_id: str, product_id: str) -> dict:
        owner = self._shopping_scope(session_id)
        with self.db.connect() as db:
            db.execute("DELETE FROM cart WHERE session_id=? AND product_id=?", (owner, product_id))
        return self.session_state(session_id)

    def add_favorite(self, session_id: str, product: dict) -> dict:
        owner = self._shopping_scope(session_id)
        with self.db.connect() as db:
            db.execute("INSERT OR REPLACE INTO favorites(session_id,product_id,product,created_at) VALUES (?,?,?,?)", (owner, product.get("product_id", ""), Database.dumps(product), utc_now()))
        return self.session_state(session_id)

    def remove_favorite(self, session_id: str, product_id: str) -> dict:
        owner = self._shopping_scope(session_id)
        with self.db.connect() as db:
            db.execute("DELETE FROM favorites WHERE session_id=? AND product_id=?", (owner, product_id))
        return self.session_state(session_id)

    def compare(self, session_id: str, product_ids: list[str], sort_by: str | None = None, descending: bool = False) -> dict:
        state = self.session_state(session_id)
        all_products = {p.get("product_id"): p for m in state["messages"] for p in m.get("products", [])}
        items = [all_products[pid] for pid in product_ids if pid in all_products][:4]
        fields = ["title", "brand", "category", "price", "sold_count", "attributes"]
        if sort_by in fields:
            items.sort(key=lambda item: (item.get(sort_by) is None, str(item.get(sort_by, "")) if sort_by not in {"price", "sold_count"} else float(item.get(sort_by) or 0)), reverse=descending)
        normalized = [{field: item.get(field) if item.get(field) not in (None, "", {}) else "数据集未提供" for field in fields} | {"product_id": item.get("product_id", "")} for item in items]
        return {"products": normalized, "fields": fields, "sort_by": sort_by, "descending": descending, "note": "字段只来自商品数据；缺失字段显示为数据集未提供。"}

    def create_order(self, session_id: str) -> dict:
        state = self.session_state(session_id)
        items = state["cart"]
        if not items:
            raise ValueError("购物车为空，无法创建订单")
        oid = "SIM-" + uuid.uuid4().hex[:10].upper()
        now = utc_now()
        owner = self._shopping_scope(session_id)
        with self.db.connect() as db:
            db.execute("INSERT INTO orders(id,session_id,status,items,created_at,confirmed_at) VALUES (?,?,?,?,?,NULL)", (oid, owner, "DRAFT", Database.dumps(items), now))
            for item in items:
                db.execute("INSERT INTO order_items(order_id,product_id,product,quantity) VALUES (?,?,?,?)", (oid, item.get("product_id", ""), Database.dumps(item), item.get("quantity", 1)))
        return {"id": oid, "status": "DRAFT", "items": items, "created_at": now, "source": "cart", "message": "模拟订单，不产生真实付款或发货。"}

    def buy_now(self, session_id: str, product: dict) -> dict:
        """Create one confirmed simulated order directly from a product card."""
        if not product.get("product_id"):
            raise ValueError("商品缺少 product_id")
        oid = "SIM-" + uuid.uuid4().hex[:10].upper()
        now = utc_now()
        owner = self._shopping_scope(session_id)
        item = {**product, "quantity": 1}
        with self.db.connect() as db:
            db.execute("INSERT INTO orders(id,session_id,status,items,created_at,confirmed_at) VALUES (?,?,?,?,?,?)", (oid, owner, "CONFIRMED", Database.dumps([item]), now, now))
            db.execute("INSERT INTO order_items(order_id,product_id,product,quantity) VALUES (?,?,?,1)", (oid, str(product["product_id"]), Database.dumps(product)))
        return {"id": oid, "status": "CONFIRMED", "items": [item], "created_at": now, "confirmed_at": now, "source": "buy_now", "message": "已生成本地模拟订单，不产生真实付款或发货。"}

    def update_order(self, session_id: str, order_id: str, action: str) -> dict:
        allowed = {"confirm": "CONFIRMED", "cancel": "CANCELLED"}
        if action not in allowed:
            raise ValueError("不支持的订单操作")
        now = utc_now()
        owner = self._shopping_scope(session_id)
        with self.db.connect() as db:
            current = db.execute("SELECT status FROM orders WHERE id=? AND session_id=?", (order_id, owner)).fetchone()
            if not current:
                raise KeyError("订单不存在")
            if current[0] != "DRAFT":
                raise ValueError("只有 DRAFT 订单可以确认或取消")
            db.execute("UPDATE orders SET status=?,confirmed_at=? WHERE id=?", (allowed[action], now if action == "confirm" else None, order_id))
        return self.session_state(session_id)

    def rebuy(self, session_id: str, order_id: str) -> dict:
        owner = self._shopping_scope(session_id)
        with self.db.connect() as db:
            order = db.execute("SELECT id FROM orders WHERE id=? AND session_id=?", (order_id, owner)).fetchone()
            if not order:
                raise KeyError("订单不存在")
            rows = db.execute("SELECT product,quantity FROM order_items WHERE order_id=?", (order_id,)).fetchall()
            for row in rows:
                item = Database.loads(row[0], {})
                item["quantity"] = row[1]
                db.execute("INSERT INTO cart(session_id,product_id,product,quantity) VALUES (?,?,?,?) ON CONFLICT(session_id,product_id) DO UPDATE SET quantity=quantity+excluded.quantity", (owner, item.get("product_id", ""), Database.dumps(item), item.get("quantity", 1)))
        return self.session_state(session_id)

    def traces(self, session_id: str) -> list[dict]:
        with self.db.connect() as db:
            traces = [dict(row) for row in db.execute("SELECT run_id,tool,args,result_count,elapsed_ms,created_at FROM tool_traces WHERE session_id=? ORDER BY id DESC LIMIT 100", (session_id,)).fetchall()]
            metrics = {row["run_id"]: dict(row) for row in db.execute("SELECT * FROM run_metrics WHERE session_id=?", (session_id,)).fetchall()}
        for trace in traces:
            trace["metrics"] = metrics.get(trace["run_id"], {})
        return traces
