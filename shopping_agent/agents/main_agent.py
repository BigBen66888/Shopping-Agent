from __future__ import annotations

import json
import re
from threading import Lock
from typing import Any

from ..models import Product, Recommendation
from ..settings import settings
from ..tools.search_tools import product_summary
from .delegation_policy import delegation_triggers
from .search_agent import SearchAgent
from .shop_agent import ShopAgent

PRODUCT_SEARCH_MAX_CALLS = 5
PRODUCT_SEARCH_HYBRID_CALLS = 2

MAIN_SYSTEM_PROMPT = """你是购物 MainAgent，先判断买家意图和任务复杂度，再行动。
简单任务由你直接调用工具。仅在本轮同时存在以下至少一种独立任务时委派：(1) 两种不同商品分别检索；(2) 商品检索与已有购物状态的读取或操作相互独立；(3) 两种不同的已有购物状态任务。单项搜索，以及搜索后对该搜索结果加购或下单等前后依赖任务，都由你自己完成。符合条件时把本轮全部商品检索需求合并成一段完整自然语言，只调用一次 delegate_search；它会在隔离上下文中自行规划最多五次商品搜索。独立的购物状态操作调用 delegate_shop，并可与 delegate_search 同轮并行。不要按商品类别派发多个 SearchAgent。仅按用户明确要求执行加购或模拟下单。两个子 Agent 使用同一个 API 模型；SearchAgent 只能读；ShopAgent 可读写，其中读取必须通过 SearchAgent 提供的只读代理。你拥有两者全部工具。
调用工具时保留买家的全部约束（预算、价格上下限、品牌、颜色、尺寸、用途等）。商品语料为英文，请自己把搜索需求概括为英文关键词，金额用 budget 参数；不要用正则猜价格。每轮 product_search 硬上限为五次，通常一到两次足够；不要用近义词反复搜索同一品类。前两次使用 BGE + BM25 + rerank，第三至第五次自动切换为纯 BM25。若预算内无结果，直接说明，不要自行放宽预算再展示不符合条件的商品。读取订单、收藏、购物车和长期记忆时不要搜索商品；复购先读订单，再用真实订单 ID。绝不捏造商品或状态。只在用户明确要求购买或确认后调用 buy_now 或确认订单。
选择子 Agent 时给它完整的相关参数，但隔离无关会话；互不依赖的子任务在同一次工具调用中并行派发。长期记忆按时间从新到旧排列，索引 0 是最新；记忆冲突时用最新记录，本轮用户明确要求优先于记忆。
先用公开的一句话说明行动，再调用工具。最后只给简短中文总结：若有商品，说明最可能的一件满足买家哪些需求；找不到就如实说明。可自然询问是否下单或整理。公开进度与工具执行过程供下方展示；不要输出私有思维链。"""

SEARCH_PROMPT = """你是隔离运行的 SearchAgent，只接收主 Agent 的一段自然语言商品需求，并只返回最终搜索结论。你可以自行规划多轮 product_search，但整轮硬上限是五次：通常用一到两次，只有确实存在不同必要品类时才增加；每个品类最多搜索一次，不要用近义词、改写查询或放宽条件反复重试。前两次会走 BGE + BM25 + rerank，第三至第五次自动使用较快的纯 BM25。商品语料为英文，搜索词用英文并完整保留预算、品牌、颜色、尺寸和用途。最多选八件真实商品写入最终回答；仅在确有必要时查看少量候选详情。不要复述内部调用过程，不编造。"""
SHOP_PROMPT = "你是 ShopAgent，可使用购物写工具；需要读取状态时调用 SearchAgent 提供的 read_state 只读代理。只操作真实商品 ID 或订单 ID。若要求删除购物车第一个商品，先读取完整购物车，按返回的 items 顺序取 items[0] 的 product_id 删除；购物车为空时如实说明。若要求删除购物车最贵的商品，先读取完整购物车，按单价排序并核对 ID，再逐个删除。若买家明确要求给购物车下单，先 create_order，再用返回的订单 ID 调用 shop_action(update_order, order_action=confirm)；不能把 DRAFT 草稿当作已下单。完成后简短报告结果。"

def _tool(name: str, description: str, properties: dict, required: list[str] = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required or [], "additionalProperties": False}}}

READ_DEFS = [
    _tool("product_search", "以英文关键词搜索商品；请包含全部筛选要求。category 只填品类名词，如 keyboard 或 mouse；无线、机械等属性放入 query/requirements。", {"query": {"type": "string"}, "budget": {"type": "number"}, "min_price": {"type": "number"}, "brand": {"type": "string"}, "category": {"type": "string"}, "color": {"type": "string"}, "size": {"type": "string"}, "requirements": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1, "maximum": 8}}, ["query"]),
    _tool("get_product_details", "获取当前候选商品的详情。", {"product_id": {"type": "string"}, "include_attributes": {"type": "boolean"}}, ["product_id"]),
    _tool("read_state", "只读购物车、订单、收藏或长期记忆。", {"kind": {"type": "string", "enum": ["cart", "orders", "favorites", "memory"]}}, ["kind"]),
]
WRITE_DEFS = [
    _tool("add_to_cart", "把当前候选商品加入购物车。", {"product_id": {"type": "string"}}, ["product_id"]),
    _tool("buy_now", "购买当前候选商品，创建模拟订单。", {"product_id": {"type": "string"}}, ["product_id"]),
    _tool("shop_action", "其他购物写操作：收藏、移除、从购物车建单、更新订单或复购。", {
        "action": {"type": "string", "enum": ["add_favorite", "remove_favorite", "remove_from_cart", "create_order", "update_order", "rebuy"]},
        "product_id": {"type": "string"}, "order_id": {"type": "string"},
        "order_action": {"type": "string", "enum": ["confirm", "cancel"]}}, ["action"]),
]

DELEGATE_DEFS = [
    _tool("delegate_search", "把本轮全部商品需求作为一段自然语言交给一个隔离的 SearchAgent；每轮只能调用一次。", {"task": {"type": "string"}}, ["task"]),
    _tool("delegate_shop", "把独立购物操作作为自然语言交给隔离的 ShopAgent。", {"task": {"type": "string"}}, ["task"]),
]

def tool_result_message(name: str, payload: Any) -> str:
    if isinstance(payload, dict):
        payload = dict(payload)
        if name in {"delegate_search", "delegate_shop"}:
            # Child conversations, calls and timings stay outside MainAgent's
            # context. It receives only the child's final natural-language output.
            payload = {"answer": str(payload.get("answer") or "")}
        elif name in {"add_to_cart", "shop_action", "buy_now"} and "cart" in payload:
            payload = {"ok": True, "cart_count": len(payload.get("cart") or []),
                       "orders": [{"id": order.get("id"), "status": order.get("status")}
                                  for order in (payload.get("orders") or [])[:5]]}
        if isinstance(payload.get("products"), list):
            payload["products"] = [product_summary(p, with_description=True) for p in payload["products"][:8]]
        if isinstance(payload.get("items"), list):
            if name == "read_state" and "total_price" in payload:
                # The model needs every price and ID to remove the actual two
                # most expensive cart entries, even when the cart exceeds 8.
                payload["items"] = [
                    {key: item.get(key) for key in ("product_id", "title", "price", "quantity")}
                    for item in payload["items"]
                ]
            else:
                payload["items"] = payload["items"][:8]
    text = json.dumps(payload, ensure_ascii=False, default=str)
    if len(text) > settings.tool_result_max_chars:
        return json.dumps({"truncated": True, "preview": text[:settings.tool_result_max_chars - 100]}, ensure_ascii=False)
    return text


class MainAgent:
    SEARCH_TOOLS = frozenset(x["function"]["name"] for x in READ_DEFS)
    SHOP_TOOLS = frozenset(x["function"]["name"] for x in WRITE_DEFS)

    def __init__(self, search_agent: SearchAgent, shop_agent: ShopAgent):
        self.search_agent = search_agent
        self.shop_agent = shop_agent

    @staticmethod
    def _tool_plan(query: str) -> tuple[list[dict], str, bool]:
        reasons = delegation_triggers(query)
        if reasons:
            instruction = (
                "\n本轮已满足独立任务委派条件。首次行动必须调用 delegate_search "
                "或 delegate_shop；不同独立任务必须尽量在同一轮并行派发。"
                "把所有商品类别和约束合并到一个自然语言任务中，本轮只调用一次 delegate_search；"
                "若有独立的已有购物车操作，同一轮另调用一次 delegate_shop。"
                "若同时要求搜索商品、加购搜索结果和处理原有购物车，先并行委派商品搜索与原购物车操作；"
                "拿到真实商品 ID 后再委派加购，最后确认各项操作结果。"
                "本轮只提供委派入口，不要直接编造商品或购物状态。"
            )
            return DELEGATE_DEFS, MAIN_SYSTEM_PROMPT + instruction, True
        instruction = "\n本轮未满足独立任务委派条件。你直接使用检索、状态和购物工具完成请求。"
        return READ_DEFS + WRITE_DEFS, MAIN_SYSTEM_PROMPT + instruction, False

    def _runtime(self, session_id: str, client: Any, on_progress: Any = None):
        pool = {p["product_id"]: p for p in self.shop_agent.service._recent_products(session_id) if p.get("product_id")}
        displayed: list[Product] = []
        written_ids: list[str] = []
        search_groups: list[list[Product]] = []
        order_ids: set[str] = set()
        results_lock = Lock()
        search_gate = Lock()
        delegate_lock = Lock()
        product_search_calls = 0
        search_delegations = 0
        main_search_context: dict[str, Any] = {"budget": None, "products": [], "groups": []}

        def pick(product_id: str) -> dict:
            with results_lock:
                if product_id not in pool:
                    raise KeyError("商品不在当前候选中")
                return pool[product_id]

        def mark_written(product_id: str) -> dict:
            product = pick(product_id)
            with results_lock:
                if product_id not in written_ids:
                    written_ids.append(product_id)
            return product

        def make_product_search(search_context: dict[str, Any]):
            def product_search(query: str, budget: float | None = None, top_k: int = 8,
                               min_price: float | None = None, brand: str | None = None,
                               category: str | None = None, color: str | None = None,
                               size: str | None = None, requirements: str | None = None):
                nonlocal product_search_calls
                # A single gate both enforces the per-turn counter and serializes
                # CPU recall. A new _runtime is created for every user turn.
                with search_gate:
                    if product_search_calls >= PRODUCT_SEARCH_MAX_CALLS:
                        raise ValueError("本轮 product_search 已达到 5 次硬上限，请直接汇总已有结果")
                    product_search_calls += 1
                    call_number = product_search_calls
                    retrieval_mode = ("hybrid" if call_number <= PRODUCT_SEARCH_HYBRID_CALLS
                                      else "bm25")
                    if budget is None:
                        budget = search_context["budget"]
                    else:
                        search_context["budget"] = budget
                    constraints = {"min_price": min_price, "brand": brand, "category": category,
                                   "color": color, "size": size, "requirements": requirements}
                    rows, stats = self.search_agent.search(
                        query, min(max(top_k, 1), 8), budget,
                        retrieval_mode=retrieval_mode, **constraints,
                    )
                products = [p.to_dict() for p, _ in rows]
                with results_lock:
                    pool.update({p["product_id"]: p for p in products})
                    search_groups.append([p for p, _ in rows])
                    existing = {p.product_id for p in displayed}
                    displayed.extend(p for p, _ in rows if p.product_id not in existing)
                    search_context["groups"].append(products)
                    seen = {item["product_id"] for item in search_context["products"]}
                    for product in products:
                        if product["product_id"] not in seen:
                            search_context["products"].append(product)
                            seen.add(product["product_id"])
                return {"products": products, "count": len(products), "recall_count": stats.get("rrf_count", 0),
                        "retrieval_query": query, "budget": budget, "constraints": constraints,
                        "search_call": call_number,
                        "search_calls_remaining": PRODUCT_SEARCH_MAX_CALLS - call_number,
                        "pipeline": "BGE+BM25+rerank" if retrieval_mode == "hybrid" else "BM25"}
            return product_search

        def get_orders():
            result = self.search_agent.orders(session_id)
            order_ids.update(str(item["id"]) for item in result.get("items", []))
            return result

        def valid_order(order_id: str) -> str:
            if order_id not in order_ids:
                raise KeyError("请先读取订单，使用真实订单 ID")
            return order_id

        def create_order():
            result = self.shop_agent.create_order(session_id)
            if result.get("id"):
                order_ids.add(str(result["id"]))
            return result

        handlers = {
            "product_search": make_product_search(main_search_context),
            "get_product_details": lambda product_id, include_attributes=False: self.search_agent.details(pick(product_id), include_attributes),
            "get_cart": lambda: self.search_agent.cart(session_id),
            "get_orders": get_orders,
            "get_favorites": lambda: self.search_agent.favorites(session_id),
            "get_memory": lambda: self.search_agent.memory(session_id),
            "add_to_cart": lambda product_id: self.shop_agent.add_to_cart(session_id, mark_written(product_id)),
            "remove_from_cart": lambda product_id: self.shop_agent.remove_from_cart(session_id, product_id),
            "buy_now": lambda product_id: self.shop_agent.buy_now(session_id, mark_written(product_id)),
            "add_favorite": lambda product_id: self.shop_agent.favorite(session_id, mark_written(product_id)),
            "remove_favorite": lambda product_id: self.shop_agent.unfavorite(session_id, product_id),
            "create_order": create_order,
            "update_order": lambda order_id, action: self.shop_agent.update_order(session_id, valid_order(order_id), action),
            "rebuy": lambda order_id: self.shop_agent.rebuy(session_id, valid_order(order_id)),
        }

        def delegate(task: str, role: str):
            nonlocal search_delegations
            if role == "search":
                with delegate_lock:
                    if search_delegations >= 1:
                        raise ValueError("本轮只能派发一个 SearchAgent；请让已有 SearchAgent 汇总全部商品需求")
                    search_delegations += 1
            child_tools = READ_DEFS if role == "search" else WRITE_DEFS + [READ_DEFS[2]]
            child_names = {item["function"]["name"] for item in child_tools}
            child_handlers = {name: handler for name, handler in handlers.items() if name in child_names}
            search_context: dict[str, Any] = {"budget": None, "products": [], "groups": []}
            if role == "search":
                child_handlers["product_search"] = make_product_search(search_context)
            prompt = SEARCH_PROMPT if role == "search" else SHOP_PROMPT
            agent_name = "SearchAgent" if role == "search" else "ShopAgent"
            progress_callback = (lambda event: on_progress({"type": "child_progress",
                "agent": agent_name, "event": event})) if on_progress else None
            result = client.run_tool_loop(
                [{"role": "system", "content": prompt}, {"role": "user", "content": task}],
                child_tools, child_handlers, max_steps=8, serializer=tool_result_message,
                progress_callback=progress_callback,
            )
            balanced_products = []
            groups = search_context["groups"]
            for depth in range(max((len(group) for group in groups), default=0)):
                for group in groups:
                    if depth < len(group) and group[depth] not in balanced_products:
                        balanced_products.append(group[depth])
                    if len(balanced_products) == 8:
                        break
                if len(balanced_products) == 8:
                    break
            return {"agent": agent_name, "answer": result.get("answer", ""),
                    "calls": result.get("calls", []),
                    "model_timings_ms": result.get("model_timings_ms", []),
                    "products": balanced_products if role == "search" else []}

        def read_state(kind: str):
            names = {"cart": "get_cart", "orders": "get_orders",
                     "favorites": "get_favorites", "memory": "get_memory"}
            if kind not in names:
                raise ValueError("不支持的读取类型")
            return handlers[names[kind]]()

        def shop_action(action: str, product_id: str | None = None,
                        order_id: str | None = None, order_action: str | None = None):
            if action == "create_order":
                return handlers[action]()
            if action == "update_order":
                if not order_id or not order_action:
                    raise ValueError("订单 ID 和操作不能为空")
                return handlers[action](order_id, order_action)
            if action == "rebuy":
                if not order_id:
                    raise ValueError("订单 ID 不能为空")
                return handlers[action](order_id)
            if action in {"add_favorite", "remove_favorite", "remove_from_cart"}:
                if not product_id:
                    raise ValueError("商品 ID 不能为空")
                return handlers[action](product_id)
            raise ValueError("不支持的购物操作")

        handlers["read_state"] = read_state
        handlers["shop_action"] = shop_action
        handlers["delegate_search"] = lambda task: delegate(task, "search")
        handlers["delegate_shop"] = lambda task: delegate(task, "shop")
        return handlers, displayed, written_ids, search_groups

    def dispatch_model(self, client: Any, query: str, session_id: str) -> dict:
        handlers, _, _, _ = self._runtime(session_id, client)
        tools, system_prompt, require_first_tool = self._tool_plan(query)
        allowed = {item["function"]["name"] for item in tools}
        handlers = {name: handler for name, handler in handlers.items() if name in allowed}
        return client.run_tool_loop(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
            tools, handlers, max_steps=8, serializer=tool_result_message,
            require_first_tool=require_first_tool,
        )

    def recommend_stream(self, client: Any, query: str, session_id: str, context: dict | None = None,
                         routing_query: str | None = None):
        from queue import Queue
        progress_queue = Queue()
        handlers, displayed, written_ids, search_groups = self._runtime(
            session_id, client, on_progress=progress_queue.put)
        tools, system_prompt, require_first_tool = self._tool_plan(routing_query if routing_query is not None else query)
        allowed = {item["function"]["name"] for item in tools}
        handlers = {name: handler for name, handler in handlers.items() if name in allowed}
        context_text = ""
        if context:
            memories = context.get("relevant_preferences") or []
            recent = context.get("recent_messages") or []
            context_text = "\n长期记忆：" + json.dumps(memories[:8], ensure_ascii=False)
            context_text += "\n较早对话摘要：" + str(context.get("summary") or "")
            context_text += "\n最近对话：" + json.dumps(recent, ensure_ascii=False)
        stream = client.run_tool_loop_stream(
            [{"role": "system", "content": system_prompt + context_text},
             {"role": "user", "content": query}],
            tools, handlers, max_steps=8, serializer=tool_result_message,
            require_first_tool=require_first_tool,
            progress_queue=progress_queue,
        )
        trace = []
        delegation_timings = []
        while True:
            try:
                event = next(stream)
            except StopIteration as completed:
                final = completed.value
                answer = str(final.get("answer") or "").strip()
                if not answer:
                    raise RuntimeError("模型未返回总结")
                yield {"type": "delta", "content": answer}
                usage = final.get("usage", {})
                input_tokens = int(usage.get("prompt_tokens", 0) or 0)
                output_tokens = int(usage.get("completion_tokens", 0) or 0)
                # Multiple searches can produce more than eight candidates. If the
                # final answer names an ID outside the first eight, promote that
                # real candidate so the user sees its product card too.
                cited = re.findall(r"(?<!\d)\d{6,}(?!\d)", answer)
                by_id = {product.product_id: product for product in displayed}
                promoted = [by_id[pid] for pid in dict.fromkeys(written_ids + cited) if pid in by_id]
                promoted_ids = {product.product_id for product in promoted}
                # Give each independent search a chance to appear in the eight
                # cards. Otherwise the first search can fill every slot.
                balanced: list[Product] = []
                seen = set(promoted_ids)
                for depth in range(max((len(group) for group in search_groups), default=0)):
                    for group in search_groups:
                        if depth < len(group) and group[depth].product_id not in seen:
                            balanced.append(group[depth])
                            seen.add(group[depth].product_id)
                cards = (promoted + balanced + [product for product in displayed if product.product_id not in seen])[:8]
                return Recommendation(query=query, answer=answer, products=cards, thinking=[],
                    mode="llm-tools", trace=trace,
                    evidence={"tool_calls": final.get("calls", []),
                    "model_timings_ms": final.get("model_timings_ms", []),
                    "delegation_timings": delegation_timings,
                    "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
                    "api_cost": round((input_tokens * settings.llm_input_cost_per_million + output_tokens * settings.llm_output_cost_per_million) / 1_000_000, 8)})
            if event["type"] == "stage":
                phase = event.get("phase")
                step = event.get("step")
                content = (f"MainAgent · 正在规划第 {step} 步" if phase == "model_wait"
                           else f"MainAgent · 第 {step} 步规划完成（{event.get('elapsed_ms', 0) / 1000:.1f} 秒）")
                yield {"type": "progress", "agent": "MainAgent", "content": content}
                continue
            if event["type"] == "heartbeat":
                active = "、".join("SearchAgent" if x == "delegate_search" else
                                    "ShopAgent" if x == "delegate_shop" else x
                                    for x in event.get("active_tools", []))
                yield {"type": "progress", "agent": "MainAgent",
                       "content": f"{active} 正在执行（已等待 {event.get('elapsed_ms', 0) / 1000:.0f} 秒）"}
                continue
            if event["type"] == "tool_start":
                name = event["tool"]
                agent = "SearchAgent" if name == "delegate_search" else "ShopAgent" if name == "delegate_shop" else "MainAgent"
                yield {"type": "progress", "agent": agent,
                       "content": f"{agent} · 正在执行 {name}"}
                continue
            if event["type"] == "child_progress":
                child = event["event"]
                agent = event["agent"]
                if child["type"] == "stage":
                    phase = child.get("phase")
                    content = (f"{agent} · 正在规划第 {child.get('step')} 步" if phase == "model_wait"
                               else f"{agent} · 第 {child.get('step')} 步规划完成（{child.get('elapsed_ms', 0) / 1000:.1f} 秒）")
                    yield {"type": "progress", "agent": agent, "content": content}
                elif child["type"] == "tool_start":
                    yield {"type": "progress", "agent": agent,
                           "content": f"{agent} · 正在执行 {child['tool']}"}
                elif child["type"] == "tool":
                    result = child.get("result", {})
                    count = result.get("count") if isinstance(result, dict) else None
                    products = result.get("products", []) if isinstance(result, dict) else []
                    preview = ""
                    if child.get("tool") == "product_search" and products:
                        first = products[0]
                        preview = f"；例如 {str(first.get('title') or '')[:65]} ¥{first.get('price', 0)}"
                    content = (f"{agent} · {child['tool']} 已完成"
                               + (f"，找到 {count} 件" if count is not None else "")
                               + preview + f"（{child.get('elapsed_ms', 0) / 1000:.1f} 秒）")
                    trace.append({"role": "event", "content": content, "agent": agent})
                    yield {"type": "trace", "content": content, "agent": agent,
                           "tool": child.get("tool")}
                elif child["type"] == "heartbeat":
                    yield {"type": "progress", "agent": agent,
                           "content": f"{agent} · 仍在执行（{child.get('elapsed_ms', 0) / 1000:.0f} 秒）"}
                continue
            if event["type"] == "model":
                content = "MainAgent · " + event["content"]
                agent = "MainAgent"
            else:
                name = event["tool"]
                agent = "SearchAgent" if name == "delegate_search" else "ShopAgent" if name == "delegate_shop" else "MainAgent"
                args = json.dumps(event.get("arguments", {}), ensure_ascii=False)
                result = event.get("result", {})
                count = result.get("count") if isinstance(result, dict) else None
                if name in {"delegate_search", "delegate_shop"} and isinstance(result, dict):
                    delegation_timings.append({"agent": agent,
                        "elapsed_ms": event.get("elapsed_ms", 0),
                        "model_timings_ms": result.get("model_timings_ms", []),
                        "tool_timings": [{"tool": call.get("tool"), "elapsed_ms": call.get("elapsed_ms")}
                                         for call in result.get("calls", [])]})
                    summary = str(result.get("answer") or "").strip().replace("\n", " ")[:300]
                    content = f"{agent} · 已完成（{event.get('elapsed_ms', 0) / 1000:.1f} 秒）" + (f"：{summary}" if summary else "")
                else:
                    content = f"{agent} · {name}({args})" + (f" → {count} 条" if count is not None else " → 已完成")
            trace.append({"role": "event", "content": content, "agent": agent})
            yield {"type": "trace", "content": content, "agent": agent, "tool": event.get("tool")}


