from __future__ import annotations
import json
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from .service import ShoppingService
from .deepseek import DeepSeekClient
from .settings import settings

service = ShoppingService()

class RecommendRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=8)
    session_id: str | None = None

class CartRequest(BaseModel):
    session_id: str
    product: dict

class ProductActionRequest(BaseModel):
    session_id: str
    product: dict

class CompareRequest(BaseModel):
    session_id: str
    product_ids: list[str] = Field(min_length=1, max_length=4)
    sort_by: str | None = None
    descending: bool = False

class ModelDispatchRequest(BaseModel):
    session_id: str
    query: str = Field(min_length=1, max_length=2000)

@asynccontextmanager
async def lifespan(app: FastAPI):
    if service.processed.exists():
        service.prepare()
    yield

app = FastAPI(title="ShoppingBench Agent", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"status": "ok", "prepared": service.agent is not None, "processed": str(service.processed), "retrieval": "bge-m3+faiss-ivfpq+bm25+rrf+rerank", "reranker": settings.reranker_model, "model_configured": bool(settings.llm_api_key), "model": settings.llm_model}

@app.post("/api/model/preflight")
def model_preflight():
    try:
        return DeepSeekClient().preflight()
    except Exception as exc:
        raise HTTPException(502, {"error_type": type(exc).__name__, "status_code": getattr(exc, "status_code", None), "message": str(exc)[:500]}) from exc

@app.get("/api/architecture")
def architecture():
    return service.architecture()

@app.post("/api/agent/execute")
def execute_model_agent(request: ModelDispatchRequest):
    try:
        return service.model_dispatch(request.session_id, request.query)
    except Exception as exc:
        raise HTTPException(502, f"{type(exc).__name__}: {str(exc)[:500]}") from exc

@app.post("/api/prepare")
def prepare(limit: int | None = None):
    try:
        return service.prepare(limit)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc

@app.post("/api/recommend")
def recommend(request: RecommendRequest):
    try:
        if request.session_id:
            return service.recommend_for_session(request.session_id, request.query, request.top_k).to_dict()
        return service.recommend(request.query, request.top_k).to_dict()
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc

@app.post("/api/recommend/stream")
def recommend_stream(request: RecommendRequest):
    """Newline-delimited JSON: trace events, answer deltas, then one done event."""
    session_id = request.session_id or service.create_session()["id"]

    def events():
        try:
            for event in service.recommend_for_session_stream(
                session_id, request.query, request.top_k
            ):
                if event.get("type") == "done":
                    event.pop("recommendation", None)
                    event["session_id"] = session_id
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as exc:
            yield json.dumps(
                {"type": "error", "message": f"{type(exc).__name__}: {str(exc)[:300]}"},
                ensure_ascii=False,
            ) + "\n"

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/api/sessions")
def create_session():
    return service.create_session()

@app.get("/api/sessions")
def list_sessions():
    return {"sessions": service.list_sessions()}

@app.get("/api/sessions/{session_id}")
def session_state(session_id: str):
    try:
        return service.session_state(session_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

@app.post("/api/sessions/{session_id}/clear")
def clear_session(session_id: str):
    try:
        return service.clear_session(session_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    try:
        return service.delete_session(session_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

@app.post("/api/cart")
def add_cart(request: CartRequest):
    return service.add_cart(request.session_id, request.product)

@app.get("/api/cart/{session_id}")
def get_cart(session_id: str):
    return service.get_cart(session_id)

@app.delete("/api/cart/{session_id}/{product_id}")
def remove_cart(session_id: str, product_id: str):
    return service.remove_cart(session_id, product_id)

@app.get("/api/lists/{session_id}")
def list_items(session_id: str):
    return service.list_items(session_id)

@app.post("/api/lists")
def add_to_list(request: ProductActionRequest):
    return service.add_to_list(request.session_id, request.product)

@app.delete("/api/lists/{session_id}/{product_id}")
def remove_from_list(session_id: str, product_id: str):
    return service.remove_from_list(session_id, product_id)

@app.post("/api/buy-now")
def buy_now(request: ProductActionRequest):
    return service.buy_now(request.session_id, request.product)

@app.post("/api/orders/{session_id}")
def create_order(session_id: str):
    try:
        return service.create_order(session_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

@app.post("/api/favorites")
def add_favorite(request: ProductActionRequest):
    return service.add_favorite(request.session_id, request.product)

@app.get("/api/favorites/{session_id}")
def get_favorites(session_id: str):
    return service.get_favorites(session_id)

@app.delete("/api/favorites/{session_id}/{product_id}")
def remove_favorite(session_id: str, product_id: str):
    return service.remove_favorite(session_id, product_id)

@app.post("/api/compare")
def compare(request: CompareRequest):
    return service.compare(request.session_id, request.product_ids, request.sort_by, request.descending)

# ---- 长期记忆（Agent 自动记录，跨会话）----

@app.get("/api/memory/{session_id}")
def get_memory(session_id: str, limit: int | None = None):
    items = service.long_term_memory(session_id, limit or 1000)
    return {"items": items, "count": len(items)}

@app.delete("/api/memory/{session_id}/{memory_id}")
def delete_memory(session_id: str, memory_id: int):
    return service.forget(session_id, memory_id)

@app.delete("/api/memory/{session_id}")
def clear_memory(session_id: str):
    return service.clear_memory(session_id)

@app.get("/api/traces/{session_id}")
def traces(session_id: str):
    return {"traces": service.traces(session_id)}

@app.get("/api/traces/{session_id}/replay")
def replay_traces(session_id: str):
    return {"session_id": session_id, "events": list(reversed(service.traces(session_id)))}

@app.post("/api/orders/{session_id}/{order_id}/{action}")
def order_action_path(session_id: str, order_id: str, action: str):
    try:
        if action == "rebuy":
            return service.rebuy(session_id, order_id)
        return service.update_order(session_id, order_id, action)
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc

@app.get("/api/orders/{session_id}")
def get_orders(session_id: str):
    return service.get_orders(session_id)

@app.post("/api/orders/{session_id}/{order_id}/rebuy")
def rebuy(session_id: str, order_id: str):
    try:
        return service.rebuy(session_id, order_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

frontend = Path(__file__).resolve().parents[1] / "frontend"
if frontend.exists():
    app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
