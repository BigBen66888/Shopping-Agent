"""ShoppingBench Agent 的 SQLite 持久化层。

所有写操作都在本地事务中完成；这里不连接真实支付、库存或商家系统。
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        self.ensure(db)
        return db

    @staticmethod
    def ensure(db: sqlite3.Connection) -> None:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS interactions (id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT, response TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, user_id TEXT NOT NULL DEFAULT 'demo', title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, products TEXT NOT NULL DEFAULT '[]', trace TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS shopping_state (session_id TEXT PRIMARY KEY, state TEXT NOT NULL DEFAULT '{}', updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS tool_traces (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, run_id TEXT NOT NULL, tool TEXT NOT NULL, args TEXT NOT NULL, result_count INTEGER NOT NULL DEFAULT 0, elapsed_ms REAL NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS preferences (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL DEFAULT '', user_id TEXT NOT NULL DEFAULT 'demo', kind TEXT NOT NULL, content TEXT NOT NULL, source TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'ACTIVE', created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS shopping_lists (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, name TEXT NOT NULL DEFAULT '默认清单', UNIQUE(session_id,name));
        CREATE TABLE IF NOT EXISTS shopping_list_items (list_id INTEGER NOT NULL, product_id TEXT NOT NULL, product TEXT NOT NULL, quantity INTEGER NOT NULL DEFAULT 1, PRIMARY KEY(list_id,product_id));
        CREATE TABLE IF NOT EXISTS favorites (session_id TEXT NOT NULL, product_id TEXT NOT NULL, product TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(session_id,product_id));
        CREATE TABLE IF NOT EXISTS cart (session_id TEXT NOT NULL, product_id TEXT NOT NULL, product TEXT NOT NULL, quantity INTEGER NOT NULL DEFAULT 1, PRIMARY KEY(session_id, product_id));
        CREATE TABLE IF NOT EXISTS orders (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, status TEXT NOT NULL, items TEXT NOT NULL, created_at TEXT NOT NULL, confirmed_at TEXT);
        CREATE TABLE IF NOT EXISTS order_items (order_id TEXT NOT NULL, product_id TEXT NOT NULL, product TEXT NOT NULL, quantity INTEGER NOT NULL DEFAULT 1, PRIMARY KEY(order_id,product_id));
        CREATE TABLE IF NOT EXISTS run_metrics (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, run_id TEXT NOT NULL, input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0, api_cost REAL NOT NULL DEFAULT 0, compression_count INTEGER NOT NULL DEFAULT 0, retrieval_ms REAL NOT NULL DEFAULT 0, rrf_ms REAL NOT NULL DEFAULT 0, rerank_ms REAL NOT NULL DEFAULT 0, errors INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
        -- 长期记忆：跨会话，按 user_id 归属；与 preferences 分开存放，
        -- 因为 preferences 需用户确认，而内存记忆由 Agent 自动抽取。
        CREATE TABLE IF NOT EXISTS memory (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL DEFAULT 'demo', kind TEXT NOT NULL DEFAULT 'fact', content TEXT NOT NULL, source_session TEXT NOT NULL DEFAULT '', confidence REAL NOT NULL DEFAULT 1.0, status TEXT NOT NULL DEFAULT 'ACTIVE', created_at TEXT NOT NULL, UNIQUE(user_id,kind,content));
        CREATE TABLE IF NOT EXISTS app_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        # 兼容上一轮最小版本已经生成的数据库。
        columns = {row[1] for row in db.execute("PRAGMA table_info(orders)").fetchall()}
        if "confirmed_at" not in columns:
            db.execute("ALTER TABLE orders ADD COLUMN confirmed_at TEXT")
        message_columns = {row[1] for row in db.execute("PRAGMA table_info(messages)").fetchall()}
        if "trace" not in message_columns:
            db.execute("ALTER TABLE messages ADD COLUMN trace TEXT NOT NULL DEFAULT '[]'")
        session_columns = {row[1] for row in db.execute("PRAGMA table_info(sessions)").fetchall()}
        if "user_id" not in session_columns:
            db.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT NOT NULL DEFAULT 'demo'")
        # Merge legacy preferences into the single long-term-memory store once.
        if not db.execute("SELECT 1 FROM app_meta WHERE key='preferences_to_memory_v1'").fetchone():
            db.execute("""
                INSERT OR IGNORE INTO memory(user_id,kind,content,source_session,status,created_at)
                SELECT user_id,kind,content,session_id,status,created_at FROM preferences
            """)
            db.execute("INSERT INTO app_meta(key,value) VALUES ('preferences_to_memory_v1',?)", (utc_now(),))
        # Backfill normalized rows once for databases created by the earlier JSON-only schema.
        for order in db.execute("SELECT id,items FROM orders WHERE id NOT IN (SELECT DISTINCT order_id FROM order_items)").fetchall():
            for item in Database.loads(order[1], []):
                db.execute(
                    "INSERT OR IGNORE INTO order_items(order_id,product_id,product,quantity) VALUES (?,?,?,?)",
                    (order[0], str(item.get("product_id", "")), Database.dumps(item), int(item.get("quantity", 1))),
                )
        # One-time migration: shopping data belongs to the single local user, not a chat session.
        marker = db.execute("SELECT value FROM app_meta WHERE key='single_user_scope_v1'").fetchone()
        if not marker:
            owner = "user:demo"
            cart_rows = db.execute("SELECT product_id,product,quantity FROM cart WHERE session_id<>?", (owner,)).fetchall()
            for row in cart_rows:
                db.execute(
                    "INSERT INTO cart(session_id,product_id,product,quantity) VALUES (?,?,?,?) "
                    "ON CONFLICT(session_id,product_id) DO UPDATE SET product=excluded.product,quantity=cart.quantity+excluded.quantity",
                    (owner, row[0], row[1], row[2]),
                )
            db.execute("DELETE FROM cart WHERE session_id<>?", (owner,))
            favorite_rows = db.execute("SELECT product_id,product,created_at FROM favorites WHERE session_id<>?", (owner,)).fetchall()
            for row in favorite_rows:
                db.execute("INSERT OR REPLACE INTO favorites(session_id,product_id,product,created_at) VALUES (?,?,?,?)", (owner, row[0], row[1], row[2]))
            db.execute("DELETE FROM favorites WHERE session_id<>?", (owner,))
            db.execute("UPDATE orders SET session_id=? WHERE session_id<>?", (owner, owner))
            db.execute("INSERT INTO app_meta(key,value) VALUES ('single_user_scope_v1',?)", (utc_now(),))

    @staticmethod
    def dumps(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def loads(value: str, fallback: Any) -> Any:
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return fallback
