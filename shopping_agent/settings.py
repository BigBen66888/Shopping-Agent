from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "model_cache"))


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    data_dir: Path = Path(os.getenv("SHOPPING_DATA_DIR", str(ROOT / "data")))
    database: Path = Path(os.getenv("SHOPPING_DB", str(ROOT / "data" / "shopping.sqlite3")))
    top_k: int = int(os.getenv("SHOPPING_TOP_K", "8"))
    llm_base_url: str = os.getenv("DEEPSEEK_BASE_URL", os.getenv("LLM_BASE_URL", "https://api.deepseek.com")).strip()
    llm_api_key: str = os.getenv("DEEPSEEK_API_KEY", os.getenv("LLM_API_KEY", "")).strip()
    llm_model: str = os.getenv("DEEPSEEK_MODEL", os.getenv("LLM_MODEL", "deepseek-flash")).strip()
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "45"))
    # Shopping queries normally do not need long hidden reasoning.  Keep it off by
    # default so a short request cannot unexpectedly consume thousands of tokens.
    deepseek_reasoning_effort: str = os.getenv("DEEPSEEK_REASONING_EFFORT", "low").strip()
    deepseek_thinking_enabled: bool = os.getenv("DEEPSEEK_THINKING_ENABLED", "0").lower() not in {"0", "false", "no"}
    llm_max_output_tokens: int = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "700"))
    llm_input_cost_per_million: float = float(os.getenv("LLM_INPUT_COST_PER_MILLION", "0"))
    llm_output_cost_per_million: float = float(os.getenv("LLM_OUTPUT_COST_PER_MILLION", "0"))
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3").strip()
    prompt_version: str = os.getenv("PROMPT_VERSION", "v1")
    embedding_mode: str = "faiss"
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3").strip()
    # 实测结论（2026-09-26，本机 torch 2.14 + CPU）：int8 动态量化不值得默认开启。
    #   - 量化 CrossEncoder 会破坏 HF 的 forward 签名映射，推理时报
    #     AttributeError/KeyError: 'ne'（input_ids 被当成 BatchEncoding）
    #   - 量化 embedding 输出与 fp32 完全一致（相似度 1.000000），但 quantize_dynamic
    #     会同时持有一份 fp32 拷贝，稳态 RSS 5.59 GB，反而比 fp32 的 1.87 GB 更差
    #   - 速度也没收益（17.2 vs 16.7 条/s）
    # 真正有效的是 LOAD_ATTRIBUTES=0 + ProductLineStore 懒加载。此项保留为实验开关。
    model_quantize: str = os.getenv("MODEL_QUANTIZE", "none").strip().lower()
    # 商品 attributes 字段每个约 0.85 KB，全量 275 万条约占 2.3 GB。
    # 只有 BM25 文本和 compare 的字段列表会用到；设为 0 可显著省内存，
    # 代价是 compare 里 attributes 显示为"数据集未提供"。
    load_attributes: bool = os.getenv("LOAD_ATTRIBUTES", "1").strip().lower() not in {"0", "false", "no"}

    # ---- 上下文管理（问题 2）----
    # 上下文预算按估算 token 计，而不是字符数：中文 1 字 ≈ 1 token，英文 4 字符 ≈ 1 token。
    context_token_budget: int = int(os.getenv("CONTEXT_TOKEN_BUDGET", "2200"))
    context_recent_limit: int = int(os.getenv("CONTEXT_RECENT_LIMIT", "8"))
    # 长期记忆：每轮结束后自动抽取稳定偏好并落库，不再要求用户说"记住"。
    memory_max_injected: int = int(os.getenv("MEMORY_MAX_INJECTED", "12"))
    memory_capture_max_tokens: int = int(os.getenv("MEMORY_CAPTURE_MAX_TOKENS", "320"))

    # ---- Token 成本控制（问题 5）----
    # 工具返回给模型前先裁剪：只保留模型真正需要的字段，避免把商品全字段回灌。
    tool_result_max_chars: int = int(os.getenv("TOOL_RESULT_MAX_CHARS", "6000"))
    # 按查询意图裁剪工具集：简单查询只暴露检索工具，省下每次约 800 token 的固定开销。


settings = Settings()
