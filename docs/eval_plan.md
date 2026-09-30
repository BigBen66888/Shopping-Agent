# Shopping Agent 评测方案（四套数据集）

> 目标：用四套互补数据集，分别评测**检索链路**、**多 Agent 协同**、**上下文压缩**、**长期记忆**。
> 状态：规划稿。本文档只描述方案与预期，不改动任何业务代码。
>
> 现有基线数据来源：`reports/retrieval_nl*.json`、`reports/token_report.json`、`reports/server.log`

---

## 0. 总览

| # | 数据集 | 条数 | 评测对象 | 数据来源 | 优先级 |
|---|---|---|---|---|---|
| A | 官方 Product 检索集 | 250 | 二路召回 + rerank | ShoppingBench 官方 `synthesize_product_test.jsonl` | P0 |
| B | 复杂问题多 Agent 集 | 20 | 主/子 Agent 委派、工具编排、并行 | 自建 | P1 |
| C | 多轮长上下文压缩集 | 20 | `ContextManager` 压缩保真度 | 自建 | P0 |
| D | 长期记忆集 | 20 | `MemoryExtractor` 抽取 + 跨会话召回 | 自建 | P0 |

**前置校验（做 A 之前必须先做）**：官方语料是 Lazada 菲律宾站（PHP 比索），本项目 `data/products.jsonl` 是规范化后的 2,746,368 条。**必须先验证两边 `product_id` 空间一致**，否则所有 gt 都对不上，A 集全部作废。

```bash
# 从官方 product 集抽 20 个 product_id，逐个到本地语料里查
# 命中率 = 100% 才能继续；若命中率低，需改用"标题匹配"方式重建 gt
```

---

## A. 官方 Product 检索集（250 条）

### A.1 为什么只用 product

官方 900 条 = product 250 + shop 250 + voucher 250 + web_simpleqa 150。
本项目**不具备**的能力：找同店多商品（无 `shop_id` 聚合）、优惠券凑单（无券模型）、联网百科推理（无 web 搜索工具）。因此只有 product 250 可用。

### A.2 数据集样例

**用法一：官方 `title` 原文当 query（测纯检索，对应现有 E 组）**

| 字段 | 内容 |
|---|---|
| query | `Arborea Hybrid ghost crash cymbals cymbal B20 cast bronze` |
| gt product_id | `591486855` |

**用法二：官方 `query` 原文当 query（测长约束 NL，对应现有 A/B/C/D 组）**

| 字段 | 内容 |
|---|---|
| query | `Find unisex black sneakers with a hook and loop closure, model shoes, available with LazFlash deals, and priced above 55 PHP.` |
| gt product_id | `4644530785` |
| 可解析约束 | color=black / is_unisex=unisex / closure=hook and loop / model=shoes / service=flashsale / price>55 |

**建议统一成如下 jsonl schema**（与现有 `eval/queries_nl.jsonl` 兼容，只多两个字段）：

```json
{
  "query": "Find unisex black sneakers with a hook and loop closure, model shoes, available with LazFlash deals, and priced above 55 PHP.",
  "product_id": "4644530785",
  "group": "G_official_nl",
  "constraints": {
    "color": "black",
    "service": "flashsale",
    "min_price": 55,
    "attributes": {"is_unisex": "unisex", "shoes_closuretype": "hook and loop", "model": "shoes"}
  }
}
```

### A.3 B 组三档对比配置（按你截图的口径）

| 组 | 配置 | 对比指标 | 验证什么 |
|---|---|---|---|
| **B1** | 仅向量召回 | Recall@5/10、Precision@K、MRR、NDCG、Search P95、最终 Task Success | 语义召回的基础能力 |
| **B2** | 向量 + BM25/关键词，RRF 合并，**不 rerank** | 同 B1，另加候选重复率、过滤准确率、召回耗时 | 第二路关键词是否补回品牌、型号和规格词 |
| **B3** | 向量 + BM25 → RRF → **rerank** | Recall、MRR、NDCG、最终 Task Success、Grounded Rate、Search P95、rerank 耗时 | 精排是否让正确候选更靠前并改善最终回答 |

> 对应代码：`semantic_vector.py`（B1）、`hybrid.py`+`rrf.py`（B2）、`reranker.py`（B3）。

### A.4 指标定义

| 指标 | 定义 | 现有实现 |
|---|---|---|
| Recall@K | 正确 product_id 是否落在前 K | ✅ `harness/metrics.py` `recall_at_k` |
| MRR | 1/首个正确答案的排名 | ✅ `harness/metrics.py` `reciprocal_rank` |
| NDCG@K | 折损累积增益 | ✅ `offline_embedding/eval_retrieval.py` `metrics()` |
| Precision@K | 前 K 中相关占比（需相关集合口径） | ⚠️ 缺，需补 |
| Search P95 | 检索阶段 P95 延迟 | ⚠️ 部分（`eval_retrieval.py` 有 avg_ms，无分位） |
| 候选重复率 | RRF 后两路重复候选占比 | ⚠️ 需新增（验证 B2 的"互补性"） |
| Grounded Rate | 最终回答引用的商品是否真在候选里 | ⚠️ 需新增 |

### A.5 预估结果（基于你现有真实基线外推）

**现有真实基线（来自 `reports/`）：**

| 数据集 | 条数 | 口径 | BM25 | 向量 | RRF | RRF+rerank |
|---|---|---|---|---|---|---|
| `retrieval_nl_identifiable` | 25 | 严格单答案 | R .52 / MRR .275 | R .44 / MRR .216 | **R .60 / MRR .308** | R .40 / MRR .223 |
| `retrieval_nl` | 20 | 严格单答案 | R .20 / MRR .127 | R .20 / MRR .110 | R .30 / MRR .115 | R .30 / **MRR .217** |
| `retrieval_nl_relevance` | 20 | 相关集合 | R .25 / MRR .145 | R .35 / MRR .175 | R .30 / MRR .158 | **R .40 / MRR .302** |

**关键观察（必须写进报告）**：在 identifiable 组上 **rerank 反而让 Recall 从 0.60 掉到 0.40**，但 MRR 在纯 NL 组上从 0.115 涨到 0.217。说明 rerank **提升排序质量、但对严格单答案的 Recall@8 有损**。这是 A 集最值得深挖的点。

**外推到 250 条的预估（含 95% CI）：**

| 组 | Recall@8 | MRR | NDCG@8 | Search P95 |
|---|---|---|---|---|
| B1 仅向量 | 0.38 – 0.48 | 0.18 – 0.25 | 0.24 – 0.31 | ~15–40 ms |
| B2 RRF 无 rerank | 0.48 – 0.60 | 0.25 – 0.34 | 0.32 – 0.42 | ~120–200 ms |
| B3 RRF + rerank | 0.35 – 0.47 | **0.30 – 0.42** | **0.35 – 0.46** | ~900–1600 ms |

> 依据：identifiable 组（含品牌型号，最接近官方 title 用法）的实测值 ± 因扩量到 250 条带来的方差收敛（CI 宽度约收窄 40%）。
> **重点结论预判**：B3 的 MRR/NDCG 优于 B2，但 Recall@8 可能更低 —— 需在报告中明确"精排是排序优化不是召回优化"。

**统计注意**：250 条的 95% CI 半宽约 ±0.06（p≈0.5 时最坏）。现有 20/25 条的 CI 半宽约 ±0.21，**功效严重不足**，这正是扩量的主要理由。建议沿用 `scripts/evaluate_abcd.py` 的 bootstrap（2000 次，seed=20260923）。

---

## B. 复杂问题多 Agent 集（20 条，自建）

### B.1 目标

测**真子 Agent 委派**（`delegate_search` / `delegate_shop`）、**并行派发**（`ThreadPoolExecutor`）、**权限隔离**（`ToolRegistry` deny-by-default）、**写入链路**（`add_to_cart` → `buy_now` → 订单）。

### B.2 数据集样例

```json
{
  "case_id": "B01",
  "query": "帮我找一款 2000 元以内的蓝牙耳机，然后加进购物车；另外顺便看看有没有便宜的鼠标，也加一个。",
  "expect": {
    "delegates": ["delegate_search", "delegate_shop"],
    "parallel_dispatch": true,
    "must_call": ["product_search", "add_to_cart"],
    "forbidden_tools": ["buy_now"],
    "final_cart_items_min": 2
  },
  "checks": ["权限隔离", "并行派发", "写入落库"]
}
```

建议覆盖的 6 类场景（20 条分配）：

| 类型 | 条数 | 测什么 |
|---|---|---|
| 跨 Agent 委派（读+写混合） | 4 | `delegate_search` + `delegate_shop` 是否都触发 |
| 并行派发 | 3 | 同一轮多个 delegate 是否走 `ThreadPoolExecutor`（看耗时是否 ≈ max 而非 sum） |
| 权限越界尝试 | 3 | 诱导 SearchAgent 写购物车 / ShopAgent 搜索 → 应被 `ToolRegistry` 拒绝 |
| 多约束 + 多轮写入 | 4 | 预算 + 品牌 + 品类，最后落订单 |
| 无结果 / 冲突需求 | 3 | 应优雅终止（`terminate`），不编造商品 |
| 纯记忆请求 | 3 | 应**不返回商品卡**（现有测试 `test_stream_api_and_frontend_show_only_relevant_cards` 的逻辑） |

### B.3 指标

| 指标 | 定义 |
|---|---|
| Task Success | 端到端是否达成（商品数达标 + 落库正确） |
| 委派准确率 | 该用子 Agent 时是否用了；不该用时是否没滥用 |
| 并行加速比 | `串行耗时 / 并行耗时`，>1.5 才算真并行 |
| 权限违规数 | 越界调用成功次数（**目标恒为 0**） |
| 工具调用步数 | 各 case 的步数分布 |
| 幻觉率 | 推荐了不在候选里的商品的比例 |
| 端到端 P50/P95 | 整轮延迟 |

### B.4 预估结果

| 指标 | 预估 | 说明 |
|---|---|---|
| Task Success | 0.60 – 0.75 | 多步链路，末端易断 |
| 委派准确率 | 0.70 – 0.85 | 提示词已明确规则 |
| 并行加速比 | 1.3 – 1.8 | 取决于 API 并发上限 |
| 权限违规数 | **0** | deny-by-default 已实现，预期零违规 |
| 幻觉率 | 0.05 – 0.15 | 取决于 `terminate` 触发情况 |
| 端到端 P50 | 12 – 25 s | 单轮 4037 token，多步叠加 |

---

## C. 多轮长上下文压缩集（20 条，自建）

### C.1 目标

现有配置：`CONTEXT_TOKEN_BUDGET=2200`、`CONTEXT_RECENT_LIMIT=8`、`MEMORY_MAX_INJECTED=12`。
压缩靠 `ContextManager.build()`（滑窗 → 摘要 → 截断），摘要由 `summarize()` 生成。**必须构造长对话才能触发压缩**，官方单轮集完全测不到。

### C.2 数据集样例

```json
{
  "case_id": "C01",
  "turns": [
    "我想买个机械键盘",
    "预算 500 左右",
    "要 87 键的",
    "轴体无所谓",
    "最好是无线",
    "要有 RGB",
    "这个太重了",
    "换个轻一点的",
    "要能 Mac 用的",
    "再便宜点"
  ],
  "probe_turn": "我之前说的预算是多少？",
  "expected_in_summary": ["预算 500", "87 键", "无线"],
  "assert": {
    "compression_triggered": true,
    "compacted_through_id_monotonic": true,
    "memory_leak": false
  }
}
```

建议覆盖的 5 类（20 条分配）：

| 类型 | 条数 | 测什么 |
|---|---|---|
| **压缩触发 + 保真度** | 6 | 构造 >2200 token，触发 `summarize()`，再追问老事实 → 测**信息保留率** |
| 指代消解 | 4 | 第 1 轮"机械键盘" → 第 3 轮"换个便宜的"（省略主语） |
| 跨轮约束遗忘 | 4 | 早期预算/品牌是否还在 `_slim_state` 里 |
| 压缩位置正确性 | 3 | 多轮后核对 `compacted_through_id` 是否跳变/重复压缩 |
| 超长对话（50+ 轮） | 3 | 测压缩是否会丢关键约束 + token 是否线性爆炸 |

### C.3 指标

| 指标 | 定义 | 目标 |
|---|---|---|
| **压缩保真度 (Fidelity)** | 追问老事实答对的条数 / 总追问数 | ≥ 0.80 |
| 指代消解准确率 | 省略主语时是否命中正确对象 | ≥ 0.80 |
| 约束保留率 | 早期约束在压缩后仍生效的比例 | ≥ 0.75 |
| 压缩触发正确性 | 该压缩时压缩、不该压缩时不压缩 | ≥ 0.95 |
| **Token 增长曲线** | 轮数 vs 输入 token（应次线性） | 斜率 << 1 |
| 压缩耗时 | `summarize()` 单次延迟 | < 3 s |

### C.4 预估结果

| 指标 | 预估 | 风险点 |
|---|---|---|
| 压缩保真度 | 0.65 – 0.80 | ⚠️ 见下方风险 |
| 指代消解准确率 | 0.75 – 0.90 | 滑窗 8 条内通常没问题 |
| 约束保留率 | 0.60 – 0.75 | 约束易在压缩中丢失 |
| 压缩触发正确性 | 0.90 – 1.00 | token 估算逻辑清晰 |
| 长对话 token 增长 | 次线性 | 压缩生效则为次线性 |

> ⚠️ **已知风险（必须先确认）**：`ContextManager.summarize()` 的实现方式是
> `";".join(previous, "最近对话：" + recent[-500:])` —— **纯字符串拼接最近 500 字符，没有 LLM 参与、没有信息蒸馏**。
> 若确实如此，压缩保真度会显著低于预期（预估降到 0.45–0.60），且"压缩"只是**截断**而非摘要。
> 这本身就是一个**高价值的评测发现**。建议 C 集第一件事就是验证这一点。

---

## D. 长期记忆集（20 条，自建）

### D.1 目标

测 `MemoryExtractor` 的抽取精度（稳定偏好 vs 一次性需求）、跨会话召回、冲突处理、注入上限截断。
记忆存在 SQLite `memory` 表（`UNIQUE(user_id, kind, content)` 去重 + `status='DELETED'` 软删除），**无向量检索**，读取按"品类字符串匹配 + id 倒序"排序。

### D.2 数据集样例

**D-a 抽取精度（一次性需求不应被记）**

```json
{
  "case_id": "D01",
  "user_message": "我要买一个黑色的背包",
  "expected_memories": [],
  "why": "一次性需求，不是稳定偏好"
}
```

```json
{
  "case_id": "D02",
  "user_message": "我一般只买黑色的东西",
  "expected_memories": [{"kind": "attribute", "content": "用户偏好黑色"}]
}
```

**D-b 跨会话召回**

```json
{
  "case_id": "D08",
  "session_1": ["我只用罗技的鼠标"],
  "session_2_new": "推荐个鼠标",
  "expected_injected": ["罗技"],
  "assert": {"memory_injected": true, "recommended_brand": "logitech"}
}
```

建议覆盖的 4 类（20 条分配）：

| 类型 | 条数 | 测什么 |
|---|---|---|
| 一次性需求 vs 稳定偏好（负例） | 6 | 应返回 `[]` 的不能记（**误记率**） |
| 稳定偏好（正例） | 5 | 品牌/预算/品类/属性/约束五类 kind 各覆盖 |
| 跨会话召回 | 5 | 新会话是否注入旧偏好并影响推荐 |
| 冲突与截断 | 4 | 黑白冲突取最新？超 12 条时挤掉谁？ |

### D.3 指标

| 指标 | 定义 | 目标 |
|---|---|---|
| 抽取 Precision | 记对的 / 总共记的 | ≥ 0.85 |
| 抽取 Recall | 该记的被记的 / 该记的总数 | ≥ 0.80 |
| **误记率 (False Positive)** | 一次性需求被记的比例 | **≤ 0.10** |
| 跨会话召回率 | 新会话成功注入并影响推荐的比例 | ≥ 0.80 |
| 冲突解析正确率 | 取最新/最相关的比例 | ≥ 0.75 |
| 去重有效率 | `UNIQUE` 约束拦截重复的比例 | ≥ 0.95 |

### D.4 预估结果

| 指标 | 预估 | 依据 |
|---|---|---|
| 抽取 Precision | 0.85 – 0.95 | `_EXTRACT_SYSTEM` 提示词质量高，明确区分 durable vs one-off |
| 抽取 Recall | 0.75 – 0.90 | 受 5 条/轮上限影响 |
| 误记率 | 0.05 – 0.15 | 已有测试覆盖该场景 |
| 跨会话召回率 | 0.70 – 0.90 | 排序用字符串匹配，可能漏 |
| 冲突解析正确率 | 0.65 – 0.80 | ⚠️ `id` 倒序 = 取最新，但**不感知语义冲突** |
| 去重有效率 | ~1.00 | DB 层 `UNIQUE` 保证 |

> 这是四套里**最可能表现最好的一套** —— 提示词与 DB 约束都做得扎实。

---

## 汇总：一份对照表

| 数据集 | 条数 | 核心指标 | 预估表现 | 主要风险 |
|---|---|---|---|---|
| A 官方检索 | 250 | Recall@8 / MRR / NDCG / P95 | RRF Recall 0.48–0.60；rerank MRR 0.30–0.42 | **rerank 损 Recall**；product_id 空间可能不一致 |
| B 多 Agent | 20 | Task Success / 权限违规数 / 并行加速比 | Success 0.60–0.75；违规 0 | 多步链路末端易断 |
| C 上下文压缩 | 20 | **压缩保真度** / 约束保留率 | 0.65–0.80（可能仅 0.45–0.60） | ⚠️ `summarize()` 疑似非真摘要 |
| D 长期记忆 | 20 | 抽取 P/R / 误记率 | P 0.85–0.95；误记 0.05–0.15 | 冲突解析不感知语义 |

### 建议执行顺序

1. **校验 `product_id` 空间一致性**（A 的前置，不做则 A 作废）
2. **A 集**：抽官方 product 250 条 → 跑 B1/B2/B3 三档 + bootstrap CI
3. **C 集**：先验证 `summarize()` 是否真摘要（单点验证，成本低、信息量大）
4. **D 集**：扩现有单条测试为批量集
5. **B 集**：依赖真实 API，放最后（成本最高）

### 待补的指标实现（`harness/metrics.py` 现仅 2 个函数）

```python
# 现有
recall_at_k(ranked_ids, expected_id, k=8) -> float
reciprocal_rank(ranked_ids, expected_id) -> float

# 建议新增
precision_at_k(ranked_ids, relevant_ids, k) -> float
ndcg_at_k(ranked_ids, relevant_ids, k) -> float
percentile(latencies, p) -> float          # P50/P95
```

### 另需注意的既有问题（本次只读发现）

| 问题 | 位置 | 影响 |
|---|---|---|
| `index_version` 字符串过时（`local-tfidf+bm25+rrf-v1`） | `harness/runner.py` | 报告元数据失真 |
| `reranker_version` 字符串过时（`independent-linear-reranker-v1`） | `harness/runner.py` | 报告元数据失真 |
| `trace_store.py` 全项目零引用 | `harness/trace_store.py` | 死代码 |
| `evaluate_harness.py` 仍调 `service.recommend` | `scripts/` | 该方法现已必需 API 模型，**脚本大概率跑不起来** |
| `evaluate_abcd.py` 绕开 `SearchAgent` 直连 `HybridRetriever` | `scripts/` | 消融**不含** min_price/brand 等新约束，口径与线上不一致 |
| rerank 候选硬编码 `[:20]` | `reranker.py` | 召回 80 只精排 20，中间 60 条白召回 |

---

## 附录：评测集文件命名建议

```
eval/
  official_product_250.jsonl      # A 集：官方 product 250 条（含 constraints 字段）
  multiagent_20.jsonl             # B 集：复杂问题多 Agent
  longctx_20.jsonl                # C 集：多轮长上下文（turns 数组）
  memory_20.jsonl                 # D 集：长期记忆（user_message + expected_memories）

reports/
  eval_A_retrieval.json           # B1/B2/B3 三档 + CI
  eval_B_multiagent.json
  eval_C_context.json
  eval_D_memory.json
```

---

## E. API 成本估算（deepseek-flash / DeepSeek-V4.1-Flash）

> 单价来源：https://api-docs.deepseek.com/zh-cn/quick_start/pricing/（2026-09-28 查）
> Token 用量来源：本项目 `data/shopping.sqlite3` 的 `run_metrics` 表**实测数据**（n=11 条有效记录）

### E.1 官方单价（元 / 百万 tokens）

| 计费项 | 空闲时段 | 高峰时段 |
|---|---|---|
| 输入（缓存命中） | 0.02 | 0.04 |
| 输入（缓存未命中） | **1.0** | **2.0** |
| 输出 | **4.0** | **8.0** |

**时段定义（北京时间）**：周一至周五 9:00–12:00、14:00–18:00 为**高峰**；其余时段含**周末全天**为**空闲**。
空闲价 = 高峰价 × 0.5。→ **实测跑评测建议安排在晚上或周末，直接省一半。**

模型信息：`deepseek-flash` = DeepSeek-V4.1-Flash，上下文 1M，输出上限 384K，支持 Tool Calls，并发限制 2500。

### E.2 实测 Token 用量（来自你的 `run_metrics` 表）

| 指标 | input_tokens | output_tokens |
|---|---|---|
| 样本数 | 11（已排除 1 条全 0） | 11 |
| 均值 | **8,487** | **873** |
| 中位数 | 6,673 | 741 |
| 最大值 | 17,548 | 1,486 |

> ⚠️ 注意：`run_metrics.api_cost` 字段**全是 0** —— 因为 `.env` 里 `LLM_INPUT_COST_PER_MILLION=0`、`LLM_OUTPUT_COST_PER_MILLION=0` 未填。
> 建议填入 `1.0` / `4.0`（空闲价），之后成本可自动落库统计。
> 另注：有 8 条 `output_tokens > 700`，超过 `.env` 的 `LLM_MAX_OUTPUT_TOKENS=700` —— 说明该限制未对全部调用生效（可能是流式或子 Agent 未受约束），值得排查。

### E.3 单轮成本（按缓存命中率）

| 缓存命中率 | 空闲时段 | 高峰时段 |
|---|---|---|
| 0% | 0.01198 元 | 0.02396 元 |
| 30% | 0.00948 元 | 0.01897 元 |
| **50%** | **0.00782 元** | **0.01564 元** |
| 70% | 0.00616 元 | 0.01232 元 |
| 90% | 0.00449 元 | 0.00899 元 |

> 缓存命中率取决于 system prompt + tool schema 是否稳定复用。你的 system prompt 496 token + tool schema 480 token 是固定前缀，**理论上命中率可以很高**，但每轮注入的上下文/记忆会变化，保守取 50%。

### E.4 四套数据集成本明细

**关键洞察：A 集的检索本身不花钱。** B1（仅向量）/ B2（RRF）/ B3（RRF+rerank）全部是**本地 CPU 计算**（FAISS + BM25 + CrossEncoder），**API 成本 = 0**。只有指标里的 `Task Success`、`Grounded Rate` 若要用 LLM 判分才产生费用。

**缓存命中率 50% 时（推荐基准）：**

| 数据集 | 轮数假设 | 空闲时段 | 高峰时段 |
|---|---|---|---|
| A 判分 250 条 | 250 次判分 | 0.195 元 | 0.389 元 |
| B 多 Agent | 20 × 6 轮 × 3（含子 Agent） = 360 | 2.816 元 | 5.631 元 |
| C 长上下文 | 20 × 10 轮 = 200 | 1.564 元 | 3.128 元 |
| D 长期记忆 | 20 × 4 轮 = 80 | 0.626 元 | 1.251 元 |
| **合计** | 890 次调用 | **约 5.2 元** | **约 10.4 元** |

**缓存命中率 0% 时（最坏情况）：** 总计约 8.0 元（空闲）/ 16.0 元（高峰）

### E.5 敏感性分析（空闲时段，缓存 50%）

| 情景 | 轮数倍数 | 总成本 |
|---|---|---|
| 乐观（轮数减半） | ×0.5 | 2.6 元 |
| **基准** | ×1.0 | **5.2 元** |
| 悲观（轮数翻倍） | ×2.0 | 10.4 元 |
| 极端（翻 3 倍） | ×3.0 | 15.6 元 |

### E.6 结论与省钱建议

**总预算 ≈ 5–15 元人民币**，即使极端情况也不到 20 元。**API 成本在这个项目里根本不是瓶颈**，真正的瓶颈是本地检索延迟（rerank ~900–1600 ms）和端到端耗时。

省钱建议（按收益排序）：

| 建议 | 预期节省 | 说明 |
|---|---|---|
| **改用空闲时段跑评测** | **50%** | 避开周一至周五 9–12 / 14–18 点 |
| 填好 `.env` 成本字段 | 0（仅是可见性） | `LLM_INPUT_COST_PER_MILLION=1.0`、`LLM_OUTPUT_COST_PER_MILLION=4.0` |
| 增大 system prompt 缓存复用 | ~20% | 减少每轮注入的变动内容 |
| 限制子 Agent 步数 | 视情况 | 现为 `max_steps=5`（子）/ `8`（主） |
| 排查 `max_tokens=700` 未生效 | 视情况 | 有 8 条输出超限，白花钱 |

> **一句话**：跑完整四套评测的 API 花费约等于**一杯奶茶**，可以放心跑；把精力放在 A 集的 product_id 校验和 C 集的 `summarize()` 验证上。
