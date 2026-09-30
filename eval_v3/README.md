# Shopping Agent 评测 v3

最终结论见 [`results/REPORT.md`](results/REPORT.md)，业务源码修改见 [`CHANGES.md`](CHANGES.md)。

## 数据与脚本

- `datasets/product_250_local.jsonl`：官方 product 250 条对齐本地商品后生成的完整标题、标题片段、属性短需求。`prepare_product_queries.py` 可重复生成；纯检索不使用官方自然语言 query，也不调用模型 API。
- `retrieval_four_way.py`、`summarize_retrieval.py`：BM25、向量、融合、融合＋精排的逐条排名与汇总。
- `datasets/context_multiturn.jsonl`：自建真实逐轮会话；`check_context_feasibility.py` 验证标注条件在本地语料有满足商品；`context_end_to_end.py` 分别跑压缩和完整历史，`rescore_context.py` 对已保存候选按当前字段规则重算 ASR/CAR。
- `multiagent_full.py`、`rescore_multiagent.py`：真实主/子 Agent 工具编排、并行和购物动作评估。
- `memory_conflict_eval.py`：跨会话记忆冲突评估。

所有购物动作仅写入 `results/` 下的隔离 SQLite。`results/baseline/` 留存修改前证据；`results/focused/` 留存单问题修复探针；`results/*postfix*` 为最终 C 完整对照。
