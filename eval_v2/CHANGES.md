# 本轮源码修改记录

依据官方 250 条 product 集和 B/C/D 补测，仅修改了下列直接影响工程效果的代码。评测结果见 `results/REPORT.md`。

| 文件 | 原因与修改 | 验证 |
|---|---|---|
| `shopping_agent/rrf.py`、`shopping_agent/hybrid.py` | 两路等权 RRF 在官方自然语言 query 上低于 BM25 单路。新增可配置权重，线上 BM25:向量设为 1.5:1。 | 官方 250 条 RRF Recall@8 从 28.8% 到 32.8%；配对 bootstrap 差值 95% CI 为 +1.2 到 +7.2 个百分点。 |
| `shopping_agent/product_store.py` | 多个子 Agent 共享文件句柄时 `seek/readline` 竞态。改为每线程独立句柄。 | 修复前 16 线程随机读取 10,000 次有 467 次错读/解析失败；修复后并发回归测试 10,000 次全对，三条真实双委派任务均完成。 |
| `shopping_agent/deepseek.py` | 原来无法从真实双委派调用计算并行加速比。为工具调用增加耗时和批次耗时记录。 | 三条真实双委派任务平均批次加速比 1.80，且无权限越界写入。 |
| `shopping_agent/agents/main_agent.py` | 旧记忆与新记忆冲突时模型曾选旧颜色；补充最新优先的系统规则。 | 修复后冲突颜色问答连续 3 次选新颜色；三条跨会话推荐均把偏好带入搜索参数。 |
| `shopping_agent/agents/main_agent.py` | 多商品回答可能引用未展示的第 9 件以后候选；按答案中的真实商品 ID 把候选提到前 8 张卡。 | 新增卡片对齐回归测试；真实双委派用例中明确提到的商品 ID 均在卡片内。 |
| `shopping_agent/agents/main_agent.py` | 无结果后模型二次搜索漏传预算，超预算商品被展示。后续漏传预算时继承本轮已明确传入的预算，并明确禁止自行放宽预算。 | 无结果样例由 1/3 提升到 3/3；新增预算继承回归测试。 |
| `scripts/browser_session_regression.mjs` | 原 30 秒等待不足以覆盖 CPU 精排耗时；改为可通过环境变量设置，默认 120 秒。 | 隔离服务上的 Edge 浏览器真实提问、商品卡、记忆无卡、清空及删除历史均通过。 |
| `eval_v2/report.py` | 将已测的 B/C/D 指标和 A 组 NDCG、Precision、精排延迟完整写入报告，并纳入最终前端验证记录。 | 重新生成 `results/REPORT.md`，核对指标来自原始 JSON。 |

未更换向量模型、FAISS 索引或商品语料。`nprobe=64→256` 在官方集上没有召回收益；简单去停用词虽然加速 BM25，但 Recall@8 略降，所以没有用于线上。CrossEncoder 的 batch 4/8、线程 4/8 实测没有稳定提速，也未修改线上精排配置。

最终验证：`.venv` 下全量 `pytest` 为 23 passed；`SHOPPING_DB=eval_v2/results/frontend.sqlite3`、`LOAD_ATTRIBUTES=0` 的隔离服务 `/health` 返回 `prepared=true`；Edge 浏览器回归完整通过。首次在受限沙箱内运行 Edge 时调试连接断开，授权本机浏览器执行后成功，未发现前端业务故障。
