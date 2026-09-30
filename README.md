# Shopping Agent

基于 ShoppingBench 商品语料的本地网页购物智能体。项目使用 FastAPI 提供对话、商品检索、收藏、购物车、模拟订单和跨会话记忆；检索链路为 **BM25 + BGE-M3/FAISS + RRF + BGE reranker**，回答与任务编排使用 OpenAI 兼容模型接口（默认 DeepSeek）。

- 项目地址：<https://github.com/BigBen66888/Shopping-Agent>
- 详细评测：[eval_v3/results/REPORT.md](eval_v3/results/REPORT.md)
- 离线索引构建说明：[offline_embedding/README.md](offline_embedding/README.md)

## 演示

https://github.com/user-attachments/assets/01bdc7de-10bd-45b3-8c10-b66a2214f871

## 功能概览

- 多轮购物对话与流式回答
- BM25、BGE-M3 向量召回、RRF 融合和 reranker 精排
- 多 Agent 搜索/购物任务编排
- 购物车、收藏、对比、模拟下单与订单操作
- 上下文压缩、长期偏好记忆与冲突覆盖
- 本地 SQLite 持久化；不会连接真实支付系统

## 环境要求

- Python 3.10 或更高版本（推荐 3.11/3.12）
- 推荐至少 16 GB 内存、15 GB 可用磁盘空间
- 首次准备数据和模型需要网络；构建全量 FAISS 索引推荐使用 Kaggle 双 T4 GPU
- DeepSeek 或其他 OpenAI 兼容服务的 API Key

> 数据集、模型权重、FAISS/BM25 索引、数据库、日志和 `.env` 均被 `.gitignore` 排除，不应提交到 GitHub。

## 从零开始运行

### 1. 下载代码并创建环境

```powershell
git clone https://github.com/BigBen66888/Shopping-Agent.git
cd Shopping-Agent
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS/Linux 将 `\.venv\Scripts\python.exe` 换成 `./.venv/bin/python`。

### 2. 下载并预处理 ShoppingBench 数据集

```powershell
.\.venv\Scripts\python.exe scripts\download_dataset.py
.\.venv\Scripts\python.exe scripts\prepare_dataset.py
```

下载脚本从 [ShoppingBench 官方仓库](https://github.com/yjwjy/ShoppingBench) 获取约 1.47 GB 的 `documents.jsonl.gz`，校验 SHA-256 后生成 `data/products.jsonl`。全量语料约 274 万件商品，处理后文件约 3 GB；重复运行支持校验和断点续传。

### 3. 准备匹配的 FAISS 索引

应用必须同时拥有以下三个互相匹配的文件：

```text
data/
├─ products.jsonl
└─ faiss_bge_m3/
   ├─ index.faiss
   └─ manifest.json
```

索引必须由同一份 `products.jsonl` 构建，不能混用其他版本。推荐流程：

1. 运行 `python offline_embedding/make_kaggle_bundle.py` 生成仅含源码的上传包。
2. 在 Kaggle 打开 [offline_embedding/Kaggle_BGE_M3_IVFPQ.ipynb](offline_embedding/Kaggle_BGE_M3_IVFPQ.ipynb)，启用 GPU T4 x2 和网络后依次运行。
3. 下载 notebook 输出的 `export/products.jsonl` 与 `export/faiss_bge_m3/`，复制到本项目的 `data/`。

完整的断点续建、显存参数和目录说明见 [离线索引文档](offline_embedding/README.md)。全量 FAISS IVF-PQ 文件约 176 MB，仓库不提供或提交该文件。

### 4. 下载本地检索模型

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py
```

脚本从 Hugging Face 下载：

- [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3)：查询向量编码
- [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3)：候选精排

模型保存到 `data/models/`，不会上传 GitHub。若未手动下载，首次联网启动时程序也会尝试使用 Hugging Face 缓存或自动下载；显式下载更便于离线运行。

数据、索引和模型就绪后可做一致性验证：

```powershell
.\.venv\Scripts\python.exe offline_embedding\verify_local.py
```

### 5. 填写自己的 API Key

```powershell
Copy-Item .env.example .env
```

打开新生成的 `.env`，至少填写：

```dotenv
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_API_KEY=替换为你自己的密钥
DEEPSEEK_MODEL=deepseek-flash
```

不要把真实密钥写入源码、README 或 `.env.example`，也不要提交 `.env`。可运行以下命令只检查连通性（不会打印密钥）：

```powershell
.\.venv\Scripts\python.exe scripts\deepseek_preflight.py
```

### 6. 启动项目

Windows 可双击 `start-shopping-agent.cmd`，或执行：

```powershell
.\start_app.ps1
```

脚本会在 `8000`–`8010` 中选择空闲端口，并在准备完成后打开浏览器。第一次启动需要建立约 1.67 GB 的 BM25 缓存，可能等待数分钟；之后会直接加载缓存。关闭服务可双击 `stop-shopping-agent.cmd` 或执行 `./stop_app.ps1`。

也可以跨平台手动启动：

```powershell
.\.venv\Scripts\python.exe -m uvicorn shopping_agent.api:app --host 127.0.0.1 --port 8000
```

浏览器访问 <http://127.0.0.1:8000>，健康检查为 <http://127.0.0.1:8000/health>。

## 评测结果

评测日期为 2026-09-29。检索在约 274 万件商品上运行；相关性采用严格的唯一目标商品 ID。多 Agent、上下文与记忆结果属于小样本工程验证，不能外推为所有真实场景的成功率。完整数据口径、逐项结果和修复前对照见 [评测报告](eval_v3/results/REPORT.md)。

### 检索质量（250 条）

| 输入 | 方法 | R@1 | R@5 | R@8 | MRR@8 | NDCG@8 |
|---|---|---:|---:|---:|---:|---:|
| 完整标题 | BM25 | 65.6% | 86.8% | 90.8% | .748 | .787 |
| 完整标题 | 向量 | 71.6% | 90.0% | 91.6% | .791 | .822 |
| 完整标题 | 融合 | 71.2% | 91.2% | **94.4%** | .791 | .828 |
| 截短标题 | BM25 | 42.0% | 61.2% | 69.6% | .506 | .551 |
| 截短标题 | 向量 | 24.0% | 41.6% | 48.0% | .313 | .353 |
| 截短标题 | 融合 | 41.2% | 66.0% | 73.2% | .511 | .564 |
| 截短标题 | 融合 + 精排 | **53.2%** | **73.2%** | **76.8%** | **.606** | **.645** |
| 属性短需求 | BM25 | **44.8%** | **60.0%** | **64.8%** | **.510** | **.544** |

完整标题的两路候选池覆盖率为 99.6%，截短标题为 93.6%，属性短需求为 84.4%。精排能明显改善截短标题，但 CPU 精排 P95 约 24.8 秒，是当前主要交互延迟瓶颈。

### Agent、上下文与记忆

| 评测项 | 样本 | 结果 |
|---|---:|---:|
| 自然复杂购物任务完成 | 6 | 6/6 |
| 出现真实子 Agent 委派 | 6 | 5/6（83.3%） |
| 子任务并行加速 | 5 个并行批次 | 平均 1.67× |
| 加购商品与展示卡片一致 | 3 | 3/3 |
| 压缩历史最终首卡 ASR / CAR | 5 | 100% / 100% |
| 完整历史最终首卡 ASR / CAR | 5 | 100% / 100% |
| 跨会话偏好冲突覆盖 | 6 | 6/6 |
| 非用户要求的购买 | 6 个复杂任务 | 0 |

压缩组保留的历史文本 token 平均为完整历史的 85.3%。当前全量单元测试结果为 **46 passed**；隔离前端回归覆盖对话、商品卡、会话清空/刷新、历史删除和 404 行为并通过。

## 主要目录

```text
shopping_agent/     后端、Agent、检索与记忆实现
frontend/           无需构建的网页前端
scripts/            数据/模型下载、预处理和 API 预检
offline_embedding/  Kaggle GPU 索引构建工具
eval_v3/            当前评测脚本、数据与结果
tests/              单元测试
data/               本地数据、索引、模型和数据库（不上传）
```

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

购买和订单操作仅写入本地 SQLite，均为模拟行为，不会产生真实付款。

