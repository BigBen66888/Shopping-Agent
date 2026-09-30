# Kaggle 双 T4 离线构建 BGE-M3 索引

Kaggle 免费版提供 **2 × T4（各 16 GB）**，单次会话比 Colab 免费版更稳。这里在 GPU 上计算商品向量、训练 FAISS IVF-PQ，导出本地 CPU 可读取的文件。购物助手在本地只计算**查询**向量，不再编码商品或重建索引。稀疏召回仍用 BM25，二路 RRF 取 20 条，BGE-reranker-v2-m3 在本地 CPU 上精排取 8 条，因此 reranker 单独下载到本地。

## 1. 在本地制作 Kaggle 代码包

```powershell
.\.venv\Scripts\python.exe offline_embedding\make_kaggle_bundle.py
```

生成 `offline_embedding/kaggle_bundle.zip`，只包含源码，不含本地数据或模型。

## 2. 准备一个私有 Kaggle 数据集

Kaggle 不能挂载 Google Drive，用 **Add Data → Upload** 建一个私有数据集，建议包含：

> 挂载后的实际路径形如 `/kaggle/input/datasets/<owner>/<slug>/`，**深度不固定**（不同 Kaggle 版本也可能是 `/kaggle/input/<slug>/`）。因此所有查找都用 `rglob` 递归匹配文件名，不依赖固定层级。

| 文件 | 大小 | 是否必需 |
| --- | --- | --- |
| `kaggle_bundle.zip` | 很小 | 必需 |
| `documents.part1.jsonl.gz`、`documents.part2.jsonl.gz` | 约 1.5 GB | 可选，缺失则联网下载官方单文件 |
| `bge-m3/`（模型目录，含 `modules.json`/`config.json`） | 约 2.3 GB | 可选，缺失则联网下载 |

数据和模型放进数据集之后，每个会话都不用重新下载，断线重跑非常快。

上传 notebook [`Kaggle_BGE_M3_IVFPQ.ipynb`](Kaggle_BGE_M3_IVFPQ.ipynb)，在 **Settings** 里把 **Accelerator** 设为 **GPU T4 x2**、**Internet** 设为 **On**，然后按顺序执行。

## 3. 会下载什么、构建什么

- 原始数据默认取 [官方 ShoppingBench 仓库](https://github.com/yjwjy/ShoppingBench) 的 Git LFS 文件 `resources/documents.jsonl.gz`（约 1.47 GB），校验官方 SHA-256 后流式规范化成 `products.jsonl`。也接受两个本地拆分分片，共 2,746,368 行，有效商品数可能略低；脚本默认至少检查 240 万条。
- 从 [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) 下载商品与查询编码模型（约 2.3 GB）。Kaggle 不下载 reranker。
- T4 只负责生成 BGE-M3 的 1024 维归一化稠密向量，最大输入长度 128 token。FAISS 用 CPU 训练并写入 `IndexIVFPQ`，默认 `nlist=1024, M=64, nbits=8, nprobe=64`、内积距离。FAISS 文件本身即可在 Windows `faiss-cpu` 中读取，不需要 GPU 索引转换。约 275 万条的 PQ 编码约 176 MB，另有索引结构开销；不会保存约 11 GB 的原始 float32 向量矩阵。
- 每约 20 万条或 30 分钟把进度写到 `/kaggle/working/checkpoints`。会话断线后重新运行 notebook，会从已保存的 `index.ntotal` 继续。首次 IVF-PQ 训练样本 5 万条，训练结束后也会保存检查点。

## 4. 速度：fp16 是最大杠杆，双卡是第二个

T4（Turing）**没有 TF32**，FP32 只有约 **8.1 TFLOPS**，FP16 TensorCore 有约 **65 TFLOPS**。实测 1 万条：

| 配置 | 吞吐 | 全量 275 万条 |
| --- | --- | --- |
| fp32 / eager / bs32（改之前） | 52 条/s | 14.7 小时 |
| fp16 / eager / bs32 | 172 条/s | 4.4 小时 |
| **fp16 / sdpa / bs32** | **201 条/s** | **3.8 小时** |
| fp16 / sdpa / bs128 | 193 条/s | 3.9 小时 |
| fp16 / sdpa / bs256 | 188 条/s | 4.1 小时 |

注意：**加大 batch 反而略微变慢**，说明单卡已经跑满，瓶颈不在批量。因此真正的第二个杠杆是**第二张 T4**。

所以 `--batch-size` 的含义是**每张 GPU 的批量**（默认 32），不是总量。

### 为什么不用 `DataParallel`

`nn.DataParallel` 会在**每次 forward** 调用 `replicate()`，把约 1.1 GB 的 fp16 权重重新拷到第二张卡，每批约 100 ms 的隐性开销。`encoder.py` 的做法是**每张卡常驻一份模型**，用 Python 线程分别驱动——CUDA 算子会释放 GIL，两张卡真正并行，权重不用反复搬运。

提供三种后端，由 `--parallel` 选择：

| 后端 | 说明 |
| --- | --- |
| `thread`（默认，多卡时） | 每卡一份模型 + 多线程，无重复拷贝 |
| `mp` | sentence-transformers 自带的多进程池；若瓶颈在 Python/分词侧而非 GPU，这个更快 |
| `single` | 只用 cuda:0 |

notebook 的构建单元之前有一个基准单元，实测单卡与双卡后直接给出建议参数：

```bash
python offline_embedding/bench_gpu.py --work-dir /kaggle/working/shopping_work --sample 2000
```

按单卡 201 条/s 估算，双卡若接近线性加速（约 1.85 倍）则约 **2 小时**跑完全量，远低于 Kaggle 单次会话上限。

## 5. 精度：默认 fp16，但会自检

构建脚本先取 64 条真实文本，比较 fp16 与 fp32 向量的平均余弦相似度，**低于 0.999 直接中止**，不会静默降质。确有问题时改用 `--precision fp32` 回退。

由于索引最终写入的是 **IVF-PQ 的 8 比特量化码本**，其量化误差远大于 fp16 与 fp32 的嵌入差异，因此**本地查询侧保持 CPU fp32 编码即可，无需对齐精度**。

`precision` / `attn` / `batch_size` 都会写进 `manifest.json`，便于追溯。

## 6. 把产物放回本地

跑完点 **Save Version**，在 notebook 的 **Output** 标签下载：

```text
export/
├─ products.jsonl
└─ faiss_bge_m3/
   ├─ index.faiss
   └─ manifest.json
```

把 **`export` 内的内容**复制到本地 `Shopping_agent/data/`，保持目录结构；不要把 `export` 文件夹本身套进 `data/`，也不要混用旧的 50 万条 `products.jsonl` 与新的全量索引。

若本地已用**同一份原始分片**跑过 `prepare_full.py`，`products.jsonl` 会与 Kaggle 生成的完全一致，可以只下载 `faiss_bge_m3/`（约 176 MB），省掉约 3 GB 的下载。

完整本地服务还需要精排模型：

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py
.\.venv\Scripts\python.exe offline_embedding\verify_local.py
```

`scripts/download_models.py` 会把 BGE-M3 和 [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) 放到 `data/models/`；reranker 只给本地 20 条候选打分，不参与 embedding 建索引。验证脚本会逐行检查商品数量、顺序签名、FAISS 条数，并在 CPU 上跑一次查询编码和 rerank。

250 万条商品载入内存可能对 16 GB 机器造成压力，验证通过也不保证整个服务的峰值内存足够。

## 7. 本地运行全量 275 万语料的内存账

`data/products.jsonl` 全量 2,746,368 条，按实测（Windows / 16 GB 机器）：

| 组件 | 内存 | 说明 |
| --- | --- | --- |
| products 列表（含 attributes） | 6.0 GB | 2.46 KB/条 |
| products 列表（`LOAD_ATTRIBUTES=0`） | 4.4 GB | 1.61 KB/条，省 2.3 GB |
| **`ProductLineStore` 懒加载** | **0.02 GB** | 只记行字节偏移，按需解码 |
| BM25 倒排索引 | 约 1.2 GB | `postings: dict[str, array("I")]` |
| FAISS IVF-PQ 索引 | 0.2 GB | |
| BGE-M3（CPU fp32） | 约 1.5 GB | mmap，编码后逐步触达 |
| bge-reranker-v2-m3（CPU fp32） | 约 2.4 GB | |
| **合计（懒加载方案）** | **约 5.7 GB** | 可跑 |

⚠️ 模型是 **mmap 加载**的：刚加载完 RSS 很低（BGE-M3 仅 0.7 GB），但编码一次后会触达全部权重，
BGE-M3 涨到约 1.9 GB、加 reranker 后合计约 **4.27 GB**。按"加载瞬间 RSS"估算会严重低估。

### 关键：`ProductLineStore` 懒加载

`shopping_agent/product_store.py` 扫描 products.jsonl 记录每行**字节偏移**（275 万条约 22 MB），
检索时只解码排进候选的那几十条。配合 `BM25Index(materialize=False)`（不复制商品列表），
把全量评测的常驻内存从约 13.5 GB 压到约 5.7 GB。

BM25 的预算过滤用 `self.prices`（`array("d")` 缓存价格），因此**不需要为了过滤价格去解码商品行**——
否则带 budget 的查询会对每条 posting 解码一次，慢到不可用。

### int8 量化：实测后不推荐（默认关闭）

`MODEL_QUANTIZE=none` 是默认值。实测结论（torch 2.14 / CPU）：

- 量化 BGE-M3（仅 Linear）**成功**，查询向量与 fp32 **相似度 1.000000**
- 量化 **reranker 失败**：替换底层 `XLMRobertaForSequenceClassification` 会破坏 Hugging Face
  的 forward 签名映射，推理报 `AttributeError` / `KeyError: 'ne'`（`input_ids` 被当成 BatchEncoding）
- **稳态内存反而更差**：`quantize_dynamic` 会同时持有一份 fp32 拷贝，量化后 RSS 5.59 GB
  对比 fp32 的 1.87 GB
- **速度无收益**：17.2 vs 16.7 条/s

`nn.Embedding` 的动态量化还需要 `float_qparams_weight_only_qconfig`，用 dict 形式 qconfig
传给 `quantize_dynamic` 在 torch 2.x 上会产出 forward 损坏的模块。因此该开关保留仅为可复现实验。

### 测量工具

```powershell
# 抽样估算本地内存（默认只跑 10 万/25 万，不碰全量）
.\.venv\Scripts\python.exe offline_embedding\measure_local_memory.py
# 分进程测量量化收益与精度漂移
.\.venv\Scripts\python.exe offline_embedding\measure_quantization.py
```

## 8. 检索质量评测

`offline_embedding/eval_retrieval.py` 在**全量 275 万语料**上分阶段评测，输出
Recall@K / MRR / NDCG@K，并支持按查询分组：

```powershell
.\.venv\Scripts\python.exe offline_embedding\eval_retrieval.py --queries eval\queries_nl.jsonl
.\.venv\Scripts\python.exe offline_embedding\eval_retrieval.py --no-rerank   # 跳过 reranker，省内存
```

阶段划分：`BM25 only → vector only → RRF → +rerank`，能直接看出每个组件的贡献。

### ⚠️ 评测设计的坑：单一正确答案不适用于大型目录

`eval/queries_nl.jsonl` 是**人工改写的自然语言查询**（非原标题），但用"只认一个目标商品"的
严格指标时结果会严重偏低。原因不是检索坏了，而是 **275 万商品的目录里有大量近似重复品**：
"fluffy korean hair claw clip for girls" 的 Top-1 是"Korean woven hair clip ... for girls"，
语义完全正确，只是不是标注的那一个；"tin snips for cutting sheet metal" 的 Top-1 是
"Nicholson Tin Snip"、Top-3 是"Metal Sheet Cutter Aviation Tin Snip"，同样正确。

因此评测分两套查询：
- `eval/queries_nl.jsonl`：通用改写查询，用于**暴露问题**（含中英文跨语言、规格驱动、长尾变体四组）
- `eval/queries_nl_identifiable.jsonl`：**保留品牌/型号识别信息**的改写查询，用于衡量真实检索能力

若要在通用查询上得到有意义的绝对数字，需要把"唯一正确答案"换成**相关集合**
（例如同叶子品类 + 关键属性匹配），而不是单一 product_id。

## 9. 手动执行

不用 notebook 时，在 Kaggle 里依次执行：

```bash
python offline_embedding/download.py --work-dir /kaggle/working/shopping_work
python offline_embedding/prepare_full.py --work-dir /kaggle/working/shopping_work
python offline_embedding/bench_gpu.py --work-dir /kaggle/working/shopping_work --sample 2000
python offline_embedding/build_gpu.py --work-dir /kaggle/working/shopping_work \
    --checkpoint-dir /kaggle/working/checkpoints --parallel auto --precision fp16 --attn sdpa --batch-size 32
```

需要先上传代码包、挂好数据集并安装 `sentence-transformers`、`faiss-cpu`、`huggingface_hub`。

## 8. 常见问题

- **找不到 `kaggle_bundle.zip`**：确认私有数据集已通过 Add Data 挂载，且文件名没有被改。
- **显存不足**：把 `--batch-size` 降到 `16`；这是每卡的批量。
- **断点与当前数据不一致**：换一个 `--checkpoint-dir`，或删掉旧的检查点重跑。
- **下载失败**：脚本会直接显示失败的是原始 GitHub 文件还是模型。把两个 `.jsonl.gz` 和 `bge-m3/` 放进 Kaggle 数据集即可跳过联网。
- **手动下载模型时**必须保留 Hugging Face 仓库中的配置、分词器和权重文件，不要只复制权重文件。
