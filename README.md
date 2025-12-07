# stupidsimplerag

金融级单机 RAG，理念是“重预处理，轻运行时”：入库阶段用 LLM 生成摘要/表格解读/同义词，查询阶段依靠 Qdrant 混合检索 + API Rerank + 应用层时间衰减拿到稳定的 Finance Answer。

## 快速启动

```bash
git clone <repo>
cd stupidsimplerag
cp .env.example .env  # 补齐模型、Qdrant 配置
docker-compose up -d --build
```

容器会拉起 `qdrant` 和 `api`，FastAPI 默认监听 `localhost:8000`。

## HTTP API 一览

| Endpoint | Method | 载荷类型 | 用途 |
| --- | --- | --- | --- |
| `/ingest` | `POST` | `multipart/form-data` | 单文件入库，自动切分并生成富语义头部。 |
| `/ingest/batch` | `POST` | `multipart/form-data` | 批量入库，支持混合成功与部分失败的反馈。 |
| `/ingest/text` | `POST` | `application/json` | 直接推送 Markdown/TXT 字符串。 |
| `/chat` | `POST` | `application/json` | 检索 + 生成，支持多种过滤器。 |

每个入库请求都会返回 `IngestResponse`（或其数组），包含入库状态与 `doc_hash`，方便客户端判断幂等。检索响应遵循 `ChatResponse`/`SourceItem`，附带必要的元数据供前端展示。

### `POST /ingest`（单文件入库）

- **字段**：`file`（必填），值为 `.md`/`.txt` 文件；编码需为 UTF-8。
- **可选 Header**：`X-File-Mtime`，可使用 Unix 时间戳（秒/ms）或 ISO8601 字符串；若提供则作为该文档的业务日期写入元数据。
- **响应 (`IngestResponse`)**

```json
{
  "status": "ok | skipped",
  "chunks": 12,
  "filename": "20250228_NVIDIA_Report.md",
  "doc_hash": "b9c8e4...",
  "error": null
}
```

`status=skipped` 代表同一份文档（按 `doc_hash`）已存在，本次不会重复入库。

### `POST /ingest/batch`（批量入库）

- **字段**：重复添加 `files=@*.md` 或 `files=@*.txt`。
- **可选 Header**：每个文件都可以附带 `X-File-Mtime`。
- **响应**：`[IngestResponse, ...]`。出错的文件会返回 `status="error"` 与 `error` 字段，成功的文件与单文件版本一致。
- **并发**：受 `BATCH_INGEST_CONCURRENCY`（默认 4）控制，日志会记录每个文件的插入状态。

### `POST /ingest/text`（在线文本入库）

- **请求体**

```json
{
  "content": "# 文档正文…",          // 必填，Markdown 或纯文本
  "filename": "optional_name.md"    // 选填，未提供则自动生成
}
```

- **可选 Header**：`X-File-Mtime`，可使用 Unix 时间戳（秒/ms）或 ISO8601 字符串；若提供则作为该文档的业务日期写入元数据，否则使用当前 UTC 日期。
- **行为**：复用同一套去重/切分/插入逻辑。
- **响应**：`IngestResponse`。

### `POST /chat`（检索 + 生成）

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `query` | `string` | 是 | 用户问题，纯文本。 |
| `start_date` | `YYYY-MM-DD` | 否 | 起始日期（包含），用于 Qdrant 元数据过滤。 |
| `end_date` | `YYYY-MM-DD` | 否 | 结束日期（包含）。 |
| `filename` | `string` | 否 | 精确文件名过滤。 |
| `filename_contains` | `string` | 否 | 对文件名做大小写不敏感的包含匹配。 |
| `keywords_any` | `string[]` | 否 | 仅返回命中任意关键词的切片。 |
| `keywords_all` | `string[]` | 否 | 仅返回同时覆盖所有关键词的切片。 |

- **响应 (`ChatResponse`)**

```json
{
  "answer": "LLM 生成的回答",
  "sources": [
    {
      "filename": "20250228_NVIDIA_Report.md",
      "date": "2025-02-28",
      "score": 0.8431,
      "keywords": "NVDA,英伟达,GPU",
      "text": "原始文档切片前 200 字..."
    }
  ]
}
```

返回结果会先经过 Qdrant 混合检索 + API Rerank，再套用 `keywords_*` 过滤和 `apply_time_decay`；`FINAL_TOP_K` 控制最终 `sources` 数量，TTL 缓存以 `query + date/filename/keyword` 组合为键缓存 1 小时。

### 示例

```bash
# 单文件入库（可选 X-File-Mtime header）
curl -X POST http://localhost:8000/ingest \
  -H "X-File-Mtime: 1704067200" \
  -F "file=@./docs/sample_report.md"

# 文本入库（支持 X-File-Mtime header）
curl -X POST http://localhost:8000/ingest/text \
  -H "Content-Type: application/json" \
  -H "X-File-Mtime: 2025-01-01T00:00:00Z" \
  -d '{"filename":"20250301_NVDA.md","content":"# 财报\\n..."}'

# 检索查询
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"英伟达最新财报表现","start_date":"2025-01-01"}'
```

## 工作流速览

1. **预处理**：Chonkie 按 token 切分 → LLM 一次性生成 `summary/table_narrative/keywords` → 生成富语义 header 并拼接到每个 chunk。
2. **入库**：TextNode 携带 `filename/date/keywords/original_text/doc_hash` 写入 Qdrant 混合集合（Dense + FastEmbed BM42 Sparse），`doc_hash=sha256(content)` 用于跳过重复入库；节点 ID 由 `doc_hash + chunk_id` 哈希保证幂等/复用。
3. **查询**：LlamaIndex QueryEngine 走 hybrid 检索（Top-K=100，可选时间过滤）→ API rerank Top-N=20 → `apply_time_decay` 线性衰减旧文档分数 → 保留 FINAL_TOP_K 命中。
4. **服务层**：FastAPI + TTLCache 以 `query+date_range` 为 key 做 1 小时语义缓存；Markdown-only 入库保证运行时简单可靠。

## 关键配置 (.env)

- **LLM / Embedding / Rerank**：`LLM_MODEL`, `OPENAI_API_KEY`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `RERANK_API_URL`（填 OpenAI 兼容基础地址，走 `/v1/chat/completions`，如 `https://api.siliconflow.cn/v1` 或 HuggingFace OpenAI Proxy），`RERANK_MODEL`（推荐 `Qwen/Qwen3-Reranker-4B`）。默认遵循 OpenAI 兼容协议。
- **Qdrant**：`QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_URL`, `QDRANT_API_KEY`, `COLLECTION_NAME`, `QDRANT_HTTPS`。如使用托管集群，只需填 URL + API Key。
- **策略**：`TOP_K_RETRIEVAL`, `TOP_N_RERANK`, `FINAL_TOP_K`, `TIME_DECAY_RATE`, `SPARSE_TOP_K`。可按业务调优 recall / latency。
- **FastEmbed 缓存**：`FASTEMBED_CACHE_PATH`, `FASTEMBED_SPARSE_MODEL`；`preload_models.py` 可提前把 BM42 模型下载到镜像。
- **日志**：统一使用 `LOG_LEVEL` 控制（默认 INFO），无需额外的模块级配置。设置为 `debug` 可查看入库 LLM/Embedding 细节。

## 本地开发

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload  # 默认读取 .env
curl -X POST http://localhost:8000/ingest -F "file=@docs/sample.md"
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"query":"英伟达最新财报表现"}'
```

常见操作：

- Markdown 需在入库前完成转换（爬虫/ETL 阶段处理 PDF/TXT）。
- 若调整 `EMBEDDING_DIM`，请清空 `qdrant_data` 以重新建集合。
- 连接托管 Qdrant 时可用 `curl -X GET <QDRANT_URL>` + `api-key` 快速做健康检查。
- 离线批量导入：`python offline_ingest.py --dir ./docs --api-base http://127.0.0.1:8000 --batch-size 4`，递归读取 `.md/.txt` 并调用 `/ingest` 或 `/ingest/batch`，可用 `--dry-run` 预览。
- 重置 Qdrant 集合：`python reset_qdrant.py -y` 会按 `.env` 中的 `COLLECTION_NAME` 与 `EMBEDDING_DIM` 删除并重建集合（危险操作，务必确认目标环境）。

## 了解更多

- 架构细节、Prompt、数据模型请参考 `AGENTS.md` 与 `app/*.py` 源码，那里涵盖全量实现。
- 想提前热身稀疏模型，可运行 `python preload_models.py` 加速第一次部署。
