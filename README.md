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

| Endpoint | 用途 | 请求格式 | 说明 |
| --- | --- | --- | --- |
| `POST /ingest` | 单文件入库 | `file=@xxx.md` (multipart) | 仅接受 Markdown，上传后自动切分 + 富语义头部注入。 |
| `POST /ingest/batch` | 批量入库 | 多个 `files=@*.md` 字段 | 返回 JSON 数组，结构与 `/ingest` 相同。 |
| `POST /chat` | 检索 + 生成 | `{"query":"…","start_date":"YYYY-MM-DD","end_date":"YYYY-MM-DD"}` | filter 字段可选，响应遵循 `ChatResponse`/`SourceItem`。 |

> 文件名、关键词、时间范围都会被映射到 Qdrant 元数据过滤；返回的 `sources` 会附带 `filename/date/score/keywords/text` 方便外部前端直接渲染。

### 示例

```bash
curl -X POST http://localhost:8000/ingest -F "file=@./docs/sample_report.md"

curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"query":"英伟达最新财报表现","start_date":"2025-01-01"}'
```

## 工作流速览

1. **预处理**：Chonkie 按 token 切分 → LLM 一次性生成 `summary/table_narrative/keywords` → 生成富语义 header 并拼接到每个 chunk。
2. **入库**：TextNode 携带 `filename/date/keywords/original_text` 写入 Qdrant 混合集合（Dense + FastEmbed BM42 Sparse），节点 ID 由 `filename + chunk_id` 哈希保证幂等。
3. **查询**：LlamaIndex QueryEngine 走 hybrid 检索（Top-K=100，可选时间过滤）→ API rerank Top-N=20 → `apply_time_decay` 线性衰减旧文档分数 → 保留 FINAL_TOP_K 命中。
4. **服务层**：FastAPI + TTLCache 以 `query+date_range` 为 key 做 1 小时语义缓存；Markdown-only 入库保证运行时简单可靠。

## 关键配置 (.env)

- **LLM / Embedding / Rerank**：`LLM_MODEL`, `OPENAI_API_KEY`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `RERANK_API_URL` 等；默认遵循 OpenAI 兼容协议，可指向 DeepSeek、SiliconFlow。
- **Qdrant**：`QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_URL`, `QDRANT_API_KEY`, `COLLECTION_NAME`, `QDRANT_HTTPS`。如使用托管集群，只需填 URL + API Key。
- **策略**：`TOP_K_RETRIEVAL`, `TOP_N_RERANK`, `FINAL_TOP_K`, `TIME_DECAY_RATE`, `SPARSE_TOP_K`。可按业务调优 recall / latency。
- **FastEmbed 缓存**：`FASTEMBED_CACHE_PATH`, `FASTEMBED_SPARSE_MODEL`；`preload_models.py` 可提前把 BM42 模型下载到镜像。

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

## 了解更多

- 架构细节、Prompt、数据模型请参考 `AGENTS.md` 与 `app/*.py` 源码，那里涵盖全量实现。
- 想提前热身稀疏模型，可运行 `python preload_models.py` 加速第一次部署。
