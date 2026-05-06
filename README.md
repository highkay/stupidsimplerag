# stupidsimplerag

专用、高效、节省资源的单机 RAG：重预处理，轻运行时。

入库阶段通过 LLM 把原始文档转成“富语义切片”（摘要 / 表格叙述 / 关键词），查询阶段使用 Qdrant 混合检索（Dense + Sparse）+ 可选 Rerank + 时间衰减，兼顾质量与成本。

## 架构与能力

- FastAPI 无状态服务，内置 1 小时 TTLCache 查询缓存。
- 入库 ETL：`chonkie` 切分 + LLM 三合一分析 + metadata 注入。
- 检索链路：Hybrid Retrieval -> 可选 Rerank -> 内容过滤 -> 关键词过滤 -> 时间衰减。
- 存储：单集合 Qdrant（Dense `text-dense` + Sparse BM42）。
- 前端：内置 HTMX + Jinja2 控制台（仪表盘、上传、查询、文档管理）。

## 项目结构

```text
/stupidsimplerag
├── app/
│   ├── __init__.py              # 自动加载 .env + 初始化日志
│   ├── main.py                  # FastAPI 路由（API + UI）
│   ├── core.py                  # Qdrant、检索引擎、Rerank、后处理器
│   ├── ingest.py                # 文档分析与切分
│   ├── models.py                # Pydantic 请求/响应模型
│   ├── openai_utils.py          # OpenAI 兼容配置与 LLM 包装
│   ├── logging_config.py        # Loguru/stdlib 日志桥接
│   ├── templates/               # Jinja2 页面与 HTMX partial
│   └── static/                  # CSS/Favicon
├── offline_ingest.py            # 离线批量入库脚本
├── reset_qdrant.py              # 危险：删除并重建集合
├── preload_models.py            # 预下载 FastEmbed 稀疏模型
├── test/                        # 集成与功能测试
├── Dockerfile
├── docker-compose.yml
└── AGENTS.md
```

## 快速启动

```bash
git clone <repo>
cd stupidsimplerag
cp .env.example .env
docker compose up -d
```

默认 `docker-compose.yml` 使用镜像 `highkay/stupidsimplerag:latest`：

- API: `http://localhost:8005`（容器内 8000）
- Qdrant: `http://localhost:6333`

使用自托管 `llama.cpp + Vulkan + F16 GGUF` Embedding（独立服务，不混入主 API 镜像）：

```bash
docker compose -f docker-compose.yml -f docker-compose.llama-embedding.yml up -d
```

启用该模式前，需先准备本地 GGUF 目录并设置：

- `EMBEDDING_GGUF_HOST_PATH`
- `EMBEDDING_GGUF_FILE`
- `EMBEDDING_COLLECTION_NAME`（建议新集合，如 `financial_reports_jina_v5_1024`）

该覆盖文件会基于当前源码构建 `highkay/stupidsimplerag:llama-embedding`，避免误用旧的 `latest` 运行时镜像，并使用 `ghcr.io/ggml-org/llama.cpp:server-vulkan` 将 embedding 计算卸载到 Vulkan 设备。

Vulkan 模式要求宿主机可用 `/dev/dri`，并且容器需加入 `video` group；覆盖文件已内置这些配置。

该覆盖文件会把 API 自动切到：

- `EMBEDDING_API_BASE=http://embedding:8080/v1`
- `EMBEDDING_MODEL=jina-embeddings-v5-text-small-retrieval`
- `EMBEDDING_DIM=1024`
- `EMBEDDING_QUERY_PREFIX=Query:`
- `EMBEDDING_DOCUMENT_PREFIX=Document:`

注意：从 `768` 维模型切到 `1024` 维时，必须迁移到新 Qdrant 集合后再回灌，不能直接复用旧 dense 向量集合。

本地开发（不走容器）：

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## HTTP API

### 健康检查

| Endpoint | Method | 说明 |
| --- | --- | --- |
| `/health` | `GET` | 轻量存活检查，供容器健康检查与负载均衡探测使用 |

### 入库

| Endpoint | Method | 说明 |
| --- | --- | --- |
| `/ingest` | `POST` | 单文件入库（`multipart/form-data`） |
| `/ingest/batch` | `POST` | 批量入库（`files` 多字段） |
| `/ingest/text` | `POST` | 直接上传文本（`application/json`） |

`/ingest` 与 `/ingest/batch`：

- 支持 `.md` / `.txt`
- `force_update`（可选）：`true` 时先删同名文档再重建
- `scope`（可选）：逻辑命名空间，作为文档身份的一部分写入 metadata
- 可选请求头 `X-File-Mtime`：支持 Unix 时间戳（秒/毫秒）或 ISO 时间，按 `APP_TIMEZONE`（或 `TZ`）归一为日期

`/ingest/text` 请求体：

```json
{
  "content": "# 文档正文",
  "filename": "optional.md",
  "force_update": false,
  "scope": "reports/2025"
}
```

### 查询与检索

| Endpoint | Method | 说明 |
| --- | --- | --- |
| `/grounding/query` | `POST` | 单文档、多候选项的结构化 grounding，返回可直接消费的证据包 |
| `/chat` | `POST` | 标准检索 + 可选生成 |
| `/chat/lod` | `POST` | 两阶段 LOD 检索（先选文档，再聚焦生成，过滤维度与 `/chat` 对齐） |

`/grounding/query` 请求体：

```json
{
  "document": {
    "doc_hash": "optional-doc-hash",
    "filename": "optional.md",
    "scope": "reports/2025"
  },
  "candidates": [
    {
      "identifier": "002475.SZ",
      "name": "立讯精密",
      "aliases": ["立讯"],
      "candidate_type": "stock"
    }
  ],
  "max_excerpts": 3,
  "skip_rerank": false
}
```

`/grounding/query` 说明：

- 与 `/chat` 独立，不返回自然语言问答，只返回结构化证据。
- 文档定位优先级：`doc_hash + scope` -> `doc_hash` -> `filename + scope`。
- 响应包含 `document` 与 `candidate_results[]`，后者含 `relevance_tier`、`source_zone`、`source_reason`、`excerpts`、`candidate_brief`。
- 当前支持 `body_grounded`、`relation_grounded`、`list_only`、`not_found` 四类相关性层级。

`/chat` 支持字段：

- `query`（必填）
- `start_date` / `end_date`
- `filename`（精确匹配）
- `filename_contains`（不区分大小写模糊匹配）
- `keywords_any` / `keywords_all`（匹配 `keywords/keyword_list`，也匹配 `scope`、`filename` 中的标签 token；`strong/moderate/weak` 兼容 `high/medium/low`）
- `scope`
- `skip_rerank`（跳过重排）
- `skip_generation`（仅返回切片，不调用生成）

`/chat` 与 `/chat/lod` 的响应 `sources[]` 会返回 `filename/date/score/keywords/text/scope`，
其中 `scope` 从切片 metadata 透传；无该字段时为 `null`。

示例：

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "英伟达最新财报表现",
    "start_date": "2025-01-01",
    "keywords_any": ["NVDA", "GPU"],
    "skip_generation": true
  }'
```

### 文档管理

| Endpoint | Method | 说明 |
| --- | --- | --- |
| `/documents` | `GET` | 列出文档（按 `filename + scope` 聚合） |
| `/documents/{filename}` | `DELETE` | 删除指定文档切片；可选查询参数 `scope`，未传时仅删除无 scope 文档 |

## Web 控制台

| 页面/接口 | 说明 |
| --- | --- |
| `/` | Dashboard（Qdrant 状态、缓存状态、配置快照） |
| `/ui/upload` | 单文件上传页面 |
| `/ui/upload/batch` | 批量上传页面 |
| `/ui/chat` | 查询页面 |
| `/ui/documents` | 文档列表 partial |
| `/ui/documents/{filename}` | 删除文档（HTMX） |

说明：UI 表单默认 `accept=.md`，但后端 API 实际支持 `.md` 和 `.txt`。

## 关键实现细节（与代码一致）

1. 去重与幂等
- `doc_hash = sha256(content)`。
- 若同一 `scope` 内已存在相同 `doc_hash` 且未 `force_update`，入库返回 `status=skipped`。
- 节点 ID 使用 `md5(doc_hash + scope + chunk_idx)`，避免跨 scope 冲突。
- `scope` 参与文档身份；同内容可在不同 scope 中并存。

1.1 grounding metadata
- `process_file()` 会额外写入 `doc_summary`、`section_type`、`section_order`、`block_index`、`chunk_index`、`heading_path`、`is_list_zone`、`is_qa_zone`。
- 这些字段服务于 `/grounding/query`，不会改变 `/chat` 与 `/chat/lod` 的请求/响应合同。

2. 检索后处理顺序
- `ContentFilterPostprocessor`
- `KeywordFilterPostprocessor`
- `APIReranker`（可跳过）
- `TimeDecayPostprocessor`

3. Rerank 协议兼容
- `RERANK_API_URL` 包含 `/rerank`：按 rerank endpoint 协议调用。
- 否则按 OpenAI Chat Completions 协议调用重排模型。

4. 缓存行为
- `/chat` 结果使用 TTLCache（1h）。
- 缓存键包含 query、日期范围、文件过滤、关键词过滤、skip 标志、scope。
- 任一入库接口成功写入新切片后会清空查询缓存。
- `/chat/lod` 当前不缓存。

5. 时区基线
- 应用统一使用 `APP_TIMEZONE`（回退 `TZ`，默认 `Asia/Shanghai`）。
- 时间戳解析、`/ingest/text` 默认日期、时间衰减均按同一时区计算。
- 依赖 `tzdata` 保证无系统 IANA 时区数据库环境下的行为一致性。

## 配置说明（.env）

### 核心配置

- `LLM_MODEL`, `OPENAI_API_KEY`, `OPENAI_API_BASE`
- `EMBEDDING_MODEL`, `EMBEDDING_API_KEY`, `EMBEDDING_API_BASE`, `EMBEDDING_DIM`
- `EMBEDDING_QUERY_PREFIX`, `EMBEDDING_DOCUMENT_PREFIX`（为空时保持旧行为；Jina retrieval 建议分别设为 `Query:` / `Document:`）
- `RERANK_API_URL`, `RERANK_API_KEY`, `RERANK_MODEL`
- `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_URL`, `QDRANT_API_KEY`, `COLLECTION_NAME`
- `API_PUBLIC_PORT`, `QDRANT_PUBLIC_PORT`
- `EMBEDDING_PUBLIC_PORT`, `EMBEDDING_GGUF_HOST_PATH`, `EMBEDDING_GGUF_FILE`
- `SELF_HOSTED_EMBEDDING_MODEL`, `SELF_HOSTED_EMBEDDING_DIM`, `EMBEDDING_COLLECTION_NAME`
- `APP_TIMEZONE`（默认 `Asia/Shanghai`）、`TZ`（容器系统时区，建议与 `APP_TIMEZONE` 一致）

### 检索与策略

- `TOP_K_RETRIEVAL`
- `TOP_N_RERANK`
- `FINAL_TOP_K`
- `SPARSE_TOP_K`
- `TIME_DECAY_RATE`

### 并发 / 重试 / 超时（高级）

- `LLM_CONTEXT_WINDOW`, `LLM_CONCURRENCY`, `LLM_MAX_RETRIES`, `LLM_RETRY_BACKOFF`
- `EMBEDDING_MAX_RETRIES`, `EMBEDDING_RETRY_BACKOFF`
- `RERANK_TIMEOUT`, `RERANK_MAX_RETRIES`, `RERANK_RETRY_BACKOFF`, `RERANK_RETURN_DOCUMENTS`
- `INSERT_BATCH_SIZE`, `INGEST_INSERT_MAX_RETRIES`, `INGEST_INSERT_RETRY_BACKOFF`
- `QUERY_MAX_RETRIES`, `QUERY_RETRY_BACKOFF`
- `BATCH_INGEST_CONCURRENCY`, `BATCH_MAX_FILES`
- `QDRANT_CLIENT_TIMEOUT`

### 日志

- `APP_LOG_LEVEL`（优先）
- `LOG_LEVEL`

## 运维脚本

1. `offline_ingest.py`
- 递归导入目录下 `.md/.txt` 或单文件。
- 支持 `--batch-size`、`--skip-completed`、`--progress-file`、`--max-retries`、`--dry-run`。
- 批量模式对超时采用指数退避重试。

2. `reset_qdrant.py`
- 删除并重建集合（危险操作）。
- 默认会交互确认；`-y` 跳过确认。
- 重建后会创建 `date_numeric`、`doc_hash`、`scope`、`filename` payload index。

3. `preload_models.py`
- 提前下载 FastEmbed 稀疏模型到 `FASTEMBED_CACHE_PATH`。

## 测试

```bash
pytest -q
```

当前测试构成：

- `test/test_features_scope.py`：Mock 测试（scope 透传、LOD 路由）。
- `test/test_features_scope.py` 额外覆盖 scope 去重、文档列表与 scoped 删除行为。
- `test/test_grounding.py`：独立 grounding 路由、分类结果与新 metadata 测试。
- `test/test_offline_ingest.py`：离线批量入库脚本的批次结果处理测试。
- `test/test_api.py`：真实链路测试（依赖可用的 LLM/Embedding/Qdrant）。

提示：`test/conftest.py` 会把 `COLLECTION_NAME` 改为 `pytest_integration_test`，避免污染默认集合。

## Docker 与 CI

- Dockerfile 使用 `python:3.13-slim`，启动命令为 Gunicorn + Uvicorn worker。
- `docker-compose.yml` 维持基础 API + Qdrant 栈；`docker-compose.llama-embedding.yml` 以覆盖文件方式增加独立 `llama.cpp` Vulkan Embedding 服务，避免把模型塞进主镜像。
- Dockerfile 构建层依赖 BuildKit cache mount 复用 `apt`、`pip` 与 FastEmbed 缓存；推荐本地构建时启用 `DOCKER_BUILDKIT=1`。
- Docker 镜像内置 `HEALTHCHECK`，探测 `GET /health`。
- 镜像内默认 `TZ=Asia/Shanghai`，并安装 `tzdata`；`docker-compose` 同步为 `api`/`qdrant` 注入 `TZ`。
- 镜像构建阶段会优先复制本地 `model_cache`，再默认执行 `preload_models.py` 校验/补齐 BM42；临时快速构建可传 `--build-arg PRELOAD_FASTEMBED=0` 跳过。
- GitHub Actions（`.github/workflows/docker-publish.yml`）在 `main` 相关文件变更时自动推送镜像：
  - `highkay/stupidsimplerag:latest`
  - `highkay/stupidsimplerag:<commit_sha>`

## 迁移到 1024 维 Embedding

`gemini-embedding-001` 的 `768` 维 dense 向量集合，不能直接切换到 `jina-embeddings-v5-text-small-retrieval` 的 `1024` 维集合继续使用。推荐蓝绿迁移：

1. 启动 `docker-compose.llama-embedding.yml`，使用新的 `EMBEDDING_COLLECTION_NAME`。
2. 用 `python reset_qdrant.py --collection <new_collection> --dim 1024 -y` 创建新集合。
3. 通过现有 `/ingest`、`/ingest/batch`、`/ingest/text` 或 `offline_ingest.py` 对原始文档全量回灌。
4. 验证 `/chat`、`/chat/lod`、`/grounding/query` 后，再把生产流量切到新集合。
5. 旧 `768` 维集合确认不再需要后再删除。

若接受停机，最简单路径是直接切换 `.env` 中的 `COLLECTION_NAME` / `EMBEDDING_DIM` 后执行 `python reset_qdrant.py -y`，然后重新回灌全部文档。

## 关联文档

- 维护者/代理协作规则与代码约束：`AGENTS.md`
