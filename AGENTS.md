# AGENTS.md

本文件面向维护者与代码代理，描述 **当前仓库真实实现**、关键约束与改动时的联动检查点。

## 1. 项目定位

`stupidsimplerag` 是一个单机高性能 RAG 服务，核心策略是：

- 入库阶段重处理：LLM 提取摘要/表格叙述/关键词，构建富语义切片。
- 查询阶段轻逻辑：混合检索 + 可选 Rerank + 应用层过滤与时间衰减。

## 2. 实际运行架构

1. 接入层（`app/main.py`）
- FastAPI API + HTMX/Jinja2 控制台。
- `TTLCache(maxsize=1000, ttl=3600)` 做查询缓存。

2. 计算层
- `app/ingest.py`：文档分析、切分、metadata 注入。
- `app/core.py`：Embedding/Rerank/Qdrant 初始化与检索链路。

3. 存储层
- Qdrant 单集合（Dense: `text-dense`；Sparse: BM42）。
- 自动创建 payload index：`date_numeric`、`doc_hash`、`scope`、`filename`。

4. 模型层
- LLM/Embedding/Rerank 均走 OpenAI 兼容协议。
- 自托管 embedding 推荐通过独立 `llama.cpp` Vulkan 服务接入，不与主 API 镜像混装；Jina retrieval 场景可通过 `EMBEDDING_QUERY_PREFIX` / `EMBEDDING_DOCUMENT_PREFIX` 启用 query/document 前缀。
- Rerank 同时支持 `/rerank` endpoint 与 `/chat/completions` 风格。

## 3. 代码地图（按职责）

- `app/__init__.py`
  - 自动加载 `.env`。
  - 初始化日志桥接（`configure_logging`）。

- `app/main.py`
  - API 路由：`/health`、`/ingest`、`/ingest/batch`、`/ingest/text`、`/grounding/query`、`/chat`、`/chat/lod`、`/documents`。
  - UI 路由：`/`、`/ui/upload`、`/ui/upload/batch`、`/ui/chat`、`/ui/documents`。
  - 关键行为：缓存键构造、重试策略、成功入库后缓存失效、`scope` 归一化、按 `(filename, scope)` 删除重建、同文档 in-flight ingest 合并/串行化、HTMX 入口。

- `app/ingest.py`
  - `compute_doc_hash(content)`（SHA256）。
  - `analyze_document()`：LLM 返回 `LLMAnalysis`。
  - `process_file()`：先按语义 block 预处理，再按文档规模/区块类型自适应切片并写 metadata（含 grounding 所需文档摘要与区域标签）。

- `app/core.py`
  - Embedding 封装（OpenAI 原生 + fallback 到兼容 REST）。
  - `APIReranker`（双协议 + 重试 + 同步/异步）。
  - `get_query_engine()`：统一组装过滤器与后处理器链。
  - Qdrant 文档管理：`adoc_exists`、`delete_nodes_by_filename`、`list_all_documents`。
  - 独立 grounding 查询：单文档 chunks 拉取、候选项命中分类、可选 rerank 精排。

- `app/models.py`
  - `ChatRequest` / `ChatResponse` / `IngestResponse` / `DocumentInfo` / `TextIngestRequest` / `LLMAnalysis` / `Grounding*`。

- `app/preprocess.py`
  - 文档分区 helper：确定性切分 `title/body/qa/appendix_list/appendix_table`。

- `app/openai_utils.py`
  - API key/base 分派逻辑。
  - `OpenAICompatibleLLM` 支持多模型逗号轮询。
  - `build_llm()` / `GatewayRoutedLLM` 支持 `chat` / `ingest` purpose 分池、权重、least-inflight、熔断，以及可选的 scheduler（全局并发预算 / 预留槽位 / 自适应 pool limit）。

- `app/utils.py`
  - 文件名日期提取：时间戳、`YYYYMMDD`、`MMDD`、两位年份兜底。
  - 兼容 `NodeWithScore` metadata 访问。

- 运维脚本
  - `offline_ingest.py`：离线递归入库与进度跟踪。
  - `reset_qdrant.py`：危险重置集合。
- `preload_models.py`：预下载 FastEmbed 稀疏模型。
- `Dockerfile`：使用 BuildKit cache mount 缓存 `apt`、`pip` 与 FastEmbed 构建依赖，并可用本地 `model_cache` 加速模型预热。
- `docker-compose.llama-embedding.yml`：可选覆盖文件，增加独立 `llama.cpp` Embedding 服务、基于当前源码构建 `api` 镜像，并把 API 切到 1024 维自托管检索栈。
- `llm_router.example.json`：结构化 LLM router 示例，默认沿用现有 `LLM_API_BASE` / `LLM_API_KEY` 或 `OPENAI_*`，仅拆分模型池与路由策略。

## 4. API 合同（当前实现）

### 4.1 健康检查

1. `GET /health`
- 轻量存活检查，仅返回 `{"status": "ok"}`，供 Docker healthcheck / 负载均衡探测使用。

### 4.2 入库

1. `POST /ingest`
- `multipart/form-data`：`file` + 可选 `force_update`、`scope`。
- 支持 `.md` / `.txt`。
- 可选 header：`X-File-Mtime`（Unix 秒/毫秒 或 ISO）。

2. `POST /ingest/batch`
- `files` 多文件上传。
- 受 `BATCH_MAX_FILES` 与 `BATCH_INGEST_CONCURRENCY` 约束。

3. `POST /ingest/text`
- JSON：`content`、可选 `filename`、`force_update`、`scope`。
- 若无可解析 `X-File-Mtime`，默认使用应用时区当天日期（`APP_TIMEZONE`，回退 `TZ`，默认 `Asia/Shanghai`）。

入库响应模型统一为 `IngestResponse`：

```json
{
  "status": "ok|skipped|error",
  "chunks": 12,
  "filename": "example.md",
  "doc_hash": "...",
  "scope": "reports/2025",
  "error": null
}
```

### 4.3 查询

1. `POST /grounding/query`
- 独立于 `/chat` 的结构化 grounding API。
- 输入：单篇文档选择器（`doc_hash` 优先，或 `filename + scope`）+ 一组 `candidates[]`。
- 输出：文档摘要、逐候选项 `relevance_tier/source_zone/source_reason/excerpts/candidate_brief`。
- `skip_rerank=true` 时仍可完全工作。

1. `POST /chat`
- 过滤维度：日期范围、filename 精确、filename_contains 模糊、`keywords_any/all`、`scope`。
- `keywords_any/all` 匹配 `keywords/keyword_list`，也匹配 `scope`、`filename` 中的标签 token；`strong/moderate/weak` 兼容 `high/medium/low`。
- `skip_rerank`：跳过重排。
- `skip_generation`：只跑检索，回答由切片拼接兜底。

2. `POST /chat/lod`
- 两阶段：L1 选 Top 文档 -> L2 仅在这些文档中检索并生成。
- 过滤维度与 `/chat` 对齐：L2 会保留日期、filename、filename_contains、keywords 与 scope 过滤。
- 当前不接入缓存。

### 4.4 文档管理

- `GET /documents`：按 `(filename, scope)` 聚合并返回 chunk 数。
- `DELETE /documents/{filename}`：支持可选查询参数 `scope`；未传时仅删除无 scope 文档，并清缓存。

## 5. 检索与排序流水线

`get_query_engine()` 组装顺序：

1. Qdrant metadata filter
- `date_numeric` 范围过滤
- `filename` 精确匹配
- `filename_contains` 文本匹配（不区分大小写）
- `scope` 精确匹配
- `filenames_in`（LOD 二阶段）
- `scopes_in`（LOD 二阶段内部使用）

2. Postprocessor 链（按顺序）
- `ContentFilterPostprocessor`
- `KeywordFilterPostprocessor`
- `APIReranker`（可跳过）
- `TimeDecayPostprocessor`

## 6. Metadata 约束（切片入库）

`process_file()` 生成节点 metadata 字段：

- `filename`
- `date`（`YYYY-MM-DD`）
- `date_numeric`（`YYYYMMDD` 整数）
- `doc_hash`
- `doc_summary`
- `keywords`（逗号拼接）
- `keyword_list`（数组）
- `original_text`
- `section_type`
- `section_order`
- `block_index`
- `chunk_index`
- `heading_path`
- `is_list_zone`
- `is_qa_zone`
- `scope`（可选）

补充：`_node_to_source()` 会从节点 metadata 显式回填 `scope` 到 `SourceItem.scope`；
若节点无该字段则返回 `null`。
补充：节点 ID 会纳入 `scope`，避免相同内容跨 scope 写入时冲突。

## 7. 环境变量（代码中实际读取）

1. 模型与协议
- `LLM_MODEL`, `OPENAI_API_KEY`, `OPENAI_API_BASE`
- `CHAT_LLM_MODEL`, `INGEST_LLM_MODEL`
- `CHAT_LLM_API_BASE`, `CHAT_LLM_API_KEY`, `INGEST_LLM_API_BASE`, `INGEST_LLM_API_KEY`
- `LLM_ROUTER_CONFIG`, `LLM_ROUTER_CONFIG_FILE`, `LLM_ROUTER_STATS_INTERVAL`
- `EMBEDDING_MODEL`, `EMBEDDING_API_KEY`, `EMBEDDING_API_BASE`, `EMBEDDING_DIM`, `EMBEDDING_TIMEOUT`
- `EMBEDDING_QUERY_PREFIX`, `EMBEDDING_DOCUMENT_PREFIX`
- `RERANK_API_URL`, `RERANK_API_BASE`, `RERANK_API_KEY`, `RERANK_MODEL`, `RERANK_TIMEOUT`, `RERANK_RETURN_DOCUMENTS`

2. 检索策略
- `TOP_K_RETRIEVAL`, `TOP_N_RERANK`, `FINAL_TOP_K`, `SPARSE_TOP_K`, `TIME_DECAY_RATE`

3. 并发/重试/批处理
- `LLM_CONTEXT_WINDOW`, `LLM_CONCURRENCY`, `LLM_MAX_RETRIES`, `LLM_RETRY_BACKOFF`
- `EMBEDDING_MAX_RETRIES`, `EMBEDDING_RETRY_BACKOFF`
- `RERANK_MAX_RETRIES`, `RERANK_RETRY_BACKOFF`
- `INSERT_BATCH_SIZE`, `INGEST_INSERT_MAX_RETRIES`, `INGEST_INSERT_RETRY_BACKOFF`
- `QUERY_MAX_RETRIES`, `QUERY_RETRY_BACKOFF`
- `BATCH_INGEST_CONCURRENCY`, `BATCH_MAX_FILES`
- `offline_ingest.py` CLI 额外支持 `--delay-seconds`，用于低负载慢速回填

4. Qdrant
- `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_HTTPS`, `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_CLIENT_TIMEOUT`, `COLLECTION_NAME`

5. Docker / 端口 / 自托管 Embedding
- `API_PUBLIC_PORT`, `QDRANT_PUBLIC_PORT`, `EMBEDDING_PUBLIC_PORT`
- `GUNICORN_WORKERS`, `GUNICORN_THREADS`
- `EMBEDDING_GGUF_HOST_PATH`, `EMBEDDING_GGUF_FILE`, `EMBEDDING_CONTEXT_WINDOW`, `EMBEDDING_UBATCH`, `EMBEDDING_THREADS_BATCH`
- `SELF_HOSTED_EMBEDDING_MODEL`, `SELF_HOSTED_EMBEDDING_DIM`, `EMBEDDING_COLLECTION_NAME`
补充：当前 Vulkan 自托管 Embedding 默认推荐 `EMBEDDING_UBATCH=1024`、`EMBEDDING_THREADS_BATCH=8`。

6. Sparse 模型缓存
- `FASTEMBED_CACHE_PATH`, `FASTEMBED_SPARSE_MODEL`

7. 日志
- `APP_LOG_LEVEL`（优先）
- `LOG_LEVEL`

8. 时区
- `APP_TIMEZONE`（应用时区，默认 `Asia/Shanghai`）
- `TZ`（容器/系统时区，建议与 `APP_TIMEZONE` 一致）

## 8. 测试现状

- `test/test_features_scope.py`：mock 测试（scope 与 LOD 路由行为）。
- `test/test_embedding_service.py`：embedding 前缀与兼容客户端初始化测试。
- `test/test_grounding.py`：grounding 请求/响应与新 metadata 测试。
- `test/test_offline_ingest.py`：离线批量入库脚本的批次结果处理测试。
- `test/test_api.py`：真实链路测试（依赖外部模型服务与 Qdrant）。
- `test/conftest.py`：测试时集合名覆盖为 `pytest_integration_test`。

## 9. 文档维护约束

修改以下内容时必须同步更新 `README.md` + 本文件：

1. API 路由签名/字段变更。
2. metadata 字段增删。
3. 环境变量新增、默认值变更、重试策略变更。
4. Docker 启动方式、端口映射、镜像名变更。
5. 脚本 CLI 参数变更。

## 10. 典型改动检查清单

1. 改检索链路：确认 `/chat` 与 `/chat/lod` 行为一致性。
2. 改入库链路：确认 `force_update`、`doc_hash` 去重、缓存清理未回归。
3. 改过滤逻辑：补 `keywords_any/all`、`scope` 相关测试，并确认文档管理接口仍按 `(filename, scope)` 安全工作。
4. 改 grounding：确认 `/grounding/query` 仍独立于 `/chat`，并覆盖 `body_grounded / relation_grounded / list_only / not_found`。
5. 改配置项：补 `.env.example` 与 README 对应说明。
