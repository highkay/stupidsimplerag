# stupidsimplerag 是一个 **金融级、单机高性能 RAG 最终技术方案**。

本方案的核心理念是 **“重预处理，轻运行时”**：在入库阶段利用 LLM 将非结构化文档转化为“富语义文档”（含摘要、表格解读、同义词），从而在查询阶段利用低成本的混合检索和逻辑运算实现高精度召回。

-----

### 1\. 架构总览

  * **接入层 (FastAPI)**: 提供无状态 RESTful 接口，内置 LRU 语义缓存。
  * **计算层 (Python)**:
      * **ETL**: Chonkie 极速切分 + LLM 三合一预处理 (摘要/表格/同义词)。
      * **策略**: 应用层时间衰减算法 + 多策略文件名解析。
  * **存储层 (Qdrant)**: 单机硬盘索引 (On-Disk Indexing)，混合检索 (Dense + Sparse/BM42)。
  * **模型层 (API + Local)**:
      * LLM/Embedding/Rerank: 全部通过 OpenAI 兼容协议调用外部 API (如 DeepSeek, SiliconFlow)。
      * Sparse Embedding: 本地 FastEmbed (CPU, 轻量级)。

-----

### 2\. 项目结构

```text
/stupidsimplerag
├── .env                    # 核心配置文件
├── docker-compose.yml      # 容器编排
├── Dockerfile              # API 镜像构建
├── requirements.txt        # Python 依赖
└── app
    ├── __init__.py
    ├── main.py             # FastAPI 入口 & 缓存逻辑
    ├── core.py             # RAG 核心 (Qdrant, Rerank, Engine)
    ├── ingest.py           # 入库流水线 (LLM预处理, Chonkie切分)
    ├── utils.py            # 工具函数 (日期解析, 时间衰减)
    └── models.py           # Pydantic 数据模型
```

> 说明：项目的实际仓库名称为 `stupidsimplerag`，上面的结构即当前仓库根目录下的布局。

#### API 数据模型 (`app/models.py`)

`app/models.py` 集中定义 FastAPI 接口使用的 Pydantic 数据结构，常用模型包括：

- `ChatRequest`：字段为 `query` (必填)、`start_date`、`end_date`（YYYY-MM-DD，可选），与 `app/main.py` 中的入参保持一致。
- `SourceItem` / `ChatResponse`（或等价返回模型）：封装推理结果，包含 `answer` 以及若干 `sources`（每个包含 `filename`、`date`、`score`、`keywords`、`text`）。

示例请求与响应：

```json
POST /chat
Request:
{
  "query": "英伟达最新财报表现",
  "start_date": "2025-01-01"
}

Response:
{
  "answer": "...生成的自然语言回答...",
  "sources": [
    {
      "filename": "20250228_NVIDIA_Report.md",
      "date": "2025-02-28",
      "score": 0.8431,
      "keywords": "NVDA,英伟达,GPU",
      "text": "原始文档切片前200字..."
    }
  ]
}
```

-----

### 3\. 配置文件 (.env)

请根据你的 API 供应商填写。

```ini
# --- 模型服务配置 (OpenAI 兼容协议) ---
# 用于生成回答、摘要预处理
LLM_MODEL=deepseek-chat
OPENAI_API_KEY=sk-xxxxxx
OPENAI_API_BASE=https://api.deepseek.com/v1

# 用于 Dense Vector (768/1024/1536维均可)
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=sk-xxxxxx
EMBEDDING_API_BASE=https://api.deepseek.com/v1
EMBEDDING_DIM=1536
SPARSE_TOP_K=12

# 用于 Rerank (建议用 BAAI/bge-reranker-v2-m3)
RERANK_API_URL=https://api.siliconflow.cn/v1/rerank
RERANK_API_KEY=sk-xxxxxx
RERANK_MODEL=BAAI/bge-reranker-v2-m3

# --- Qdrant 数据库配置 ---
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_HTTPS=false
QDRANT_URL=
QDRANT_API_KEY=
COLLECTION_NAME=financial_reports

# --- 策略配置 ---
TOP_K_RETRIEVAL=100     # 初筛数量 (广撒网)
TOP_N_RERANK=20         # Rerank后保留数量 (为时间衰减留余地)
FINAL_TOP_K=10          # 最终返回给用户的数量
TIME_DECAY_RATE=0.005   # 时间衰减速率
```

> 若使用托管 Qdrant 集群，将 `QDRANT_URL=https://<cluster-host>:6333`、`QDRANT_API_KEY=<token>`。可通过以下命令验证连通性：
>
> ```bash
> curl -X GET 'https://<waiting-for-cluster-host>:6333' \
>   --header 'api-key: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.JxVRUBtR2jfv4XxYRvFYnBArICgRzhbPv_b3zScsANo'
> ```

-----

### 4\. 依赖列表 (requirements.txt)

```text
fastapi
uvicorn
python-multipart
# LlamaIndex 核心组件
llama-index-core
llama-index-llms-openai
llama-index-embeddings-openai
llama-index-vector-stores-qdrant
qdrant-client
# 极速切分
chonkie
# 混合检索 (本地轻量级)
fastembed
# 工具库
python-frontmatter
pydantic
httpx
python-dateutil
cachetools
```

-----

### 5\. 核心代码实现

#### `app/utils.py` (日期解析与时间衰减)

```python
import re
import os
import datetime
from dateutil import parser

def extract_date_from_filename(filename: str) -> str:
    """多策略提取文件名中的日期，返回 YYYY-MM-DD 或 None"""
    base_name = os.path.basename(filename)
    
    # 策略 1: Unix 时间戳 (10位或13位)
    # 匹配: 1727395200.md, report_1727395200.md
    ts_match = re.search(r'(^|[^0-9])(\d{10}|\d{13})([^0-9]|$)', base_name)
    if ts_match:
        try:
            ts = float(ts_match.group(2))
            if ts > 3000000000: ts /= 1000 # 处理毫秒级
            return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        except: pass

    # 策略 2: 标准日期格式 (YYYY-MM-DD, YYYYMMDD等)
    date_match = re.search(r'(\d{4})[-./_]?(\d{2})[-./_]?(\d{2})', base_name)
    if date_match:
        try:
            d_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
            return parser.parse(d_str).strftime("%Y-%m-%d")
        except: pass
            
    return None

def apply_time_decay(nodes, decay_rate=0.005):
    """
    对 Rerank 后的结果进行时间降权
    公式: NewScore = OldScore / (1 + Rate * DaysDiff)
    """
    today = datetime.date.today()
    for node in nodes:
        date_str = node.metadata.get("date")
        if not date_str: continue
        
        try:
            doc_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
            delta = (today - doc_date).days
            if delta > 0:
                # 线性平滑衰减，避免旧文档分数归零
                decay_factor = 1.0 / (1.0 + decay_rate * delta)
                if node.score:
                    node.score *= decay_factor
        except: pass
        
    # 重新排序
    nodes.sort(key=lambda x: (x.score or 0.0), reverse=True)
    return nodes
```

#### `app/ingest.py` (ETL 流水线)

核心逻辑：调用一次 LLM 完成摘要、表格转文字、同义词提取。

````python
import os
import json
import hashlib
from typing import List, Dict
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI
from chonkie import TokenChunker
from app.utils import extract_date_from_filename

# 专用 LLM 实例用于预处理
extractor_llm = OpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_API_BASE"),
    temperature=0.1
)

chunker = TokenChunker(chunk_size=512, chunk_overlap=50)

async def analyze_document(text: str) -> Dict:
    """LLM 三合一预处理：摘要、表格叙述化、关键词扩展"""
    context = text[:4000] # 截取头部避免超长
    prompt = f"""
    分析以下金融文档片段(Markdown)。请输出纯JSON格式，包含：
    1. summary: 50字内核心摘要。
    2. table_narrative: 将表格中的关键财务数据/增长率转化为自然语言陈述。无表格则留空。
    3. keywords: 提取5-8个关键实体(包含股票代码、公司别名、行业术语)。
    
    文档内容:
    {context}
    """
    try:
        response = await extractor_llm.acomplete(prompt)
        content = str(response).strip()
        # 清洗 Markdown 标记
        if content.startswith("```"): 
            content = content.split("\n", 1)[1].rsplit("\n", 1)[0]
        return json.loads(content)
    except:
        return {"summary": "", "table_narrative": "", "keywords": []}

async def process_file(filename: str, content: str) -> List[TextNode]:
    # 1. 提取时间
    meta_date = extract_date_from_filename(filename) or "1970-01-01"
    
    # 2. LLM 智能分析 (高价值步骤)
    analysis = await analyze_document(content)
    
    # 3. 构造注入头部 (Rich Context Header)
    # 这个 header 会被拼接到每个切片前，参与 Embedding
    keywords_str = ",".join(analysis.get('keywords', []))
    header_text = (
        f"Date: {meta_date}\n"
        f"Summary: {analysis.get('summary', '')}\n"
        f"Key Data: {analysis.get('table_narrative', '')}\n"
        f"Tags: {keywords_str}\n"
        f"---\n"
    )
    
    # 4. 极速切分
    chunks = chunker(content)
    nodes = []
    
    for i, chunk in enumerate(chunks):
        # 拼接 Header
        full_text = f"{header_text}{chunk.text}"
        
        node = TextNode(
            text=full_text,
            metadata={
                "filename": filename,
                "date": meta_date,
                "keywords": keywords_str,
                # 保留原始纯净文本供展示
                "original_text": chunk.text
            }
        )
        # 5. 幂等性 ID (防止重复入库)
        node.id_ = hashlib.md5(f"{filename}_{i}".encode()).hexdigest()
        nodes.append(node)
        
    return nodes
````

#### `app/core.py` (引擎构建)

集成自定义 API Rerank 和 Qdrant 混合检索。

```python
import os
import httpx
from typing import List, Optional
from llama_index.core import VectorStoreIndex, StorageContext, Settings, QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from qdrant_client import QdrantClient

# 全局配置
Settings.llm = OpenAI(model=os.getenv("LLM_MODEL"), temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model=os.getenv("EMBEDDING_MODEL"))

# --- 自定义通用 Rerank 类 ---
class APIReranker(BaseNodePostprocessor):
    def __init__(self, top_n: int = 10):
        super().__init__()
        self.top_n = top_n
        self.api_url = os.getenv("RERANK_API_URL")
        self.api_key = os.getenv("RERANK_API_KEY")
        self.model = os.getenv("RERANK_MODEL")

    def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None) -> List[NodeWithScore]:
        if not nodes: return []
        try:
            payload = {
                "model": self.model,
                "query": query_bundle.query_str,
                "documents": [n.node.get_content() for n in nodes],
                "top_n": self.top_n,
                "return_documents": False
            }
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(self.api_url, json=payload, headers=headers)
                resp.raise_for_status()
                # 适配常见 API 返回格式 (results 或 data)
                data = resp.json()
                results = data.get("results", data.get("data", []))
                
                new_nodes = []
                for res in results:
                    idx = res.get("index")
                    score = res.get("relevance_score", res.get("score"))
                    if idx is not None:
                        node = nodes[idx]
                        node.score = float(score) # 更新为 Rerank 分数
                        new_nodes.append(node)
                
                new_nodes.sort(key=lambda x: x.score, reverse=True)
                return new_nodes
        except Exception as e:
            print(f"Rerank Failed: {e}, returning original")
            return nodes[:self.top_n]

# --- Qdrant 初始化 (混合检索) ---
client = QdrantClient(host=os.getenv("QDRANT_HOST"), port=int(os.getenv("QDRANT_PORT")))
vector_store = QdrantVectorStore(
    client=client, 
    collection_name=os.getenv("COLLECTION_NAME"),
    enable_hybrid=True, # 开启 Dense + Sparse
    # 使用支持中文的多语言模型 BM42
    fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

def insert_nodes(nodes):
    VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context).insert_nodes(nodes)

def get_query_engine(start_date=None, end_date=None):
    # 1. 构造时间过滤器
    filters_list = []
    if start_date: filters_list.append(MetadataFilter(key="date", value=start_date, operator=FilterOperator.GTE))
    if end_date: filters_list.append(MetadataFilter(key="date", value=end_date, operator=FilterOperator.LTE))
    filters = MetadataFilters(filters=filters_list) if filters_list else None

    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    
    # 2. 返回引擎: 混合检索(alpha=0.5) -> Top 100 -> Filter -> Rerank(Top 20)
    return index.as_query_engine(
        similarity_top_k=int(os.getenv("TOP_K_RETRIEVAL", 100)),
        vector_store_kwargs={"query_mode": "hybrid", "hybrid_fusion_weight": 0.5},
        filters=filters,
        node_postprocessors=[APIReranker(top_n=int(os.getenv("TOP_N_RERANK", 20)))]
    )
```

#### `app/main.py` (入口与缓存)

```python
import os
import hashlib
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from cachetools import TTLCache
from typing import Optional, List
from app.ingest import process_file
from app.core import insert_nodes, get_query_engine
from app.utils import apply_time_decay

app = FastAPI(title="Finance RAG Engine")

# 本地 LRU 缓存: 容量 1000, 有效期 1 小时
CACHE = TTLCache(maxsize=1000, ttl=3600)

class ChatRequest(BaseModel):
    query: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None

@app.post("/ingest")
async def ingest_api(file: UploadFile = File(...)):
    if not file.filename.endswith(".md"):
        raise HTTPException(400, "Only .md supported")
    content = (await file.read()).decode("utf-8")
    nodes = await process_file(file.filename, content)
    if nodes: insert_nodes(nodes)
    return {"status": "ok", "chunks": len(nodes), "filename": file.filename}

@app.post("/chat")
async def chat_api(req: ChatRequest):
    # 1. 检查缓存
    cache_key = hashlib.md5(f"{req.query}_{req.start_date}_{req.end_date}".encode()).hexdigest()
    if cache_key in CACHE:
        return CACHE[cache_key]

    try:
        # 2. 执行查询 (Retrieval + Rerank)
        engine = get_query_engine(req.start_date, req.end_date)
        response = engine.query(req.query)
        
        # 3. 应用层时间衰减
        decay_rate = float(os.getenv("TIME_DECAY_RATE", 0.005))
        final_nodes = apply_time_decay(response.source_nodes, decay_rate=decay_rate)
        
        # 4. 截取最终 Top-K
        final_k = int(os.getenv("FINAL_TOP_K", 10))
        result_nodes = final_nodes[:final_k]
        
        # 5. 构造输出
        result = {
            "answer": str(response), # 包含 LLM 生成的回答
            "sources": [{
                "filename": n.metadata["filename"],
                "date": n.metadata["date"],
                "score": round(n.score, 4),
                "keywords": n.metadata.get("keywords"),
                "text": n.metadata.get("original_text", n.text)[:200]
            } for n in result_nodes]
        }
        
        # 6. 写入缓存
        CACHE[cache_key] = result
        return result
        
    except Exception as e:
        raise HTTPException(500, str(e))
```

> 说明：入库接口仅接受 Markdown (`.md`) 文件。这是因为在数据爬取阶段已经完成统一的 Markdown 化预处理，可以极大减轻 API 这一层的解析负担，是“重预处理、轻运行时”策略下的取舍。如果上传 PDF、TXT 等格式会直接返回 400，需在入库前完成 Markdown 转换。

-----

### 6\. 部署配置

#### `Dockerfile`

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖 (如有需要)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### `docker-compose.yml`

**关键点**：`QDRANT__STORAGE__OPTIMIZERS__DELETED_THRESHOLD` 和 volume 挂载确保了硬盘索引的性能和持久化。

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: rag_qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_data:/qdrant/storage
    environment:
      # 优化配置：更激进地释放内存，依赖磁盘
      - QDRANT__STORAGE__OPTIMIZERS__DELETED_THRESHOLD=0.5
    restart: always

  api:
    build: .
    container_name: rag_api
    ports:
      - "8000:8000"
    volumes:
      - ./app:/app  # 开发模式挂载源码
    env_file: .env
    depends_on:
      - qdrant
    restart: always
```

-----

### 7\. 使用指南

1.  **启动**: `docker-compose up -d --build`
2.  **入库 (Ingest)**:
      * 该步骤会触发 LLM 进行三合一预处理，速度稍慢但价值高。
      * 目前只接受 Markdown (`.md`) 文件，原因是爬虫阶段已经统一做过 Markdown 化处理，可大幅降低线上 CPU/GPU 占用；如需导入其它格式，请先完成离线转换。
      * `curl -X POST "http://localhost:8000/ingest" -F "file=@./data/20251001_NVIDIA_Report.md"`
      * 支持批量上传：`curl -X POST "http://localhost:8000/ingest/batch" -F "files=@a.md" -F "files=@b.md"`
3.  **查询 (Chat)**:
      * `curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"query": "英伟达最新财报表现", "start_date": "2025-01-01"}'`
      * 响应结构示例：
        ```json
        {
          "answer": "...",
          "sources": [
            {"filename": "20250228_NVIDIA_Report.md", "date": "2025-02-28", "score": 0.8431, "keywords": "NVDA,英伟达,GPU", "text": "......"}
          ]
        }
        ```

### 方案总结

  * **极简且强壮**: 没有复杂的 Vector DB 集群维护，单机 Qdrant + 硬盘索引抗住百万数据。
  * **智能**: 所有的“脏活累活”（表格、同义词、摘要）在入库时由 LLM 一次性干完，查询时享受高质量元数据的红利。
  * **兼容性**: 无论文件名怎么乱，多策略解析器都能兜底；无论 API 是 OpenAI 还是 DeepSeek，都能无缝接入。
  * **可控**: 时间衰减和缓存都在 Python 层控制，逻辑透明，易于调试参数。
