import hashlib
import json
import logging
import os
from typing import List

from chonkie import TokenChunker
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI

from app.models import LLMAnalysis
from app.openai_utils import get_openai_kwargs
from app.utils import extract_date_from_filename

logger = logging.getLogger("uvicorn.error")


extractor_llm = OpenAI(
    model=os.getenv("LLM_MODEL"),
    temperature=0.1,
    **get_openai_kwargs("LLM"),
)

chunker = TokenChunker(chunk_size=512, chunk_overlap=50)


def _strip_code_fence(content: str) -> str:
    if content.startswith("`") and content.endswith("`"):
        content = content.strip("")
    if content.startswith("`"):
        content = content.split("\n", 1)[1]
    if content.endswith("`"):
        content = content.rsplit("\n", 1)[0]
    return content


async def analyze_document(text: str) -> LLMAnalysis:
    context = text[:4000]
    prompt = f"""
你是一名资深金融分析师，负责将原始 Markdown 片段转化为检索用的富语义数据。请严格按照下列 JSON 结构输出（禁止额外文本或 Markdown 代码块）：
{{
  "summary": "<<=60字的中文摘要，突出业绩、风险或政策变化>",
  "table_narrative": "<将表格/数值转写成自然语言，若无表格留空字符串>",
  "keywords": ["<股票代码或公司名>", "<行业关键词>", "...至多8个唯一关键词"]
}}
要求：
- 使用中文输出，保持数值/单位原样。
- keywords 必须是字符串数组，含 ticker、机构简称、行业术语，避免重复。
- 若无法提取信息，字段填空字符串，keywords 为空数组。

待分析文档：
```markdown
{context}
```
"""
    try:
        logger.debug("Analyzing document chunk len=%d", len(context))
        response = await extractor_llm.acomplete(prompt)
        content = getattr(response, "text", str(response)).strip()
        content = _strip_code_fence(content)
        data = json.loads(content)
        logger.debug("Analyzer output keys=%s", list(data.keys()))
        return LLMAnalysis.model_validate(data)
    except Exception:
        logger.exception("Analyzer failed, returning empty analysis")
        return LLMAnalysis()


async def process_file(filename: str, content: str, ingest_date: str | None = None) -> List[TextNode]:
    meta_date = ingest_date or extract_date_from_filename(filename) or "1970-01-01"
    try:
        meta_date_numeric = int(meta_date.replace("-", ""))
    except ValueError:
        meta_date_numeric = 19700101
    logger.info("Processing file=%s extracted_date=%s", filename, meta_date)
    analysis = await analyze_document(content)

    keywords = analysis.keywords
    keywords_str = ",".join(keywords)
    header_text = (
        f"Date: {meta_date}\n"
        f"Summary: {analysis.summary}\n"
        f"Key Data: {analysis.table_narrative}\n"
        f"Tags: {keywords_str}\n"
        f"---\n"
    )

    nodes: List[TextNode] = []
    for idx, chunk in enumerate(chunker(content)):
        chunk_text = getattr(chunk, "text", str(chunk))
        full_text = f"{header_text}{chunk_text}"
        node = TextNode(
            text=full_text,
            metadata={
                "filename": filename,
                "date": meta_date,
                "date_numeric": meta_date_numeric,
                "keywords": keywords_str,
                "keyword_list": keywords,
                "original_text": chunk_text,
            },
        )
        node.id_ = hashlib.md5(f"{filename}_{idx}".encode()).hexdigest()
        nodes.append(node)
    logger.info("File=%s produced %d chunks", filename, len(nodes))
    return nodes
