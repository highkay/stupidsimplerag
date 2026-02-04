import pytest
from io import BytesIO
import time
import os

def test_full_workflow_real_service(client):
    """
    全链路集成测试：
    1. 检查 Dashboard 连通性
    2. 上传真实文件（触发真实 LLM 解析与向量化）
    3. 验证文档列表
    4. 验证删除逻辑
    """
    # 1. Dashboard 连通性
    resp_dash = client.get("/")
    assert resp_dash.status_code == 200
    assert "pytest_integration_test" in resp_dash.text

    # 2. 上传测试文档 (使用真实逻辑)
    fname = "real_integration_test.md"
    # 构造一段有意义的文本，方便 LLM 提取关键词
    content = """
    # 2026年半导体行业趋势报告
    受 AI 芯片需求驱动，预计 2026 年全球半导体市场规模将增长 15%。
    主要风险包括供应链波动和先进制程良率问题。
    """
    file = (fname, BytesIO(content.encode("utf-8")), "text/markdown")
    
    print(f"\n[Step] Ingesting {fname} (Real LLM/Embedding)...")
    resp_ingest = client.post(
        "/ingest",
        files={"file": file},
        data={"force_update": "true"}
    )
    assert resp_ingest.status_code == 200
    res_data = resp_ingest.json()
    assert res_data["status"] == "ok"
    assert res_data["chunks"] > 0
    print(f"Success: Created {res_data['chunks']} chunks.")

    # 3. 验证列表显示
    time.sleep(1) # 等待 Qdrant 写入确认
    resp_list = client.get("/documents")
    assert resp_list.status_code == 200
    docs = resp_list.json()
    assert any(d["filename"] == fname for d in docs)

    # 4. 测试 Chat 检索 (验证向量检索是否通畅)
    chat_payload = {
        "query": "2026年半导体市场增长预测是多少？",
        "skip_generation": True # 仅验证检索链路
    }
    resp_chat = client.post("/chat", json=chat_payload)
    assert resp_chat.status_code == 200
    chat_data = resp_chat.json()
    # 验证是否检索到了刚才上传的文件
    assert any(s["filename"] == fname for s in chat_data["sources"])
    print(f"Success: Found source with score {chat_data['sources'][0]['score']}")

    # 5. 删除清理
    resp_del = client.delete(f"/documents/{fname}")
    assert resp_del.status_code == 200
    
    # 验证列表为空
    time.sleep(1)
    resp_list_final = client.get("/documents")
    assert not any(d["filename"] == fname for d in resp_list_final.json())
    print("Cleanup: Document deleted successfully.")
