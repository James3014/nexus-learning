import json

from nexus_learning.retrieval_audit import AuditEntry, log_retrieval_audit


def test_retrieval_audit_logger(tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    
    # 執行 logging
    log_retrieval_audit(
        AuditEntry(
            query="test query for python error",
            threshold=0.1,
            top_k=3,
            embedding_version="v2.0",
            hits=[("skill_abc", 0.95), ("skill_def", 0.88)],
            task_type="bug",
            task_id="feat-123",
            trace_id="trace-001"
        ),
        project_root=project_root
    )
    
    log_file = project_root / ".nexus" / "audit" / "retrieval_log.jsonl"
    assert log_file.exists()
    
    content = log_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(content) == 1
    
    data = json.loads(content[0])
    assert data["query"] == "test query for python error"
    assert data["task_id"] == "feat-123"
    assert data["trace_id"] == "trace-001"
    assert data["threshold"] == 0.1
    assert data["top_k"] == 3
    assert data["embedding_version"] == "v2.0"
    assert data["task_type"] == "bug"
    
    # Check hits
    assert len(data["hits"]) == 2
    assert data["hits"][0]["skill_id"] == "skill_abc"
    assert data["hits"][0]["score"] == 0.95
    assert data["hits"][1]["skill_id"] == "skill_def"
    assert data["hits"][1]["score"] == 0.88
    
    # Check timestamp exists
    assert "timestamp" in data
