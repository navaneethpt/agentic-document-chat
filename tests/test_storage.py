from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

import pytest

from rag_chat.documents import DocumentError
from rag_chat.indexing import ingest
from rag_chat.sessions import SessionExpired


def test_ingestion_deduplication_and_session_isolation(manager, tokenizer):
    first, _ = manager.ensure(None)
    second, _ = manager.ensure(None)
    ingest(first, "fruit.txt", b"apple apple apple", lambda: tokenizer)
    ingest(first, "sea.txt", b"ocean ocean ocean", lambda: tokenizer)
    duplicate = ingest(first, "renamed.txt", b"apple apple apple", lambda: tokenizer)
    assert duplicate.duplicate and first.collection.count() == 2
    assert len(first.documents) == 2
    assert second.collection.count() == 0
    result = first.collection.query(query_texts=["apple"], n_results=1)
    assert result["metadatas"][0][0]["filename"] == "fruit.txt"
    ingest(second, "other.txt", b"launch launch", lambda: tokenizer)
    assert second.collection.query(query_texts=["apple"], n_results=1)["documents"][0] == ["launch launch"]
    manager.clear(first.id)
    assert second.collection.count() == 1


def test_document_limit_allows_duplicate(manager, tokenizer):
    library, _ = manager.ensure(None)
    for index in range(10):
        ingest(library, f"{index}.txt", f"text {index}".encode(), lambda: tokenizer)
    assert ingest(library, "again.txt", b"text 0", lambda: tokenizer).duplicate
    with pytest.raises(DocumentError, match="10 documents"):
        ingest(library, "eleven.txt", b"new content", lambda: tokenizer)


def test_partial_batch_failure_rolls_back_only_failed_document(manager, tokenizer):
    library, _ = manager.ensure(None)
    ingest(library, "existing.txt", b"Keep this", lambda: tokenizer)
    original = library.collection
    proxy = Mock(wraps=original)
    calls = 0

    def fail_second(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Embedding failed")
        return original.add(**kwargs)

    proxy.add.side_effect = fail_second
    library.collection = proxy
    with pytest.raises(DocumentError, match="rolled back"):
        ingest(library, "fail.txt", b"word " * 450, lambda: tokenizer, batch_size=1)
    library.collection = original
    assert original.count() == 1
    assert original.get()["documents"] == ["Keep this"]
    assert len(library.documents) == 1
    ingest(library, "retry.txt", b"word " * 450, lambda: tokenizer)
    assert len(library.documents) == 2


def test_failed_rollback_quarantines_library(manager, tokenizer):
    library, _ = manager.ensure(None)
    original = library.collection
    proxy = Mock(wraps=original)
    proxy.add.side_effect = RuntimeError()
    proxy.delete.side_effect = RuntimeError()
    library.collection = proxy
    with pytest.raises(DocumentError, match="cleanup failed"):
        ingest(library, "x.txt", b"Some text", lambda: tokenizer)
    assert not library.healthy
    with pytest.raises(DocumentError, match="reset"):
        ingest(library, "y.txt", b"Other text", lambda: tokenizer)
    library.collection = original


def test_expiry_replacement_and_clear(manager):
    now = [0.0]
    manager.clock = lambda: now[0]
    library, fresh = manager.ensure(None)
    assert fresh
    now[0] = 3599
    assert manager.cleanup() == 0
    now[0] = 3600
    replacement, fresh = manager.ensure(library.id)
    assert fresh and replacement.id != library.id
    with pytest.raises(SessionExpired):
        with manager.operation(library.id):
            pass
    manager.clear(replacement.id)
    assert manager.client.list_collections() == []


def test_cleanup_skips_active_operation(manager):
    now = [0.0]
    manager.clock = lambda: now[0]
    library, _ = manager.ensure(None)
    started, finish = Event(), Event()

    def operation():
        with manager.operation(library.id):
            started.set()
            assert finish.wait(5)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(operation)
        assert started.wait(5)
        now[0] = 7200
        try:
            assert manager.cleanup() == 0
            with pytest.raises(RuntimeError, match="wait"):
                manager.clear(library.id)
        finally:
            finish.set()
        future.result()
    assert library.last_used == 7200
    now[0] = 10800
    assert manager.cleanup() == 1
