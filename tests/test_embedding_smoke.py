"""Opt-in model download and real MiniLM retrieval verification."""

import os
from pathlib import Path

import pytest

from rag_chat.indexing import ingest


@pytest.mark.skipif(os.getenv("RAG_TEST_REAL_EMBEDDINGS") != "1", reason="Opt-in model download")
def test_real_minilm_long_document_and_retrieval(monkeypatch):
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
    from rag_chat.runtime import Runtime

    if cache := os.getenv("RAG_TEST_MODEL_CACHE"):
        monkeypatch.setattr(ONNXMiniLM_L6_V2, "DOWNLOAD_PATH", Path(cache))
    runtime = Runtime()
    library, _ = runtime.manager.ensure(None)
    try:
        tokenizer = runtime.tokenizer()
        # More than the embedding model's 256-token limit: the tail must survive.
        long_text = ("An apple is a fruit. " * 120 + "Project Cedar launches on June 15, 2027.")
        with runtime.manager.operation(library.id):
            result = ingest(library, "launch.txt", long_text.encode(), lambda: tokenizer)
            ingest(library, "ocean.txt", b"The ocean contains saltwater and marine animals.", lambda: tokenizer)
            assert result.chunks > 2
            all_documents = library.collection.get()["documents"]
            assert any("June 15, 2027" in text for text in all_documents)
            assert all(len(tokenizer.encode(text, add_special_tokens=False).ids) <= 200 for text in all_documents)
            search = library.collection.query(query_texts=["When does Project Cedar launch?"], n_results=1)
            assert "June 15, 2027" in search["documents"][0][0]
    finally:
        runtime.manager.clear(library.id)
        runtime.manager.close()
