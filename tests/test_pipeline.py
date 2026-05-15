"""
Unit tests for RAG pipeline components.
Run: python -m pytest tests/ -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

from ingestion.loader import DocumentLoader, Document
from ingestion.chunker import TextChunker, Chunk
from retrieval.vector_store import VectorStore
from generation.context_builder import ContextBuilder
from validation.grounding_validator import GroundingValidator


class TestDocumentLoader:

    def test_load_all_finds_documents(self):
        loader = DocumentLoader(os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs"))
        docs = loader.load_all()
        assert len(docs) >= 4
        assert all(isinstance(d, Document) for d in docs)

    def test_metadata_extraction(self):
        loader = DocumentLoader(os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs"))
        docs = loader.load_all()
        titled_docs = [d for d in docs if d.metadata.get("title")]
        assert len(titled_docs) > 0

    def test_empty_directory(self, tmp_path):
        loader = DocumentLoader(str(tmp_path))
        docs = loader.load_all()
        assert docs == []


class TestTextChunker:

    def test_basic_chunking(self):
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = " ".join(["word"] * 120)
        chunks = chunker.chunk_document(text, doc_id="test")
        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_small_text_single_chunk(self):
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        text = "This is a short document about diabetes."
        chunks = chunker.chunk_document(text, doc_id="test")
        assert len(chunks) == 1

    def test_chunk_ids_unique(self):
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = " ".join(["word"] * 200)
        chunks = chunker.chunk_document(text, doc_id="test")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_metadata_preserved(self):
        chunker = TextChunker(chunk_size=500)
        text = "Test content"
        metadata = {"title": "Test Doc", "source_org": "Test Org"}
        chunks = chunker.chunk_document(text, doc_id="t1", metadata=metadata)
        assert chunks[0].metadata["title"] == "Test Doc"


class TestVectorStore:

    def test_add_and_search(self):
        store = VectorStore(dimension=4)
        chunks = [
            Chunk(text="diabetes treatment metformin", chunk_id="c1", doc_id="d1"),
            Chunk(text="hypertension blood pressure", chunk_id="c2", doc_id="d2"),
        ]
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float32)

        store.add_chunks(chunks, embeddings)
        assert store.total_chunks == 2

        query = np.array([[0.9, 0.1, 0.0, 0.0]], dtype=np.float32)
        results = store.search(query, top_k=1)
        assert len(results) == 1
        assert results[0]["doc_id"] == "d1"

    def test_save_and_load(self, tmp_path):
        store = VectorStore(dimension=4)
        chunks = [Chunk(text="test", chunk_id="c1", doc_id="d1")]
        embeddings = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        store.add_chunks(chunks, embeddings)

        store.save(str(tmp_path))

        new_store = VectorStore(dimension=4)
        loaded = new_store.load(str(tmp_path))
        assert loaded is True
        assert new_store.total_chunks == 1

    def test_empty_search(self):
        store = VectorStore(dimension=4)
        query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        results = store.search(query)
        assert results == []


class TestContextBuilder:

    def test_qa_prompt_structure(self):
        builder = ContextBuilder()
        results = [
            {
                "text": "Metformin is the first-line treatment.",
                "doc_id": "HC-001",
                "metadata": {"title": "Diabetes Guidelines"},
                "score": 0.85,
            }
        ]
        messages = builder.build_prompt("What treats diabetes?", results)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "REFERENCE DOCUMENTS" in messages[1]["content"]
        assert "HC-001" in messages[1]["content"]

    def test_summarization_prompt(self):
        builder = ContextBuilder()
        results = [{"text": "Test content", "doc_id": "D1", "metadata": {}, "score": 0.9}]
        messages = builder.build_summarization_prompt("Summarize this", results)
        assert "SUMMARIZATION REQUEST" in messages[1]["content"]

    def test_empty_results_handled(self):
        builder = ContextBuilder()
        messages = builder.build_prompt("Test?", [])
        assert "No relevant documents found" in messages[1]["content"]


class TestGroundingValidator:

    def test_well_grounded_response(self):
        validator = GroundingValidator(strictness=0.3)
        response = "Metformin is the first-line treatment for Type 2 Diabetes. [Source: HC-001]"
        results = [
            {
                "text": "Metformin remains the recommended first-line pharmacological agent for Type 2 Diabetes.",
                "doc_id": "HC-001",
                "metadata": {},
            }
        ]
        validation = validator.validate(response, results)
        assert validation.overall_score > 0

    def test_citation_verification(self):
        validator = GroundingValidator()
        response = "Some fact [Source: HC-001]. Another fact [Source: FAKE-999]."
        results = [{"text": "source text", "doc_id": "HC-001", "metadata": {}}]
        validation = validator.validate(response, results)
        assert "HC-001" in validation.citation_check["valid_citations"]
        assert "FAKE-999" in validation.citation_check["invalid_citations"]

    def test_empty_response(self):
        validator = GroundingValidator()
        validation = validator.validate("", [{"text": "x", "doc_id": "d1", "metadata": {}}])
        assert validation.is_grounded is False

    def test_hallucination_detection_hedging(self):
        validator = GroundingValidator()
        response = "I think metformin is good. I believe it works."
        results = [{"text": "Metformin is first-line.", "doc_id": "HC-001", "metadata": {}}]
        validation = validator.validate(response, results)
        hedging_warnings = [w for w in validation.warnings if "Hedging" in w]
        assert len(hedging_warnings) > 0

    def test_validation_report_generated(self):
        validator = GroundingValidator()
        response = "Metformin treats diabetes. [Source: HC-001]"
        results = [{"text": "Metformin for diabetes.", "doc_id": "HC-001", "metadata": {}}]
        validation = validator.validate(response, results)
        assert "GROUNDING VALIDATION REPORT" in validation.details
        assert "Verdict:" in validation.details


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
