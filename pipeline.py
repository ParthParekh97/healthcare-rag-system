"""
RAG Pipeline Orchestrator

Coordinates the full retrieve-augment-generate-validate flow:
  1. Ingest documents -> chunk -> embed -> index
  2. Accept query -> embed query -> retrieve top-K chunks
  3. Build grounded prompt with context engineering
  4. Generate response via LLM
  5. Validate response for hallucination and grounding
"""

import os

import config
from ingestion import DocumentLoader, TextChunker, Embedder
from retrieval import VectorStore
from generation import ContextBuilder, ResponseGenerator
from validation import GroundingValidator, ValidationResult


class RAGPipeline:

    def __init__(self):
        self.loader = DocumentLoader(config.DATA_DIR)
        self.chunker = TextChunker(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.context_builder = ContextBuilder()
        self.generator = ResponseGenerator()
        self.validator = GroundingValidator()
        self._initialized = False

    def initialize(self, force_rebuild: bool = False):
        if not force_rebuild and self.vector_store.load():
            print(f"Loaded existing index with {self.vector_store.total_chunks} chunks.")
            self._initialized = True
            return

        print("Building index from documents...")
        documents = self.loader.load_all()
        if not documents:
            raise RuntimeError(f"No documents found in {config.DATA_DIR}")

        print(f"Loaded {len(documents)} documents.")

        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk_document(
                content=doc.content,
                doc_id=doc.doc_id,
                metadata=doc.metadata,
            )
            all_chunks.extend(chunks)

        print(f"Created {len(all_chunks)} chunks.")

        texts = [chunk.text for chunk in all_chunks]
        embeddings = self.embedder.embed_texts(texts)

        self.vector_store = VectorStore(dimension=embeddings.shape[1])
        self.vector_store.add_chunks(all_chunks, embeddings)
        self.vector_store.save()

        print(f"Index built and saved with {self.vector_store.total_chunks} chunks.")
        self._initialized = True

    def query(self, question: str, top_k: int = None, mode: str = "qa") -> dict:
        if not self._initialized:
            self.initialize()

        query_embedding = self.embedder.embed_query(question)

        retrieved = self.vector_store.search(query_embedding, top_k=top_k)

        if mode == "summarize":
            messages = self.context_builder.build_summarization_prompt(question, retrieved)
        else:
            messages = self.context_builder.build_prompt(question, retrieved)

        gen_result = self.generator.generate_with_metadata(messages)
        response_text = gen_result["response"]

        validation = self.validator.validate(response_text, retrieved)

        return {
            "query": question,
            "response": response_text,
            "mode": mode,
            "sources": [
                {
                    "doc_id": r["doc_id"],
                    "title": r["metadata"].get("title", ""),
                    "section": r["metadata"].get("section", ""),
                    "score": r["score"],
                    "excerpt": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
                }
                for r in retrieved
            ],
            "validation": {
                "is_grounded": validation.is_grounded,
                "grounding_score": validation.overall_score,
                "warnings": validation.warnings,
                "details": validation.details,
                "citation_check": validation.citation_check,
            },
            "token_usage": gen_result["usage"],
        }

    def get_index_stats(self) -> dict:
        return {
            "total_chunks": self.vector_store.total_chunks,
            "embedding_model": config.EMBEDDING_MODEL,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
        }
