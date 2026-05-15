"""
CLI demo for the Healthcare RAG System.
Run: python demo.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

from pipeline import RAGPipeline


def print_divider():
    print("=" * 70)


def run_demo():
    print_divider()
    print("  HEALTHCARE RAG SYSTEM — CLI DEMO")
    print("  Evidence-Grounded Question Answering & Summarization")
    print_divider()
    print()

    pipeline = RAGPipeline()
    pipeline.initialize()

    stats = pipeline.get_index_stats()
    print(f"Index Stats: {stats['total_chunks']} chunks | "
          f"Model: {stats['embedding_model']} | "
          f"Chunk size: {stats['chunk_size']}")
    print()

    demo_queries = [
        ("qa", "What is the first-line pharmacological treatment for Type 2 Diabetes and what are its contraindications?"),
        ("qa", "What screening tools are recommended for depression in primary care and how are their scores interpreted?"),
        ("summarize", "Summarize the key management strategies for Stage G4 chronic kidney disease."),
    ]

    for mode, question in demo_queries:
        print_divider()
        print(f"MODE: {mode.upper()}")
        print(f"QUERY: {question}")
        print_divider()

        result = pipeline.query(question, mode=mode)

        print("\nRESPONSE:")
        print(result["response"])

        print(f"\n--- SOURCES ({len(result['sources'])}) ---")
        for i, src in enumerate(result["sources"], 1):
            print(f"  [{i}] {src['doc_id']} — {src['title']} (score: {src['score']:.3f})")

        v = result["validation"]
        print(f"\n--- VALIDATION ---")
        print(f"  Grounded: {v['is_grounded']}")
        print(f"  Score: {v['grounding_score']:.2%}")
        if v["warnings"]:
            print(f"  Warnings:")
            for w in v["warnings"]:
                print(f"    - {w}")

        print(f"\n  Tokens: {result['token_usage']['total_tokens']}")
        print()

    print_divider()
    print("  INTERACTIVE MODE — type 'quit' to exit")
    print_divider()

    while True:
        print()
        query = input("Your question: ").strip()
        if not query or query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        mode_input = input("Mode (qa/summarize) [qa]: ").strip().lower()
        mode = "summarize" if mode_input == "summarize" else "qa"

        result = pipeline.query(query, mode=mode)
        print(f"\n{result['response']}")
        print(f"\n[Grounding: {result['validation']['grounding_score']:.0%} | "
              f"Sources: {len(result['sources'])}]")


if __name__ == "__main__":
    run_demo()
