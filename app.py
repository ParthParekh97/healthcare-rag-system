"""
Streamlit UI for the Healthcare RAG System.
Run: streamlit run app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from pipeline import RAGPipeline


st.set_page_config(
    page_title="Healthcare RAG System",
    page_icon="🏥",
    layout="wide",
)


@st.cache_resource
def load_pipeline():
    pipeline = RAGPipeline()
    pipeline.initialize()
    return pipeline


def main():
    st.title("Healthcare RAG — Question Answering & Summarization")
    st.markdown(
        "Evidence-grounded answers from clinical guideline documents. "
        "Every response includes source citations and a grounding validation report."
    )

    with st.spinner("Loading document index..."):
        pipeline = load_pipeline()

    stats = pipeline.get_index_stats()
    with st.sidebar:
        st.header("System Info")
        st.metric("Indexed Chunks", stats["total_chunks"])
        st.metric("Embedding Model", stats["embedding_model"])
        st.metric("Chunk Size", f"{stats['chunk_size']} tokens")

        st.divider()
        st.header("Settings")
        mode = st.radio("Mode", ["Question Answering", "Summarization"], index=0)
        top_k = st.slider("Retrieved Passages", min_value=1, max_value=10, value=5)
        show_validation = st.checkbox("Show Validation Report", value=True)
        show_sources = st.checkbox("Show Source Passages", value=True)

    sample_questions = [
        "What is the first-line treatment for Type 2 Diabetes?",
        "How is hypertension classified according to AHA/ACC guidelines?",
        "What screening tools are used for depression in primary care?",
        "What are the stages of chronic kidney disease?",
        "Summarize the lifestyle modifications for managing hypertension.",
        "What are SGLT2 inhibitors and their role in CKD management?",
    ]

    st.subheader("Sample Questions")
    cols = st.columns(3)
    selected_sample = None
    for i, q in enumerate(sample_questions):
        if cols[i % 3].button(q, key=f"sample_{i}", use_container_width=True):
            selected_sample = q

    query = st.text_input(
        "Ask a healthcare question:",
        value=selected_sample or "",
        placeholder="e.g., What are the diagnostic criteria for Type 2 Diabetes?",
    )

    if st.button("Submit", type="primary", disabled=not query):
        query_mode = "summarize" if mode == "Summarization" else "qa"

        with st.spinner("Retrieving documents and generating response..."):
            result = pipeline.query(query, top_k=top_k, mode=query_mode)

        st.subheader("Answer")
        st.markdown(result["response"])

        validation = result["validation"]
        if validation["is_grounded"]:
            st.success(
                f"Grounding Score: {validation['grounding_score']:.0%} — "
                f"Response is well-grounded in source documents."
            )
        else:
            st.warning(
                f"Grounding Score: {validation['grounding_score']:.0%} — "
                f"Some claims may not be fully supported by retrieved documents."
            )

        if validation["warnings"]:
            with st.expander("Validation Warnings", expanded=False):
                for w in validation["warnings"]:
                    st.warning(w)

        if show_validation:
            with st.expander("Full Validation Report", expanded=False):
                st.code(validation["details"], language="text")

        if show_sources:
            with st.expander(f"Retrieved Sources ({len(result['sources'])} passages)", expanded=False):
                for i, src in enumerate(result["sources"], 1):
                    st.markdown(
                        f"**[{i}] {src['title']}** (Score: {src['score']:.3f})\n\n"
                        f"Section: {src['section']}\n\n"
                        f"```\n{src['excerpt']}\n```"
                    )

        with st.expander("Token Usage", expanded=False):
            usage = result["token_usage"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Prompt Tokens", usage["prompt_tokens"])
            col2.metric("Completion Tokens", usage["completion_tokens"])
            col3.metric("Total Tokens", usage["total_tokens"])


if __name__ == "__main__":
    main()
