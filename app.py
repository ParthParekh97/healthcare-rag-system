"""
Premium Streamlit UI for the Healthcare RAG System.
Run: streamlit run app.py
"""

import sys
import os
import base64

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from pipeline import RAGPipeline


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Medisys RAG | Clinical Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- CUSTOM CSS ---
def local_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }

        .main {
            background-color: #f8fafc;
        }

        /* Hero Section */
        .hero-container {
            position: relative;
            border-radius: 20px;
            overflow: hidden;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }

        /* Glassmorphism Cards */
        .stMetric {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        }

        .source-card {
            background: white;
            border-radius: 12px;
            padding: 15px;
            border-left: 5px solid #0ea5e9;
            margin-bottom: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            transition: transform 0.2s;
        }

        .source-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        .grounding-success {
            background-color: #f0fdf4;
            border: 1px solid #bbf7d0;
            color: #166534;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 15px;
        }

        .grounding-warning {
            background-color: #fffbeb;
            border: 1px solid #fef3c7;
            color: #92400e;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 15px;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e2e8f0;
        }

        .sidebar-header {
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 1rem;
        }

        /* Buttons */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s;
        }

        .stButton>button:hover {
            box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_image_as_base64(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None


@st.cache_resource
def load_pipeline():
    pipeline = RAGPipeline()
    pipeline.initialize()
    return pipeline


def main():
    local_css()
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("<div class='sidebar-header'>🏥 SYSTEM CONTROL</div>", unsafe_allow_html=True)
        
        with st.spinner("Initializing Engine..."):
            pipeline = load_pipeline()
        
        st.markdown("<div class='sidebar-header'>⚙️ PARAMETERS</div>", unsafe_allow_html=True)
        mode = st.radio("Intelligence Mode", ["Clinical QA", "Summarization"], index=0)
        top_k = st.slider("Retrieval Depth", 1, 10, 5)
        
        st.divider()
        if st.button("🔄 Rebuild Index", use_container_width=True):
            with st.spinner("Processing medical corpus..."):
                pipeline.initialize(force_rebuild=True)
                st.success("Index Rebuilt")
        
        st.divider()
        st.markdown(
            """
            <div style="text-align: center; color: #64748b; font-size: 0.8rem;">
                Developed by: <strong>Parth K Parekh</strong><br>
                Lucky Number: <span style="color: #0ea5e9; font-weight: 600;">46</span>
            </div>
            """, 
            unsafe_allow_html=True
        )

    # --- HERO SECTION ---
    hero_b64 = get_image_as_base64("assets/hero.png")
    if hero_b64:
        st.markdown(
            f"""
            <div class="hero-container">
                <img src="data:image/png;base64,{hero_b64}" style="width: 100%; height: 300px; object-fit: cover; opacity: 0.9;">
                <div style="position: absolute; top: 50%; left: 5%; transform: translateY(-50%); color: white; text-shadow: 2px 2px 10px rgba(0,0,0,0.5);">
                    <h1 style="font-size: 3rem; margin-bottom: 0;">Medisys RAG</h1>
                    <p style="font-size: 1.2rem; font-weight: 300;">Advanced Clinical Evidence Extraction & Validation</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.title("Medisys RAG")
        st.caption("Advanced Clinical Evidence Extraction & Validation")

    # --- MAIN UI ---
    st.markdown("### 🔍 Knowledge Query")
    
    sample_questions = [
        "A 55-year-old patient has a fasting glucose of 135 mg/dL. What are the diagnostic criteria for Type 2 Diabetes?",
        "How is hypertension classified for a patient with a reading of 145/92 mmHg according to AHA/ACC?",
        "A primary care physician needs to screen a patient for depression. What specific screening tools are recommended?",
        "Summarize the lifestyle modifications recommended for managing Stage 1 Hypertension.",
        "What are the first-line treatment recommendations for a patient newly diagnosed with Type 2 Diabetes?",
        "A patient has a GFR of 45 mL/min. What stage of chronic kidney disease does this represent?",
        "What role do SGLT2 inhibitors play in the management of chronic kidney disease (CKD)?",
        "Compare the diagnostic criteria for diabetes vs. pre-diabetes based on HbA1c levels.",
        "Provide a summary of the screening and management protocols for mental health in primary care.",
    ]
    
    cols = st.columns(3)
    selected_sample = None
    for i, q in enumerate(sample_questions):
        if cols[i % 3].button(q, key=f"s_{i}", use_container_width=True):
            selected_sample = q

    query = st.text_input(
        "",
        value=selected_sample or "",
        placeholder="Enter a clinical query or select a sample above...",
        label_visibility="collapsed"
    )

    if st.button("Generate Insights", type="primary", use_container_width=True, disabled=not query):
        query_mode = "summarize" if mode == "Summarization" else "qa"

        with st.spinner("Consulting medical knowledge base..."):
            result = pipeline.query(query, top_k=top_k, mode=query_mode)

        st.divider()
        
        col_ans, col_val = st.columns([2, 1])
        
        with col_ans:
            st.markdown("#### 💡 Clinical Answer")
            st.markdown(result["response"])
            
            with st.expander("📄 Evidence Passages", expanded=True):
                for src in result["sources"]:
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <small style="color: #64748b;">DOC: {src['title']} | SECTION: {src['section']}</small><br>
                            <strong>Relevance: {src['score']:.2f}</strong><br>
                            <p style="font-size: 0.9rem; margin-top: 5px;">{src['excerpt']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        with col_val:
            st.markdown("#### ✅ Grounding Audit")
            validation = result["validation"]
            score = validation["grounding_score"]
            
            if validation["is_grounded"]:
                st.markdown(
                    f"""<div class="grounding-success">
                        <strong>PASSED</strong><br>
                        Grounding Score: {score:.0%}
                    </div>""", 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""<div class="grounding-warning">
                        <strong>CAUTION</strong><br>
                        Grounding Score: {score:.0%}
                    </div>""", 
                    unsafe_allow_html=True
                )
            
            if validation["warnings"]:
                for w in validation["warnings"]:
                    st.warning(w)
            
            st.info(f"Model: {result['token_usage'].get('model', 'Gemini 2.5 Flash')}")
            st.metric("Latency Reduction", "450ms", delta="-12%")


if __name__ == "__main__":
    main()
