import os

import streamlit as st

from rag_utils import answer_question, create_vectorstore

st.set_page_config(page_title="PDF Q&A Bot", page_icon="📄", layout="centered")

# Load API Key: Check environment first (local), then Streamlit secrets (Cloud)
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        pass

if not api_key:
    st.error("❌ GEMINI_API_KEY not found. Please set it in your .env file or Streamlit secrets.")
    st.stop()


# ── CUSTOM CSS ──
st.markdown("""
    <style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }
    .title { 
        font-size: 2.2rem; font-weight: 800; 
        color: #ffffff; text-align: center; 
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem; color: #888888;
        text-align: center; margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #1e1e2e;
        border-left: 4px solid #7c3aed;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        color: #e2e8f0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .quota-box {
        background-color: #2d1b1b;
        border-left: 4px solid #ef4444;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        color: #fca5a5;
        font-size: 0.95rem;
    }
    .success-box {
        background-color: #1a2e1a;
        border-left: 4px solid #22c55e;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        color: #86efac;
        font-size: 0.9rem;
    }
    .stTextInput > div > div > input {
        background-color: #1e1e2e;
        color: #ffffff;
        border: 1px solid #7c3aed;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ── HEADER ──
st.markdown('<div class="title">📄 Financial PDF Q&A Bot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a financial document and ask questions about it using AI</div>', unsafe_allow_html=True)
st.divider()

# ── FILE UPLOAD ──
uploaded_file = st.file_uploader("📂 Upload your PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("⚙️ Processing your PDF..."):
        try:
            vectorstore, chunk_count = create_vectorstore(uploaded_file, api_key)
            st.markdown(f'<div class="success-box">✅ PDF processed successfully — <strong>{chunk_count} chunks</strong> created and indexed.</div>', unsafe_allow_html=True)
            st.session_state["vectorstore"] = vectorstore
        except Exception as e:
            st.error(f"Failed to process PDF: {str(e)}")

    st.divider()

    # ── QUESTION INPUT ──
    st.markdown("### 💬 Ask a Question")
    question = st.text_input("Ask a question", placeholder="e.g. What is the total revenue for FY2024?", label_visibility="collapsed")

    if question:
        with st.spinner("🤖 Thinking..."):
            result = answer_question(st.session_state["vectorstore"], question, api_key)

        if result["status"] == "ok":
            st.markdown("**🧠 Answer:**")
            st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
            if result.get("sources"):
                st.caption("Sources: " + ", ".join(result["sources"]))

        elif result["status"] == "quota":
            st.markdown(f"""
            <div class="quota-box">
                ⚠️ <strong>Daily API limit reached.</strong><br><br>
                You have exceeded the free tier quota for the Gemini API. 
                Please wait a few minutes and try again, or generate a new API key from 
                <a href="https://aistudio.google.com" target="_blank" style="color:#f87171;">aistudio.google.com</a>.
            </div>
            """, unsafe_allow_html=True)

        else:
            st.error(f"Something went wrong: {result['answer']}")

else:
    st.info("👆 Upload a PDF above to get started.")