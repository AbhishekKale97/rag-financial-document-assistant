import os
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai

st.set_page_config(page_title="PDF Q&A Bot", page_icon="📄", layout="centered")
# Works both locally and on Streamlit Cloud
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def get_embedding(text):
    result = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values

class GeminiEmbeddings:
    def embed_documents(self, texts):
        return [get_embedding(t) for t in texts]
    def embed_query(self, text):
        return get_embedding(text)
    def __call__(self, text):
        return get_embedding(text)

def create_vectorstore(chunks):
    embeddings = GeminiEmbeddings()
    return FAISS.from_texts(chunks, embedding=embeddings)

def answer_question(vectorstore, question):
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.3
        )
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""
        response = llm.invoke(prompt)
        return {"status": "ok", "answer": response.content}
    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            return {"status": "quota", "answer": None}
        return {"status": "error", "answer": err}


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
            text = load_pdf(uploaded_file)
            chunks = split_text(text)
            vectorstore = create_vectorstore(chunks)
            st.markdown(f'<div class="success-box">✅ PDF processed successfully — <strong>{len(chunks)} chunks</strong> created and indexed.</div>', unsafe_allow_html=True)
            st.session_state["vectorstore"] = vectorstore
        except Exception as e:
            st.error(f"Failed to process PDF: {str(e)}")

    st.divider()

    # ── QUESTION INPUT ──
    st.markdown("### 💬 Ask a Question")
    question = st.text_input("", placeholder="e.g. What is the total revenue for FY2024?")

    if question:
        with st.spinner("🤖 Thinking..."):
            result = answer_question(st.session_state["vectorstore"], question)

        if result["status"] == "ok":
            st.markdown("**🧠 Answer:**")
            st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

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