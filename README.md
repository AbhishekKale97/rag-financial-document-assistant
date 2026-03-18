# 📄 Financial PDF Q&A Bot

An AI-powered chatbot that lets you upload any financial PDF document and ask questions about it in natural language. Built using RAG (Retrieval-Augmented Generation) architecture with Google Gemini and LangChain.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-latest-green?style=flat-square)
![Gemini](https://img.shields.io/badge/Google%20Gemini-2.0-orange?style=flat-square&logo=google)

---

## 🧠 How It Works
```
PDF Upload → Text Extraction → Chunking → Embedding → FAISS Vector Store
                                                              ↓
User Question → Embed Query → Similarity Search → Top 3 Chunks
                                                              ↓
                                              Gemini 2.0 Flash → Answer
```

1. **PDF Ingestion** — Extracts raw text from uploaded PDF using PyPDF
2. **Chunking** — Splits text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter
3. **Embedding** — Converts chunks into vector embeddings using Google's `gemini-embedding-001` model
4. **Vector Store** — Stores embeddings locally using FAISS for fast similarity search
5. **Retrieval** — On each question, finds the 3 most relevant chunks via cosine similarity
6. **Generation** — Passes retrieved context + question to Gemini 2.0 Flash to generate the answer

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Google Gemini 2.0 Flash |
| Embeddings | Google Gemini Embedding 001 |
| Vector Store | FAISS |
| PDF Parsing | PyPDF |
| Orchestration | LangChain |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/AbhishekKale97/pdf-qa-bot.git
cd pdf-qa-bot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a Gemini API Key
- Go to [aistudio.google.com](https://aistudio.google.com)
- Sign in → Click **Get API Key** → **Create API Key**
- Copy the key

### 4. Create a `.env` file
```
GEMINI_API_KEY=your_api_key_here
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure
```
pdf-qa-bot/
├── app.py              # Streamlit UI + main application
├── main.py             # CLI version of the app
├── .env                # API keys (not committed)
├── .gitignore          # Ignores .env and PDFs
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 📦 Requirements

Create a `requirements.txt` file with:
```
streamlit
langchain
langchain-community
langchain-google-genai
langchain-text-splitters
google-genai
pypdf
faiss-cpu
python-dotenv
numpy<2
```

---

## 💡 Use Cases

- Query annual reports and financial statements
- Extract insights from loan documents
- Analyze policy documents and contracts
- Ask questions about research papers

---

## ⚠️ Known Limitations

- Free tier Gemini API has daily quota limits — if you hit the limit, wait a few minutes or generate a new API key
- Large PDFs (100+ pages) may take longer to process
- Currently stores vector index in memory — reloading the page requires re-uploading the PDF

---

## 🔮 Future Improvements

- [ ] Deploy to Streamlit Cloud
- [ ] Support multiple PDFs simultaneously
- [ ] Persist vector store to disk
- [ ] Add chat history export
- [ ] Support for scanned PDFs via OCR

---

## 👨‍💻 Author

**Abhishek Kale**  
B.Tech AI & Data Science — DJSCE Mumbai  
[GitHub](https://github.com/AbhishekKale97)  
[Email](mailto:abhishekkale918@gmail.com)