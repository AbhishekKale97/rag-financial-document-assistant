import re

from google import genai
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def create_client(api_key):
    return genai.Client(api_key=api_key)


def _clean_text(text):
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf_documents(source):
    reader = PdfReader(source)
    documents = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned_text = _clean_text(text)
        if cleaned_text:
            documents.append(
                Document(
                    page_content=cleaned_text,
                    metadata={"page": page_number},
                )
            )
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def get_embedding(text, client):
    result = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text,
    )
    return result.embeddings[0].values


class GeminiEmbeddings(Embeddings):
    def __init__(self, client):
        self.client = client

    def embed_documents(self, texts):
        # 1. Batch requests in chunks of 100 to stay safely within payload limits
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # 2. Pass the entire list directly to the API in ONE call
            result = self.client.models.embed_content(
                model="models/gemini-embedding-001",
                contents=batch,
            )
            
            # 3. Extract and accumulate the vectors
            all_embeddings.extend([e.values for e in result.embeddings])
            
        return all_embeddings

    def embed_query(self, text):
        # Single queries (like searching a question) remain a single string call
        result = self.client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=text,
        )
        return result.embeddings[0].values

    def __call__(self, text):
        return self.embed_query(text)



def create_vectorstore(source, api_key):
    client = create_client(api_key)
    documents = load_pdf_documents(source)
    chunks = split_documents(documents)
    embeddings = GeminiEmbeddings(client)
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    return vectorstore, len(chunks)


def retrieve_documents(vectorstore, question, k=4, fetch_k=12):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k},
    )
    return retriever.invoke(question)


def format_context(documents):
    parts = []
    sources = []
    for index, document in enumerate(documents, start=1):
        page = document.metadata.get("page", "?")
        parts.append(f"[Source {index} | page {page}]\n{document.page_content}")
        sources.append(f"page {page}")
    return "\n\n".join(parts), sources


def answer_question(vectorstore, question, api_key):
    try:
        documents = retrieve_documents(vectorstore, question)
        if not documents:
            return {"status": "empty", "answer": "I could not find any relevant text in the uploaded PDF.", "sources": []}

        context, sources = format_context(documents)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.1,
        )
        prompt = f"""You are a careful financial PDF assistant.
Use only the context below to answer the question.
If the answer is not explicitly supported by the context, say you could not find it in the uploaded PDF.
Prefer exact figures, dates, names, and units.
Keep the answer concise and factual.

Context:
{context}

Question: {question}

Answer:"""
        response = llm.invoke(prompt)
        return {"status": "ok", "answer": response.content, "sources": sources}
    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            return {"status": "quota", "answer": None, "sources": []}
        return {"status": "error", "answer": err, "sources": []}