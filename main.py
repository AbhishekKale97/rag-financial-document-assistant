import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def load_pdf(path):
    reader = PdfReader(path)
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
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore

def answer_question(vectorstore, question):
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
    return response.content

if __name__ == "__main__":
    pdf_path = "sample.pdf"
    print("Loading PDF...")
    text = load_pdf(pdf_path)
    print("Splitting text...")
    chunks = split_text(text)
    print(f"Created {len(chunks)} chunks.")
    print("Building vector store...")
    vectorstore = create_vectorstore(chunks)
    print("Ready! Ask your questions.\n")

    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            break
        answer = answer_question(vectorstore, question)
        print(f"Bot: {answer}\n")