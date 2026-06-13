import os

from dotenv import load_dotenv

from rag_utils import answer_question, create_vectorstore

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if __name__ == "__main__":
    pdf_path = "sample.pdf"
    print("Loading PDF...")
    vectorstore, chunk_count = create_vectorstore(pdf_path, api_key)
    print(f"Created {chunk_count} chunks.")
    print("Ready! Ask your questions.\n")

    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            break
        result = answer_question(vectorstore, question, api_key)
        if result["status"] == "ok":
            print(f"Bot: {result['answer']}")
            if result.get("sources"):
                print(f"Sources: {', '.join(result['sources'])}\n")
        elif result["status"] == "quota":
            print("Bot: Daily Gemini API quota reached. Try again later or use a different API key.\n")
        else:
            print(f"Bot: {result['answer']}\n")