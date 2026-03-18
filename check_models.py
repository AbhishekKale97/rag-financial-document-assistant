import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
print("Key loaded:", api_key[:8], "...")  # just shows first 8 chars to verify

client = genai.Client(api_key=api_key)
for model in client.models.list():
    print(model.name)