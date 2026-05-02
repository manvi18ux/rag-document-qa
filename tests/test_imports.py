import langchain
import llama_index
import faiss
import chromadb
import streamlit
import fastapi
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('GROQ_API_KEY')

if api_key:
    print("✅ All libraries imported successfully!")
    print(f"✅ Groq API key loaded: {api_key[:10]}...")
else:
    print("❌ Groq API key NOT found. Check your .env file.")