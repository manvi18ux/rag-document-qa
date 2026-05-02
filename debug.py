import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pdf_loader import load_pdfs
from chunker import chunk_documents
from embedder import get_embeddings
from langchain_community.vectorstores import FAISS

# Step 1 - Load and chunk
print("Loading PDF...")
docs = load_pdfs("data/uploaded")
chunks = chunk_documents(docs)
print(f"Total chunks: {len(chunks)}")

# Step 2 - Build fresh vectorstore in memory
print("\nBuilding fresh vectorstore...")
embeddings = get_embeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
print("Done!")

# Step 3 - Test search for failing questions
questions = [
    "What is SERP Intelligence Collection?",
    "How many weeks is the development timeline?",
    "What is the novelty score threshold?"
]

for q in questions:
    print(f"\n{'='*50}")
    print(f"QUESTION: {q}")
    print("TOP 3 RETRIEVED CHUNKS:")
    results = vectorstore.similarity_search(q, k=3)
    for i, r in enumerate(results):
        print(f"\nChunk {i+1}:")
        print(r.page_content[:300])