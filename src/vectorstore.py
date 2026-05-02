import os

from altair import When
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from embedder import get_embeddings
from pdf_loader import load_pdfs
from web_loader import load_multiple_websites
from chunker import chunk_documents

FAISS_PATH = "vectorstore/faiss_index"
CHROMA_PATH = "vectorstore/chroma_index"


# ─── FAISS FUNCTIONS ────────────────────────────────────────

def build_faiss_vectorstore(chunks):
    print("🔄 Building FAISS vector store...")
    
    # Always build fresh, never load old index
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    os.makedirs("vectorstore", exist_ok=True)
    vectorstore.save_local(FAISS_PATH)
    print(f"✅ FAISS index saved to {FAISS_PATH}")
    
    return vectorstore


def load_faiss_vectorstore():
    print("🔄 Loading FAISS vector store from disk...")
    embeddings = get_embeddings()

    vectorstore = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("✅ FAISS index loaded!")
    return vectorstore


# ─── CHROMA FUNCTIONS ───────────────────────────────────────

def build_chroma_vectorstore(chunks):
    print("🔄 Building Chroma vector store...")
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"✅ Chroma index saved to {CHROMA_PATH}")
    return vectorstore


def load_chroma_vectorstore():
    print("🔄 Loading Chroma vector store...")
    embeddings = get_embeddings()

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    print("✅ Chroma index loaded!")
    return vectorstore


# ─── TEST ────────────────────────────────────────────────────

if __name__ == "__main__":

    # Step 1 - Load and chunk documents
    print("=== Step 1: Loading documents ===")
    pdf_docs = load_pdfs()
    chunks = chunk_documents(pdf_docs)

    # Step 2 - Build FAISS index
    print("\n=== Step 2: Building FAISS index ===")
    faiss_store = build_faiss_vectorstore(chunks)

    # Step 3 - Test similarity search on FAISS
    print("\n=== Step 3: Testing FAISS similarity search ===")
    query = "What programming languages does this person know?"
    results = faiss_store.similarity_search(query, k=2)

    print(f"\nQuery: {query}")
    print(f"Top {len(results)} results:\n")
    for i, doc in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Text: {doc.page_content[:200]}")
        print(f"Metadata: {doc.metadata}")
        print()

    # Step 4 - Save and reload FAISS
    print("=== Step 4: Reload FAISS from disk ===")
    loaded_store = load_faiss_vectorstore()
    results2 = loaded_store.similarity_search(query, k=2)
    print(f"✅ Reload works! Got {len(results2)} results after reloading\n")

    # Step 5 - Build Chroma index
    print("=== Step 5: Building Chroma index ===")
    chroma_store = build_chroma_vectorstore(chunks)

    # Step 6 - Test similarity search on Chroma
    print("\n=== Step 6: Testing Chroma similarity search ===")
    results3 = chroma_store.similarity_search(query, k=2)
    print(f"Query: {query}")
    print(f"✅ Chroma returned {len(results3)} results")
    print(f"First result: {results3[0].page_content[:200]}")

#It used the embedder to do 3 things:
#1 — Stored your chunks as vectors
#It took all 5 chunks from your resume, converted each one into 384 numbers using the embedder, and stored all of them in FAISS.
#2 — Saved the index to disk
#It saved those vectors inside your vectorstore/faiss_index folder so you don't have to rebuild it every time.
#3 — Searched by meaning
#When you asked "What programming languages does this person know?" it converted that question into 384 numbers, compared it against all stored chunk vectors, and returned the 2 most similar chunks — which should contain Python, JavaScript etc from the resume.