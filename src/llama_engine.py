import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PromptTemplate

load_dotenv()


# ─── STEP 1: Configure LLM and Embeddings ───────────────────

def setup_settings():
    print("🔄 Setting up LlamaIndex settings...")

    # Set Groq as the LLM
    Settings.llm = Groq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )

    # Set HuggingFace as embedding model (same one we used in Phase 4)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("✅ LLM and Embedding model configured!")


# ─── STEP 2: Load Documents ─────────────────────────────────

def load_documents(data_folder="data"):
    print(f"📄 Loading documents from {data_folder}...")

    documents = SimpleDirectoryReader(data_folder).load_data()

    print(f"✅ Loaded {len(documents)} document(s)")
    return documents


# ─── STEP 3: Build Index ────────────────────────────────────

def build_index(documents):
    print("🔄 Building VectorStoreIndex...")

    # LlamaIndex handles chunking and embedding internally
    index = VectorStoreIndex.from_documents(documents)

    print("✅ Index built successfully!")
    return index


# ─── STEP 4: Create Query Engine ────────────────────────────

def get_query_engine(index):
    # Custom prompt to prevent hallucination
    custom_prompt = PromptTemplate(
        """You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not found in the context, say "I don't know based on the provided documents."
Do NOT use your own knowledge. Do NOT make up any information.

Context:
{context_str}

Question:
{query_str}

Answer:
"""
    )

    query_engine = index.as_query_engine(
        similarity_top_k=4,
        text_qa_template=custom_prompt
    )
    print("✅ Query engine ready!")
    return query_engine


# ─── STEP 5: Ask a Question ─────────────────────────────────

def ask_question(query_engine, question):
    print(f"\n🤔 Question: {question}")
    print("🔄 Thinking...")

    response = query_engine.query(question)

    print(f"\n💬 Answer: {response}")
    print(f"\n📄 Sources used:")
    for i, node in enumerate(response.source_nodes):
        print(f"  Source {i+1}: Score {node.score:.4f}")
        print(f"  Preview: {node.text[:150]}...")

    return response


# ─── TEST ────────────────────────────────────────────────────

if __name__ == "__main__":

    # Step 1 - Setup
    setup_settings()

    # Step 2 - Load documents
    documents = load_documents()

    # Step 3 - Build index
    index = build_index(documents)

    # Step 4 - Get query engine
    query_engine = get_query_engine(index)

    # Step 5 - Ask same questions as Phase 5 to compare
    print("\n=== LlamaIndex QA Test ===")

    ask_question(query_engine, "What programming languages does this person know?")
    ask_question(query_engine, "What is this person's educational background?")
    ask_question(query_engine, "What is the capital of France?")

    # Step 6 - Compare with LangChain output
    print("\n=== Comparison Summary ===")
    print("LangChain: More control, explicit chunking and retrieval steps")
    print("LlamaIndex: Less code, handles chunking and embedding internally")
    print("Both use same embedding model and same Groq LLM")