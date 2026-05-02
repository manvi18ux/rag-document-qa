import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


# ─── LLM ────────────────────────────────────────────────────

def get_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )


# ─── TECHNIQUE 1: MMR RETRIEVAL ─────────────────────────────

def build_mmr_chain(vectorstore):
    print("🔄 Building MMR retrieval chain...")

    llm = get_llm()

    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )

    prompt = ChatPromptTemplate.from_template("""
You are a helpful and detailed assistant. Answer the question using ONLY the context below.
If the answer is not found, say "I don't know based on the provided documents."
Do NOT make up information.

Context:
{context}

Question:
{input}

Detailed Answer:
""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    mmr_chain = create_retrieval_chain(mmr_retriever, document_chain)

    print("✅ MMR chain ready!")
    return mmr_chain


# ─── TECHNIQUE 2: MULTIQUERY RETRIEVAL ──────────────────────

def build_multiquery_chain(vectorstore):
    print("🔄 Building MultiQuery retrieval chain...")

    llm = get_llm()

    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        llm=llm
    )

    prompt = ChatPromptTemplate.from_template("""
You are a helpful and detailed assistant. Answer the question using ONLY the context below.
If the answer is not found, say "I don't know based on the provided documents."
Do NOT make up information.

Context:
{context}

Question:
{input}

Detailed Answer:
""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    multiquery_chain = create_retrieval_chain(multiquery_retriever, document_chain)

    print("✅ MultiQuery chain ready!")
    return multiquery_chain


# ─── TECHNIQUE 3: HyDE RETRIEVAL ────────────────────────────

def build_hyde_chain(vectorstore):
    print("🔄 Building HyDE retrieval chain...")

    llm = get_llm()

    hyde_prompt = ChatPromptTemplate.from_template("""
Write a short hypothetical document that would answer this question.
Write it as if it is extracted from a real document.
Question: {question}
Hypothetical document:
""")

    hyde_chain = (
        {"question": RunnablePassthrough()}
        | hyde_prompt
        | llm
        | StrOutputParser()
    )

    class HyDERetriever:
        def __init__(self, vectorstore, hyde_chain):
            self.vectorstore = vectorstore
            self.hyde_chain = hyde_chain

        def get_relevant_documents(self, query):
            hypothetical_doc = self.hyde_chain.invoke(query)
            print(f"\n📝 Hypothetical doc generated: {hypothetical_doc[:100]}...")
            return self.vectorstore.similarity_search(hypothetical_doc, k=4)

    hyde_retriever = HyDERetriever(vectorstore, hyde_chain)

    def hyde_qa(question):
        docs = hyde_retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        full_prompt = f"""
You are a helpful and detailed assistant. Answer using ONLY the context below.
If not found say "I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Detailed Answer:
"""
        response = llm.invoke(full_prompt)
        return {
            "answer": response.content,
            "context": docs
        }

    print("✅ HyDE chain ready!")
    return hyde_qa


# ─── TECHNIQUE 4: RERANKER ──────────────────────────────────

def build_reranker_chain(vectorstore):
    print("🔄 Building Reranker chain...")

    from flashrank import Ranker, RerankRequest
    llm = get_llm()

    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

    def rerank_and_answer(question):
        # Step 1 - Retrieve top 10 chunks
        initial_docs = vectorstore.similarity_search(question, k=10)
        print(f"📥 Retrieved {len(initial_docs)} initial chunks")

        # Step 2 - Rerank using flashrank
        passages = [
            {"id": i, "text": doc.page_content}
            for i, doc in enumerate(initial_docs)
        ]

        rerank_request = RerankRequest(
            query=question,
            passages=passages
        )

        results = ranker.rerank(rerank_request)

        # Step 3 - Take top 3 after reranking
        top_3_indices = [r["id"] for r in results[:3]]
        top_3_docs = [initial_docs[i] for i in top_3_indices]
        print(f"✅ Reranked to top 3 chunks")

        # Step 4 - Generate answer from top 3
        context = "\n\n".join([doc.page_content for doc in top_3_docs])
        full_prompt = f"""
You are a helpful assistant. Answer using ONLY the context below.
If not found say "I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Detailed Answer:
"""
        response = llm.invoke(full_prompt)
        return {
            "answer": response.content,
            "context": top_3_docs
        }

    print("✅ Reranker chain ready!")
    return rerank_and_answer


# ─── COMPARE ALL 4 TECHNIQUES ───────────────────────────────

def compare_techniques(vectorstore, question):
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}")

    # Basic retrieval
    from qa_chain import build_qa_chain
    basic_chain = build_qa_chain(vectorstore)
    basic_result = basic_chain.invoke({"input": question})
    print(f"\n📌 BASIC RAG:")
    print(f"Answer: {basic_result['answer'][:300]}")
    print(f"Chunks retrieved: {len(basic_result['context'])}")

    # MMR retrieval
    mmr_chain = build_mmr_chain(vectorstore)
    mmr_result = mmr_chain.invoke({"input": question})
    print(f"\n🎯 MMR RETRIEVAL:")
    print(f"Answer: {mmr_result['answer'][:300]}")
    print(f"Chunks retrieved: {len(mmr_result['context'])}")

    # MultiQuery retrieval
    multiquery_chain = build_multiquery_chain(vectorstore)
    multiquery_result = multiquery_chain.invoke({"input": question})
    print(f"\n🔀 MULTIQUERY RETRIEVAL:")
    print(f"Answer: {multiquery_result['answer'][:300]}")

    # HyDE retrieval
    hyde_qa = build_hyde_chain(vectorstore)
    hyde_result = hyde_qa(question)
    print(f"\n💡 HyDE RETRIEVAL:")
    print(f"Answer: {hyde_result['answer'][:300]}")

    # Reranker
    reranker_qa = build_reranker_chain(vectorstore)
    reranker_result = reranker_qa(question)
    print(f"\n🏆 RERANKER:")
    print(f"Answer: {reranker_result['answer'][:300]}")
    print(f"Chunks used: {len(reranker_result['context'])}")


# ─── TEST ────────────────────────────────────────────────────

if __name__ == "__main__":
    from vectorstore import load_faiss_vectorstore

    print("=== Loading Vector Store ===")
    vectorstore = load_faiss_vectorstore()

    compare_techniques(
        vectorstore,
        "What programming languages does this person know?"
    )