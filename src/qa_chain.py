import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from vectorstore import load_faiss_vectorstore

load_dotenv()


# ─── STEP 1: Load LLM ───────────────────────────────────────

def get_llm():
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )
    print("✅ Groq LLM loaded!")
    return llm


# ─── STEP 2: Build QA Chain ─────────────────────────────────

def build_qa_chain(vectorstore):
    print("🔄 Building QA chain...")

    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    # Custom prompt
    prompt = ChatPromptTemplate.from_template("""
You are a helpful and detailed assistant. Answer the question using ONLY the context provided below.
Give a complete and explanatory answer - not just a one line answer.
Include relevant details, examples and explanations from the context.
If the answer is not found in the context, say "I don't know based on the provided documents."
Do NOT make up any information.

Context:
{context}

Question:
{input}

Detailed Answer:
""")

    # Chain that stuffs documents into prompt
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Full retrieval chain
    qa_chain = create_retrieval_chain(retriever, document_chain)

    print("✅ QA Chain ready!")
    return qa_chain


# ─── STEP 3: Ask a Question ─────────────────────────────────

def ask_question(qa_chain, question):
    print(f"\n🤔 Question: {question}")
    print("🔄 Thinking...")

    result = qa_chain.invoke({"input": question})

    print(f"\n💬 Answer: {result['answer']}")
    print(f"\n📄 Sources used:")
    for i, doc in enumerate(result['context']):
        print(f"  Source {i+1}: {doc.metadata.get('source', 'Unknown')} - Page {doc.metadata.get('page', 'N/A')}")
        print(f"  Preview: {doc.page_content[:150]}...")

    return result


# ─── STEP 4: Conversational Chain (with memory) ─────────────

def ask_with_history(qa_chain, question, chat_history):
    print(f"\n🤔 Question: {question}")
    print("🔄 Thinking...")

    # Only send the current question to retriever
    # NOT the full history — history breaks retrieval
    result = qa_chain.invoke({"input": question})

    # Save to history
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=result["answer"]))

    print(f"💬 Answer: {result['answer']}")
    return result, chat_history

# ─── TEST ────────────────────────────────────────────────────

if __name__ == "__main__":

    # Load vector store from Phase 4
    print("=== Loading Vector Store ===")
    vectorstore = load_faiss_vectorstore()

    # Build chain
    qa_chain = build_qa_chain(vectorstore)

    # Test 1 - Question from the document
    ask_question(qa_chain, "What programming languages does this person know?")

    # Test 2 - Question from the document
    ask_question(qa_chain, "What is this person's educational background?")

    # Test 3 - Question NOT in document (should say I don't know)
    ask_question(qa_chain, "What is the capital of France?")

    # Test 4 - Multi turn conversation with memory
    print("\n=== Test 4: Multi-turn Conversation ===")
    chat_history = []

    result1, chat_history = ask_with_history(
        qa_chain,
        "What companies did this person work at?",
        chat_history
    )

    result2, chat_history = ask_with_history(
        qa_chain,
        "What did they do there?",
        chat_history
    )