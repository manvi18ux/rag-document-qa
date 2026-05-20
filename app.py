import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
from dotenv import load_dotenv
from pdf_loader import load_pdfs
from web_loader import load_multiple_websites
from chunker import chunk_documents
from vectorstore import build_faiss_vectorstore, load_faiss_vectorstore
from qa_chain import build_qa_chain, ask_with_history
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ─── PAGE CONFIG ────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Document Q&A System")
st.caption("Upload a PDF or enter a URL and ask questions about it")

# ─── SESSION STATE INIT ─────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ─── SIDEBAR ────────────────────────────────────────────────

with st.sidebar:
    st.header("📁 Upload Your Documents")

    # PDF Upload
    st.subheader("Upload PDF")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    # URL Input
    st.subheader("Or Enter a URL")
    url_input = st.text_input("Enter website URL", placeholder="https://example.com")

    # Process button
    if st.button("🚀 Process Documents", type="primary"):
        if not uploaded_files and not url_input:
            st.error("Please upload a PDF or enter a URL first!")
        else:
            with st.spinner("Processing documents..."):
                all_docs = []

                # Handle PDF uploads
                if uploaded_files:
                    # Save uploaded files to data folder
                    import tempfile
                    os.makedirs("/tmp/uploaded", exist_ok=True)
                    for uploaded_file in uploaded_files:
                        file_path = f"/tmp/uploaded/{uploaded_file.name}"
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        st.write(f"📄 Saved: {uploaded_file.name}")

                    # Load saved PDFs
                    pdf_docs = load_pdfs("/tmp/uploaded")
                    all_docs.extend(pdf_docs)
                    if len(pdf_docs) == 0:
                        st.error("❌ Could not load PDF. Make sure it is not a scanned/image PDF.")
                        st.stop()
                    st.write(f"✅ Loaded {len(pdf_docs)} PDF pages")

                    # Delete files after loading to free up space
                    for uploaded_file in uploaded_files:
                        file_path = f"/tmp/uploaded/{uploaded_file.name}"
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    st.write("🗑️ Temporary files cleaned up")

                # Handle URL input
                if url_input:
                    web_docs = load_multiple_websites([url_input])
                    all_docs.extend(web_docs)
                    st.write(f"✅ Loaded {len(web_docs)} web pages")

                # Chunk documents
                chunks = chunk_documents(all_docs)
                st.write(f"✅ Created {len(chunks)} chunks")

                # Safety check
                if len(chunks) == 0:
                    st.error("❌ No text could be extracted from the document. Try a different PDF.")
                    st.stop()

                # Build vector store
                vectorstore = build_faiss_vectorstore(chunks)
                st.session_state.vectorstore = vectorstore
                st.write("✅ Vector store built")

                # Build QA chain
                qa_chain = build_qa_chain(vectorstore)
                st.session_state.qa_chain = qa_chain

                # Reset chat
                st.session_state.messages = []
                st.session_state.chat_history = []

            st.success("✅ Ready! Ask your questions below.")

    # Sidebar status
    st.divider()
    if st.session_state.vectorstore is not None:
        st.success("✅ Documents loaded and ready")
    else:
        st.warning("⚠️ No documents loaded yet")

    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# ─── CHAT INTERFACE ─────────────────────────────────────────

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📄 View Sources"):
                for i, source in enumerate(message["sources"]):
                    st.write(f"**Source {i+1}:** {source['file']} - Page {source['page']}")
                    st.caption(source["preview"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):

    # Check if documents are loaded
    if st.session_state.qa_chain is None:
        st.error("Please upload documents first using the sidebar!")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get AI answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result, st.session_state.chat_history = ask_with_history(
                st.session_state.qa_chain,
                prompt,
                st.session_state.chat_history
            )

            answer = result["answer"]
            st.write(answer)

            # Show sources
            sources = []
            if "context" in result:
                for doc in result["context"]:
                    sources.append({
                        "file": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", "N/A"),
                        "preview": doc.page_content[:200]
                    })

                if sources:
                    with st.expander("📄 View Sources"):
                        for i, source in enumerate(sources):
                            st.write(f"**Source {i+1}:** {source['file']} - Page {source['page']}")
                            st.caption(source["preview"])

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })