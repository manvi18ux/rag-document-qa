import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
import shutil

from pdf_loader import load_pdfs
from web_loader import load_multiple_websites
from chunker import chunk_documents
from vectorstore import build_faiss_vectorstore
from qa_chain import build_qa_chain, ask_with_history
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ─── APP SETUP ──────────────────────────────────────────────

app = FastAPI(
    title="RAG Document Q&A API",
    description="Upload PDFs or URLs and ask questions about them",
    version="1.0.0"
)

# Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ─── IN-MEMORY STATE ────────────────────────────────────────

# Store vectorstore and qa_chain in memory
# In production this would be a proper database
app_state = {
    "vectorstore": None,
    "qa_chain": None,
    "chat_history": []
}

# ─── REQUEST/RESPONSE MODELS ────────────────────────────────

class URLRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list

# ─── ENDPOINTS ──────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "documents_loaded": app_state["vectorstore"] is not None
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Check file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    try:
        # Save file temporarily
        os.makedirs("data/uploaded", exist_ok=True)
        file_path = f"data/uploaded/{file.filename}"

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Load and process
        docs = load_pdfs("data/uploaded")
        chunks = chunk_documents(docs)

        if len(chunks) == 0:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        # Build vectorstore and chain
        vectorstore = build_faiss_vectorstore(chunks)
        app_state["vectorstore"] = vectorstore
        app_state["qa_chain"] = build_qa_chain(vectorstore)
        app_state["chat_history"] = []

        # Delete temp file
        os.remove(file_path)

        return {
            "message": "PDF processed successfully",
            "filename": file.filename,
            "chunks_created": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest-url")
async def ingest_url(request: URLRequest):
    try:
        # Load from URL
        docs = load_multiple_websites([request.url])

        if len(docs) == 0:
            raise HTTPException(status_code=400, detail="Could not load URL")

        chunks = chunk_documents(docs)

        # Build vectorstore and chain
        vectorstore = build_faiss_vectorstore(chunks)
        app_state["vectorstore"] = vectorstore
        app_state["qa_chain"] = build_qa_chain(vectorstore)
        app_state["chat_history"] = []

        return {
            "message": "URL processed successfully",
            "url": request.url,
            "chunks_created": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    # Check if documents loaded
    if app_state["qa_chain"] is None:
        raise HTTPException(
            status_code=400,
            detail="No documents loaded. Upload a PDF or ingest a URL first."
        )

    try:
        result, app_state["chat_history"] = ask_with_history(
            app_state["qa_chain"],
            request.question,
            app_state["chat_history"]
        )

        # Extract sources
        sources = []
        if "context" in result:
            for doc in result["context"]:
                sources.append({
                    "file": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "N/A"),
                    "preview": doc.page_content[:200]
                })

        return QueryResponse(
            answer=result["answer"],
            sources=sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear")
def clear_chat():
    app_state["chat_history"] = []
    return {"message": "Chat history cleared"}


# ─── RUN ────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)