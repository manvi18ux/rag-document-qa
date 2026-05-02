# RAG Document Q&A System

An AI-powered system that lets you upload PDFs or websites and ask questions about them.

## Tech Stack
- LangChain + LlamaIndex
- FAISS + ChromaDB
- Groq (LLM - free and fast)
- Streamlit + FastAPI

## Setup
1. Clone the repo
2. Run `pip install -r requirements.txt`
3. Go to console.groq.com and get a free API key
4. Create a `.env` file and add `GROQ_API_KEY=your_key_here`
5. Run `streamlit run app.py`

## Evaluation Results(RAGAS)
==================================================
📊 RAGAS EVALUATION RESULTS
==================================================
✅ Faithfulness: 0.9815
⚠️  Answer Relevancy: timed out
✅ Context Recall: 0.8333
⚠️ Context Precision: 0.7470

==================================================
📈 Average Score: 0.8539
==================================================