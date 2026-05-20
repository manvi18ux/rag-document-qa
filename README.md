---
title: RAG Document QA
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# RAG Document Q&A System

An AI-powered system that lets you upload PDFs or websites and ask questions about them.

## Tech Stack
- LangChain + LlamaIndex
- FAISS + ChromaDB
- Groq LLM (Llama 3.1)
- Streamlit + FastAPI

## Features
- Upload PDFs and ask questions
- Ingest websites by URL
- Multi-turn conversation with memory
- Source citations for every answer
- Advanced RAG techniques (MMR, MultiQuery, HyDE)

## Evaluation Results (RAGAS)
| Metric | Score |
|---|---|
| Faithfulness | 0.9815 |
| Context Recall | 0.8333 |
| Context Precision | 0.7470 |
| Average | 0.8539 |

## Setup
1. Clone the repo
2. Run `pip install -r requirements.txt`
3. Add `GROQ_API_KEY` to `.env`
4. Run `streamlit run app.py`