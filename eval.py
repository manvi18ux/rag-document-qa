import sys
import os
import time
os.environ["USER_AGENT"] = "rag-project/1.0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from vectorstore import load_faiss_vectorstore
from qa_chain import build_qa_chain

load_dotenv()


# ─── STEP 1: Configure RAGAS ────────────────────────────────

def get_ragas_llm():
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )
    return LangchainLLMWrapper(llm)

def get_ragas_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return LangchainEmbeddingsWrapper(embeddings)


# ─── STEP 2: Test Dataset ───────────────────────────────────

test_data = [
    {
        "question": "What programming languages does this person know?",
        "ground_truth": "Python, Java, C, HTML/CSS, JavaScript, SQL"
    },
    {
        "question": "What is this person's educational background?",
        "ground_truth": "Bachelor of Science in Computer Science from State University, September 2017 to May 2021"
    },
    {
        "question": "What companies did this person intern at?",
        "ground_truth": "Electronics Company and Startup Inc"
    }
]


# ─── STEP 3: Run RAG Pipeline ───────────────────────────────

def run_pipeline_on_test_data(qa_chain, test_data):
    print("🔄 Running RAG pipeline on test questions...")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in test_data:
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"\n🤔 Question: {question}")

        result = qa_chain.invoke({"input": question})
        answer = result["answer"]
        context_docs = result["context"]
        context_texts = [doc.page_content for doc in context_docs]

        print(f"💬 Answer: {answer[:150]}...")
        print(f"📄 Contexts retrieved: {len(context_texts)}")

        questions.append(question)
        answers.append(answer)
        contexts.append(context_texts)
        ground_truths.append(ground_truth)

        # Small delay between questions to avoid rate limits
        time.sleep(2)

    return questions, answers, contexts, ground_truths


# ─── STEP 4: Evaluate each metric separately ────────────────

def evaluate_single_metric(dataset, metric, ragas_llm, ragas_embeddings):
    metric.llm = ragas_llm
    if hasattr(metric, 'embeddings'):
        metric.embeddings = ragas_embeddings

    try:
        result = evaluate(
            dataset,
            metrics=[metric],
            raise_exceptions=False
        )
        return result
    except Exception as e:
        print(f"⚠️ Error evaluating {metric.name}: {e}")
        return None


def evaluate_rag(questions, answers, contexts, ground_truths):
    print("\n🔄 Running RAGAS evaluation...")
    print("Evaluating each metric separately to avoid rate limits...\n")

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    ragas_llm = get_ragas_llm()
    ragas_embeddings = get_ragas_embeddings()

    all_scores = {}

    # Evaluate faithfulness
    print("📊 Evaluating Faithfulness...")
    result = evaluate_single_metric(dataset, faithfulness, ragas_llm, ragas_embeddings)
    if result and "faithfulness" in result:
        all_scores["faithfulness"] = result["faithfulness"]
        print(f"✅ Faithfulness done: {result['faithfulness']:.4f}")
    time.sleep(10)

    # Evaluate answer relevancy
    print("\n📊 Evaluating Answer Relevancy...")
    result = evaluate_single_metric(dataset, answer_relevancy, ragas_llm, ragas_embeddings)
    if result and "answer_relevancy" in result:
        all_scores["answer_relevancy"] = result["answer_relevancy"]
        print(f"✅ Answer Relevancy done: {result['answer_relevancy']:.4f}")
    time.sleep(10)

    # Evaluate context recall
    print("\n📊 Evaluating Context Recall...")
    result = evaluate_single_metric(dataset, context_recall, ragas_llm, ragas_embeddings)
    if result and "context_recall" in result:
        all_scores["context_recall"] = result["context_recall"]
        print(f"✅ Context Recall done: {result['context_recall']:.4f}")
    time.sleep(10)

    # Evaluate context precision
    print("\n📊 Evaluating Context Precision...")
    result = evaluate_single_metric(dataset, context_precision, ragas_llm, ragas_embeddings)
    if result and "context_precision" in result:
        all_scores["context_precision"] = result["context_precision"]
        print(f"✅ Context Precision done: {result['context_precision']:.4f}")

    return all_scores


# ─── STEP 5: Print Results ──────────────────────────────────

def print_results(scores):
    print("\n" + "="*50)
    print("📊 RAGAS EVALUATION RESULTS")
    print("="*50)

    display_names = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision"
    }

    valid_scores = []

    for key, display in display_names.items():
        if key in scores:
            score = scores[key]
            import math
            if math.isnan(score):
                print(f"⚠️  {display}: timed out")
            else:
                valid_scores.append(score)
                if score >= 0.8:
                    emoji = "✅"
                elif score >= 0.6:
                    emoji = "⚠️"
                else:
                    emoji = "❌"
                print(f"{emoji} {display}: {score:.4f}")
        else:
            print(f"⚠️  {display}: not evaluated")

    print("\n" + "="*50)
    if valid_scores:
        avg = sum(valid_scores) / len(valid_scores)
        print(f"📈 Average Score: {avg:.4f}")
    print("="*50)

    print("\n📝 Score Guide:")
    print("✅ >= 0.8 : Excellent")
    print("⚠️  >= 0.6 : Good, room for improvement")
    print("❌  < 0.6  : Needs fixing")

    return scores


# ─── MAIN ───────────────────────────────────────────────────

if __name__ == "__main__":
    from pdf_loader import load_pdfs
    from chunker import chunk_documents
    from vectorstore import build_faiss_vectorstore

    if not os.path.exists("vectorstore/faiss_index"):
        print("=== Building Vector Store from scratch ===")
        docs = load_pdfs("data")
        if len(docs) == 0:
            print("❌ No PDFs found in data/ folder!")
            print("Please copy your resume PDF into the data/ folder and run again.")
            exit()
        chunks = chunk_documents(docs)
        vectorstore = build_faiss_vectorstore(chunks)
        print("✅ Vector store built!")
    else:
        print("=== Loading existing Vector Store ===")
        vectorstore = load_faiss_vectorstore()

    qa_chain = build_qa_chain(vectorstore)

    questions, answers, contexts, ground_truths = run_pipeline_on_test_data(
        qa_chain,
        test_data
    )

    scores = evaluate_rag(questions, answers, contexts, ground_truths)
    print_results(scores)

    print("\n✅ Evaluation complete!")
    print("Add these scores to your README and resume!")