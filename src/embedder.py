from sympy import And

from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    print("🔄 Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("✅ Embedding model loaded!")
    return embeddings


# Test it directly
if __name__ == "__main__":
    embeddings = get_embeddings()

    # Test embedding a sentence
    test_sentence = "What is retrieval augmented generation?"
    vector = embeddings.embed_query(test_sentence)

    print(f"\n--- Embedding Test ---")
    print(f"Sentence: {test_sentence}")
    print(f"Vector length: {len(vector)}")
    print(f"First 5 numbers: {vector[:5]}")

#It loaded a free AI model called all-MiniLM-L6-v2 from HuggingFace onto your computer.
#This model has one job — take any text and convert it into a list of 384 numbers called a vector.
#For example when you ran the test it took this sentence:
#"What is retrieval augmented generation?"
#And converted it into this:
#[0.123, -0.045, 0.891, -0.234, ...]  ← 384 numbers total
#That's it. Just text in, numbers out. Those numbers capture the meaning of the text mathematically.
