from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf_loader import load_pdfs
from web_loader import load_multiple_websites

def chunk_documents(documents, chunk_size=500, chunk_overlap=100):
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    chunks = splitter.split_documents(documents)

    print(f"✅ Total chunks created: {len(chunks)}")
    return chunks


# Test it directly
if __name__ == "__main__":

    # Test with PDF
    print("--- Chunking PDF ---")
    pdf_docs = load_pdfs()
    pdf_chunks = chunk_documents(pdf_docs)

    print(f"\n--- Preview of first chunk ---")
    print("Text:", pdf_chunks[0].page_content)
    print("Metadata:", pdf_chunks[0].metadata)

    print("\n--- Preview of second chunk ---")
    print("Text:", pdf_chunks[1].page_content)

    # Test with website
    print("\n--- Chunking Website ---")
    web_docs = load_multiple_websites([
        "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
    ])
    web_chunks = chunk_documents(web_docs)

    print(f"\n--- Preview of first web chunk ---")
    print("Text:", web_chunks[0].page_content[:300])