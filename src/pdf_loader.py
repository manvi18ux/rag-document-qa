import os
import re
from langchain_community.document_loaders import PyPDFLoader

def load_pdfs(data_folder="data"):
    all_documents = []

    pdf_files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]

    if not pdf_files:
        print("❌ No PDF files found in the data folder.")
        return []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_folder, pdf_file)
        print(f"📄 Loading: {pdf_file}")

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Clean extra spaces from each document
        for doc in documents:
            # Replace multiple spaces with single space
            doc.page_content = re.sub(r'  +', ' ', doc.page_content)
            # Replace multiple newlines with single newline
            doc.page_content = re.sub(r'\n+', '\n', doc.page_content)
            #Fix spaces before punctuation
            doc.page_content = re.sub(r' ([.,!?])', r'\1', doc.page_content)
            # Strip leading/trailing whitespace
            doc.page_content = doc.page_content.strip()

        print(f"   ✅ Loaded {len(documents)} pages")
        all_documents.extend(documents)

    print(f"\n✅ Total pages loaded from all PDFs: {len(all_documents)}")
    return all_documents


if __name__ == "__main__":
    docs = load_pdfs()
    if docs:
        print("\n--- Preview of first page ---")
        print("Text preview:", docs[0].page_content[:300])
        print("Metadata:", docs[0].metadata)