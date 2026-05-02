from langchain_community.document_loaders import WebBaseLoader

def load_website(url):
    print(f"🌐 Loading website: {url}")

    try:
        loader = WebBaseLoader(url)
        documents = loader.load()

        print(f"✅ Loaded {len(documents)} document(s) from URL")
        return documents

    except Exception as e:
        print(f"❌ Failed to load URL: {e}")
        return []


def load_multiple_websites(urls):
    all_documents = []

    for url in urls:
        docs = load_website(url)
        all_documents.extend(docs)

    print(f"\n✅ Total documents loaded from all URLs: {len(all_documents)}")
    return all_documents


# Test it directly
if __name__ == "__main__":
    test_urls = [
        "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
    ]

    docs = load_multiple_websites(test_urls)

    if docs:
        print("\n--- Preview of first document ---")
        print("Text preview:", docs[0].page_content[:300])
        print("Metadata:", docs[0].metadata)