from retriever import retrieve_articles
from generator import generate_with_ollama

def build_context(articles):
    context = ""
    for _, row in articles.iterrows():
        context += f"Title: {row['headline']}\nSummary: {row['short_description']}\nDate: {row['date']}\n---\n"
    return context

def main():
    print("=== News RAG Assistant (Local LLM) ===")
    while True:
        question = input("\nAsk a question (or type 'exit'): ")
        if question.lower() == "exit":
            break

        articles = retrieve_articles(question, top_k=5)
        context = build_context(articles)
        prompt = (
            f"You are a helpful assistant. Use the following news articles to answer the question.\n\n"
            f"Context:\n{context}\n"
            f"Question: {question}\n"
            f"Answer:"
        )

        print("\nGenerating answer using local LLM (this may take a few seconds)...")
        answer = generate_with_ollama(prompt)
        print("\n--- Answer ---\n", answer.strip())

if __name__ == "__main__":
    main()