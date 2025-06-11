import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# Load DataFrame
df = pd.read_json("embeddings/news_df.json", lines=True)

# Load embeddings
embeddings = np.load("embeddings/news_embeddings.npy")

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Load the same model for queries
model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_articles(query, top_k=5):
    """
    Given a query string, return top_k most relevant articles as a DataFrame.
    """
    query_emb = model.encode([query])
    distances, indices = index.search(np.array(query_emb), top_k)
    results = df.iloc[indices[0]].copy()
    results["distance"] = distances[0]
    return results

if __name__ == "__main__":
    # Example usage:
    query = "COVID-19 vaccines in 2022"
    top_articles = retrieve_articles(query)
    for i, row in top_articles.iterrows():
        print(f"{row['headline']} ({row['date']})\n{row['short_description']}\n---")