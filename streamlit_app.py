import os

# Download embeddings if missing or zero-byte
if (
    not os.path.exists("embeddings/news_embeddings.npy")
    or not os.path.exists("embeddings/news_df.json")
    or os.path.getsize("embeddings/news_embeddings.npy") == 0
    or os.path.getsize("embeddings/news_df.json") == 0
):
    import download_embeddings

from retriever import retrieve_articles
from generator import generate_with_ollama
import streamlit as st

st.set_page_config(page_title="News RAG Assistant", layout="wide")

st.title("📰 News RAG Assistant (Local LLM Demo)")
st.markdown("""
This assistant answers your questions about news (2012–2022) using retrieval-augmented generation with a local Llama 3 model.<br>
<span style='color:green'>No cloud APIs. All processing runs on your machine!</span>
""", unsafe_allow_html=True)

question = st.text_input("Ask a question about the news:", "")

if question:
    with st.spinner("Retrieving articles and generating answer..."):
        articles = retrieve_articles(question, top_k=5)
        context = ""
        for _, row in articles.iterrows():
            context += f"Title: {row['headline']}\nSummary: {row['short_description']}\nDate: {row['date']}\n---\n"
        prompt = (
            f"You are a helpful assistant. Use the following news articles to answer the question.\n\n"
            f"Context:\n{context}\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        answer = generate_with_ollama(prompt)
    st.subheader("Answer")
    st.write(answer.strip())

    st.markdown("---")
    st.subheader("Retrieved News Articles")
    for _, row in articles.iterrows():
        st.markdown(f"**{row['headline']}** ({row['date']})  \n{row['short_description']}\n")

st.markdown("----\n*Powered by [Ollama](https://ollama.com/), [Sentence Transformers](https://www.sbert.net/), and [FAISS](https://faiss.ai/)*")