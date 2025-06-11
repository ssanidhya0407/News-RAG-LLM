# 📰 News RAG Assistant (Local LLM Demo)

This project demonstrates a fully local **Retrieval-Augmented Generation (RAG)** pipeline, powered by a news dataset, semantic search, and a local Llama 3 model running via [Ollama](https://ollama.com/). The app answers news-related questions using only your own computer—**no cloud APIs or third-party data processing**.

---

## 🚀 Features

- **Semantic search** over 200k+ news articles using [FAISS](https://faiss.ai/) and [Sentence Transformers](https://www.sbert.net/)
- **Local LLM generation** with [Llama 3](https://ollama.com/library/llama3) via [Ollama](https://ollama.com/)
- **Interactive web UI** built with [Streamlit](https://streamlit.io/)
- **Extensible:** Fine-tune, swap retrievers, or connect to other local models

---

## 🛠️ Setup Instructions

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/ssanidhya0407/News-RAG-LLM
cd News-RAG-LLM
pip install -r requirements.txt
```

### 2. Prepare Your Data

- Place your news dataset (e.g. `news_dataset.json`) in the `data/` folder.
- Generate embeddings (run this once):
    ```bash
    python generate_embeddings.py
    ```

### 3. Run Ollama with Llama 3

Make sure [Ollama](https://ollama.com/) is installed and run:

```bash
ollama run llama3
```

Leave this terminal open.

### 4. Launch the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser.

---

## 🖥️ Usage

- Enter a news-related question (e.g., "What were the major developments in COVID-19 vaccines in 2021?")
- The app retrieves relevant articles and generates a summary answer using local LLM
- Retrieved articles are shown for transparency and grounding

---

## 🗂️ Project Structure

```
├── app.py                # CLI entrypoint (optional)
├── streamlit_app.py      # Streamlit web app
├── retriever.py          # Semantic retriever with FAISS
├── generator.py          # Ollama LLM interface
├── generate_embeddings.py# Embedding generator script
├── requirements.txt
├── .gitignore
├── README.md
├── data/
│   └── news_dataset.json # (not tracked by git)
├── embeddings/
│   ├── news_embeddings.npy
│   └── news_df.json
```

---

## 🧩 Extending This Project

- **Fine-tune Llama 3** with custom data using Ollama's tools
- Swap in other local models (e.g., Mistral, Gemma) by changing the model name
- Expand the UI or add more retrieval strategies

---

## 🙏 Credits

- [Ollama](https://ollama.com/)
- [Sentence-Transformers](https://www.sbert.net/)
- [FAISS](https://faiss.ai/)
- [Streamlit](https://streamlit.io/)
