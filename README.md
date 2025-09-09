# 📰 News RAG Assistant (Local LLM Demo)

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-brightgreen?logo=streamlit)](https://newsviallm.streamlit.app)

Try the hosted Streamlit demo here: [https://newsviallm.streamlit.app](https://newsviallm.streamlit.app)

This project demonstrates a fully local **Retrieval-Augmented Generation (RAG)** pipeline. It uses a news dataset, semantic search, and a local Llama 3 model (via [Ollama](https://ollama.com/)) to answer questions. The entire process runs on your machine—**no cloud APIs or third-party data processing required**.

---

## ⚙️ How It Works

This RAG pipeline follows a simple, powerful workflow:

1.  **User Query**: You ask a question (e.g., "What's the latest on AI regulation?").
2.  **Retrieve**: The system searches a database of over 200,000 news articles to find the most relevant ones. This is done using **semantic search**, which understands the *meaning* of your query, not just keywords.
3.  **Augment**: The retrieved articles are combined with your original question into a detailed prompt.
4.  **Generate**: The prompt is fed to a local Large Language Model (Llama 3), which generates a coherent answer based on the provided context.

This approach allows the LLM to answer questions using up-to-date, relevant information that it wasn't originally trained on.

---

## 🚀 Features

- **Semantic search** over 200k+ news articles using [FAISS](https://faiss.ai/) and [Sentence Transformers](https://www.sbert.net/).
- **Local LLM generation** with [Llama 3](https://ollama.com/library/llama3) via [Ollama](https://ollama.com/).
- **Interactive web UI** built with [Streamlit](https://streamlit.io/) for easy use.
- **CLI interface** for command-line power users.
- **Extensible:** Easily swap models, datasets, or retrieval methods.

---

## 🛠️ Setup Instructions

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/ssanidhya0407/News-RAG-LLM
cd News-RAG-LLM
pip install -r requirements.txt
```

### 2. Get Embeddings (Automatic)

The first time you run the app, it will automatically download the pre-computed news embeddings (~500MB). No manual steps are needed.

**Optional: Generate Embeddings Manually**
If you want to use a different dataset or embedding model:
- Place your dataset (e.g., `News_Category_Dataset_v3.json`) in the `data/` folder.
- Run the generation script (this can take time and requires a powerful GPU):
    ```bash
    python generate_embeddings.py
    ```

### 3. Run Ollama with Llama 3

Make sure [Ollama](https://ollama.com/) is installed and running. Pull the Llama 3 model:

```bash
ollama run llama3
```

Leave this terminal open so the model stays loaded in memory.

### 4. Launch the App

You can run either the Streamlit web app or the command-line interface.

**Streamlit Web App:**
```bash
streamlit run streamlit_app.py
```

**Command-Line App:**
```bash
python app.py
```

---

## 🖥️ Usage

- **Streamlit App**: Enter a news-related question (e.g., "What were the major developments in COVID-19 vaccines in 2021?"). The app retrieves relevant articles, generates a summary, and displays both for transparency.
- **CLI App**: The script will prompt you for questions in a loop. Type `exit` to quit.

---

## 🗂️ Project Structure

```
├── app.py                # Main entrypoint for the command-line interface (CLI).
├── streamlit_app.py      # The Streamlit web application.
├── retriever.py          # Handles semantic search and article retrieval using FAISS.
├── generator.py          # Interfaces with the Ollama API to generate text with a local LLM.
├── generate_embeddings.py# Script to create sentence embeddings from the raw news dataset.
├── download_embeddings.py# Helper script to download pre-computed embeddings from Hugging Face.
├── requirements.txt      # Python dependencies for the project.
├── .gitignore
├── README.md
├── data/
│   └── (Contains the raw JSON dataset, not tracked by git)
├── embeddings/
│   ├── news_embeddings.npy # Stores the NumPy array of sentence embeddings.
│   └── news_df.json        # A JSON file of the news data, indexed to match the embeddings.
```

---

## 🧩 Extending This Project

- **Fine-tune Llama 3** with custom data using Ollama's `Modelfile`.
- **Swap in other local models** (e.g., Mistral, Gemma) by changing the `MODEL` variable in `generator.py`.
- **Expand the UI** or add new retrieval strategies in `retriever.py`.

---

## 🤝 Credits

- [Ollama](https://ollama.com/)
- [Sentence-Transformers](https://www.sbert.net/)
- [FAISS](https://faiss.ai/)
- [Streamlit](https://streamlit.io/)
