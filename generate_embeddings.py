import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the data
df = pd.read_json("data/News_Category_Dataset_v3.json", lines=True)

# Combine headline and short_description for embedding
texts = (df['headline'] + ". " + df['short_description']).tolist()

# Load a small, fast model
model = SentenceTransformer("all-MiniLM-L6-v2")  # ~80MB, fast and accurate

# Generate embeddings
embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

# Save embeddings and DataFrame index for later retrieval
np.save("embeddings/news_embeddings.npy", embeddings)
df.reset_index(drop=False).to_json("embeddings/news_df.json", orient="records", lines=True)