import os
import requests

def download_file(url, dest_path):
    """Download a file from a URL if it doesn't exist or is empty."""
    if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
        print(f"Downloading {dest_path} from {url} ...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded {dest_path}.")

os.makedirs("embeddings", exist_ok=True)

EMBEDDINGS_URL = "https://huggingface.co/datasets/ssanidhya0407/newsviallm-embeddings/resolve/main/news_embeddings.npy"
DF_URL = "https://huggingface.co/datasets/ssanidhya0407/newsviallm-embeddings/resolve/main/news_df.json"


download_file(EMBEDDINGS_URL, "embeddings/news_embeddings.npy")
download_file(DF_URL, "embeddings/news_df.json")