import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"  # or your model's name

def generate_with_ollama(prompt, model=MODEL):
    """Generates a response using a local LLM with Ollama.

    Args:
        prompt (str): The full prompt to send to the language model.
        model (str, optional): The name of the Ollama model to use.
            Defaults to the `MODEL` constant.

    Returns:
        str: The generated text from the model, or an empty string if
            the request fails.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")
    except Exception as e:
        print("Ollama request failed:", e)
        return ""