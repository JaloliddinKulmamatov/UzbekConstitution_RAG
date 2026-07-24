"""
Mahalliy Ollama server (http://localhost:11434) bilan ishlash uchun yupqa client.
Ollama'ni alohida o'rnatib, ishga tushirib qo'yish kerak:

    ollama pull llama3.2:3b
    ollama serve   (odatda avtomatik fonda ishlab turadi)
"""
import json
import requests

import config


class OllamaError(RuntimeError):
    pass


def check_ollama_available() -> bool:
    try:
        r = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=3)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def generate(prompt: str, system: str = None, stream: bool = False):
    """
    Ollama /api/chat endpointiga so'rov yuboradi.
    stream=False bo'lsa, to'liq javobni bitta string sifatida qaytaradi.
    stream=True bo'lsa, generator qaytaradi (matn bo'laklari, token-token).
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": config.OLLAMA_TEMPERATURE,
            "num_ctx": config.OLLAMA_NUM_CTX,
        },
    }

    try:
        resp = requests.post(
            f"{config.OLLAMA_HOST}/api/chat",
            json=payload,
            stream=stream,
            timeout=120,
        )
    except requests.exceptions.ConnectionError as e:
        raise OllamaError(
            f"Ollama serverga ulanib bo'lmadi ({config.OLLAMA_HOST}). "
            f"'ollama serve' ishlab turganini va '{config.OLLAMA_MODEL}' "
            f"modeli 'ollama pull {config.OLLAMA_MODEL}' bilan yuklanganini tekshiring."
        ) from e

    if resp.status_code != 200:
        raise OllamaError(f"Ollama xatosi ({resp.status_code}): {resp.text}")

    if not stream:
        data = resp.json()
        return data["message"]["content"]

    def _stream_gen():
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]
            if chunk.get("done"):
                break

    return _stream_gen()
