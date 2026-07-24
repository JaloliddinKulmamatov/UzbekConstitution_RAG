"""
Retrieval + prompt qurish + Ollama orqali generatsiyani birlashtiruvchi asosiy modul.
"""
import pickle

import config
import ollama_client
from retriever import BM25Retriever, DenseRetriever, HybridRetriever


def load_hybrid_retriever() -> HybridRetriever:
    with open(config.DOCS_PATH, "rb") as f:
        documents = pickle.load(f)

    bm25 = BM25Retriever.load(config.BM25_INDEX_PATH)

    dense = DenseRetriever()
    dense.load_embeddings(config.DENSE_INDEX_PATH, documents)

    return HybridRetriever(documents, bm25, dense)


def build_prompt(query: str, contexts: list) -> str:
    context_block = "\n\n".join(doc.to_context_string() for doc in contexts)
    return (
        f"KONTEKST (Konstitutsiya moddalaridan):\n{context_block}\n\n"
        f"SAVOL: {query}\n\n"
        f"Yuqoridagi kontekst asosida savolga javob bering. "
        f"Foydalanilgan har bir moddani (masalan: 18-modda) ko'rsating."
    )


class RagAnswer:
    def __init__(self, query, contexts, prompt, answer):
        self.query = query
        self.contexts = contexts
        self.prompt = prompt
        self.answer = answer


def answer_query(
    hybrid_retriever: HybridRetriever, query: str, top_k: int = None, stream: bool = False
):
    contexts = hybrid_retriever.retrieve(query, top_k=top_k)
    prompt = build_prompt(query, contexts)

    if stream:
        return contexts, ollama_client.generate(prompt, system=config.SYSTEM_PROMPT, stream=True)

    answer_text = ollama_client.generate(prompt, system=config.SYSTEM_PROMPT, stream=False)
    return RagAnswer(query, contexts, prompt, answer_text)
