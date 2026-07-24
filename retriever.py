"""
Hybrid retrieval: BM25 (sparse/leksik) + dense embeddings (semantik) qidiruvni
Reciprocal Rank Fusion (RRF) orqali birlashtiradi.

Nega hybrid?
- BM25 aniq atamalar, modda raqamlari, tushunchalarni topishda kuchli
  (masalan foydalanuvchi "18-modda nima haqida" desa).
- Dense embeddings ma'no bo'yicha o'xshashlikni topadi (masalan foydalanuvchi
  "davlat tili haqida qonun bormi" desa-yu, matnda "davlat tili" so'zi
  boshqacharoq shaklda kelsa ham topa oladi).
- RRF ikkalasini oddiy va barqaror tarzda birlashtirish uchun standart usul
  (og'irliklarni qo'lda sozlash shart emas, faqat reyting o'rinlari asosida ishlaydi).
"""
import re
import pickle
import numpy as np
from rank_bm25 import BM25Okapi

import config
from ingest import Document

# Oddiy, kutubxonasiz o'zbek tili uchun tokenizer.
# Lotin alifbosidagi maxsus belgilarni ('‘', '’', 'ʻ' kabi apostrof turlari) hisobga oladi.
_TOKEN_RE = re.compile(r"[a-zA-Zʻʼ‘’'`]+|\d+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = text.replace("‘", "'").replace("’", "'").replace("ʻ", "'").replace("ʼ", "'")
    return _TOKEN_RE.findall(text)


class BM25Retriever:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.tokenized_corpus = [tokenize(d.search_text) for d in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Qaytaradi: [(doc_id, bm25_score), ...] ball bo'yicha kamayish tartibida."""
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in ranked_idx]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "BM25Retriever":
        with open(path, "rb") as f:
            return pickle.load(f)


class DenseRetriever:
    """
    sentence-transformers orqali multilingual embedding modelidan foydalanadi.
    Model faqat shu klass ishlatilganda import qilinadi (lazy import) — shunda
    embeddinglar oldindan hisoblab index qilingan bo'lsa, servis ishga tushishda
    torch/sentence-transformers'ni majburiy yuklamaydi... lekin qidiruv vaqtida
    query'ni embed qilish uchun baribir kerak bo'ladi.
    """

    def __init__(self, documents: list[Document] = None, embeddings: np.ndarray = None):
        self.documents = documents
        self.embeddings = embeddings  # shape: (n_docs, dim), L2-normalized
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(config.EMBEDDING_MODEL)
        return self._model

    def build_index(self, documents: list[Document]):
        model = self._load_model()
        texts = [config.E5_PASSAGE_PREFIX + d.search_text for d in documents]
        embeddings = model.encode(
            texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True
        )
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.documents = documents
        self.embeddings = embeddings.astype(np.float32)

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        model = self._load_model()
        q_emb = model.encode(
            [config.E5_QUERY_PREFIX + query], convert_to_numpy=True
        )[0]
        q_emb = q_emb / np.linalg.norm(q_emb)
        sims = self.embeddings @ q_emb  # cosine similarity (chunki normalized)
        ranked_idx = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in ranked_idx]

    def save(self, path: str):
        np.save(path, self.embeddings)

    def load_embeddings(self, path: str, documents: list[Document]):
        self.embeddings = np.load(path)
        self.documents = documents


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]], k: int = 60
) -> list[tuple[int, float]]:
    """
    Bir nechta reytinglar ro'yxatini (masalan BM25 va dense) birlashtiradi.
    Har bir ro'yxat: [(doc_id, score), ...] — score'lar reytingga aylantiriladi,
    haqiqiy score qiymatlari (masalan BM25 vs cosine) taqqoslanmaydi, faqat
    o'rinlar ishlatiladi. Bu ikki turdagi score shkalasini normalizatsiya
    qilish muammosini butunlay chetlab o'tadi.

    RRF formula: score(d) = sum over lists of 1 / (k + rank(d))
    """
    fused_scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


_MODDA_NUMBER_RE = re.compile(r"(\d{1,3}(?:-\d{1,2})?)\s*[- ]?\s*modda", re.IGNORECASE)

# Foydalanuvchi qaysi qonunni nazarda tutayotganini aniqlash uchun kalit so'zlar.
# Buyurtma muhim emas — birinchi topilgan mos qonun_kodi ishlatiladi.
_LAW_HINTS = [
    (re.compile(r"jinoyat\s*kodeks|jk\b", re.IGNORECASE), "JK"),
    (re.compile(r"konstitutsiy", re.IGNORECASE), "KONST"),
]


class HybridRetriever:
    def __init__(self, documents: list[Document], bm25: BM25Retriever, dense: DenseRetriever):
        self.documents = documents
        self.bm25 = bm25
        self.dense = dense
        # (qonun_kodi, modda_raqami) -> doc_id tez qidiruv uchun. Modda raqamlari
        # turli qonunlarda takrorlanishi mumkin (masalan har ikkalasida ham "1-modda"
        # bor), shuning uchun qonun kodi bilan birga kalitlanadi.
        self._modda_to_doc_id: dict[tuple[str, str], int] = {
            (d.qonun_kodi, d.modda_raqami): d.doc_id for d in documents
        }

    def _detect_law_hint(self, query: str) -> str | None:
        for pattern, qonun_kodi in _LAW_HINTS:
            if pattern.search(query):
                return qonun_kodi
        return None

    def _direct_modda_lookup(self, query: str) -> list[int]:
        """
        Agar foydalanuvchi savolida aniq modda raqami ko'rsatilgan bo'lsa
        (masalan "18-modda nima haqida?"), o'sha moddani to'g'ridan-to'g'ri
        topib, kafolatlangan tarzda kontekstga qo'shadi. Semantik/leksik
        qidiruv bunday aniq ID so'rovlarida ba'zan sarlavhadan chalg'ib
        ketishi mumkin, shu sabab bu qat'iy fallback zarur.

        Modda raqamlari ikkala qonunda ham takrorlanishi mumkinligi sababli:
        - Agar savolda qonun nomi ko'rsatilgan bo'lsa (masalan "Jinoyat kodeksining
          18-moddasi"), faqat o'sha qonundan moddani qaytaradi.
        - Aks holda, mos keladigan barcha qonunlardagi moddalarni qaytaradi (LLM'ga
          ikkalasini ham ko'rib, kerakli birini tanlash imkoni beriladi).
        """
        law_hint = self._detect_law_hint(query)
        found = []
        for match in _MODDA_NUMBER_RE.finditer(query):
            num = match.group(1)
            if law_hint:
                doc_id = self._modda_to_doc_id.get((law_hint, num))
                if doc_id is not None:
                    found.append(doc_id)
            else:
                for (qonun_kodi, modda_raqami), doc_id in self._modda_to_doc_id.items():
                    if modda_raqami == num:
                        found.append(doc_id)
        return found

    def retrieve(self, query: str, top_k: int = None) -> list[Document]:
        top_k = top_k or config.TOP_K_FINAL
        bm25_results = self.bm25.search(query, config.TOP_K_BM25)
        dense_results = self.dense.search(query, config.TOP_K_DENSE)
        fused = reciprocal_rank_fusion([bm25_results, dense_results], k=config.RRF_K)

        direct_ids = self._direct_modda_lookup(query)
        ordered_ids = direct_ids + [doc_id for doc_id, _ in fused if doc_id not in direct_ids]
        top_doc_ids = ordered_ids[:top_k]
        return [self.documents[i] for i in top_doc_ids]

    def retrieve_with_debug(self, query: str, top_k: int = None):
        """Diagnostika uchun: har bir usul qanday natija berganini ko'rsatadi."""
        top_k = top_k or config.TOP_K_FINAL
        bm25_results = self.bm25.search(query, config.TOP_K_BM25)
        dense_results = self.dense.search(query, config.TOP_K_DENSE)
        fused = reciprocal_rank_fusion([bm25_results, dense_results], k=config.RRF_K)
        debug = {
            "bm25_top": [(self.documents[i].modda_raqami, round(s, 3)) for i, s in bm25_results[:5]],
            "dense_top": [(self.documents[i].modda_raqami, round(s, 3)) for i, s in dense_results[:5]],
            "fused_top": [(self.documents[i].modda_raqami, round(s, 4)) for i, s in fused[:top_k]],
        }
        top_doc_ids = [doc_id for doc_id, _ in fused[:top_k]]
        return [self.documents[i] for i in top_doc_ids], debug
