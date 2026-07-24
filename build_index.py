"""
Indekslarni bir marta quring, keyin har safar chat ilovasi ularni diskdan
tez yuklaydi (har safar embedding hisoblashning hojati yo'q).

Ishlatish:
    python build_index.py
"""
import pickle
import time

import config
from ingest import load_documents
from retriever import BM25Retriever, DenseRetriever


def main():
    print("1/3: CSV fayllar yuklanmoqda...")
    documents = load_documents(config.DATA_SOURCES)
    from collections import Counter
    counts = Counter(d.qonun_kodi for d in documents)
    print(f"    {len(documents)} ta modda topildi: {dict(counts)}")

    with open(config.DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)

    print("2/3: BM25 indeksi qurilmoqda...")
    bm25 = BM25Retriever(documents)
    bm25.save(config.BM25_INDEX_PATH)
    print(f"    Saqlandi: {config.BM25_INDEX_PATH}")

    print("3/3: Dense embedding indeksi qurilmoqda (birinchi marta model yuklanadi, biroz vaqt oladi)...")
    t0 = time.time()
    dense = DenseRetriever()
    dense.build_index(documents)
    dense.save(config.DENSE_INDEX_PATH)
    print(f"    Saqlandi: {config.DENSE_INDEX_PATH} ({time.time()-t0:.1f}s)")

    print("\nTayyor! Endi `python cli.py` yoki `python app_web.py` ni ishga tushiring.")


if __name__ == "__main__":
    main()
