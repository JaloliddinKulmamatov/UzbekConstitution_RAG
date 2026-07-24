"""
CSV fayl(lar)ni o'qib, RAG uchun hujjat (chunk) obyektlariga aylantiradi.

Tizim bir nechta qonun manbasini bir vaqtda qo'llab-quvvatlaydi (masalan
Konstitutsiya + Jinoyat kodeksi). Har bir CSV bir xil ustunlarga ega bo'lishi
kerak: qonun_kodi, qonun_nomi, modda_raqami, bolim, bob_raqami, bob_nomi, matn.

Yangi qonun qo'shish uchun shunchaki shu formatdagi CSV faylni
config.DATA_SOURCES ro'yxatiga qo'shish kifoya.
"""
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class Document:
    doc_id: int
    qonun_kodi: str          # masalan "KONST", "JK"
    qonun_nomi: str          # to'liq nomi, masalan "O'zbekiston Respublikasi Konstitutsiyasi"
    modda_raqami: str        # string, chunki "18-1" kabi qo'shilgan moddalar ham bo'lishi mumkin
    bolim: str
    bob_raqami: str
    bob_nomi: str
    matn: str
    search_text: str = field(default="")

    def to_context_string(self) -> str:
        """LLM promptiga qo'shiladigan formatlangan ko'rinish."""
        bob = f", {self.bob_nomi}" if isinstance(self.bob_nomi, str) and self.bob_nomi.strip() else ""
        return (
            f"[{self.qonun_nomi} — {self.modda_raqami}-modda | {self.bolim}{bob}]\n"
            f"{self.matn}"
        )


def _load_single_csv(csv_path: str, start_id: int) -> list[Document]:
    df = pd.read_csv(csv_path, dtype={"modda_raqami": str})
    df["bob_nomi"] = df["bob_nomi"].fillna("")
    df["bob_raqami"] = df["bob_raqami"].fillna("")

    documents = []
    for i, row in df.iterrows():
        bob_part = f" {row['bob_nomi']}." if row["bob_nomi"] else ""
        search_text = (
            f"{row['qonun_nomi']}. {row['bolim']}.{bob_part} "
            f"{row['modda_raqami']}-modda. {row['matn']}"
        )
        documents.append(
            Document(
                doc_id=start_id + i,
                qonun_kodi=str(row["qonun_kodi"]),
                qonun_nomi=str(row["qonun_nomi"]),
                modda_raqami=str(row["modda_raqami"]),
                bolim=str(row["bolim"]),
                bob_raqami=str(row["bob_raqami"]),
                bob_nomi=str(row["bob_nomi"]),
                matn=str(row["matn"]),
                search_text=search_text,
            )
        )
    return documents


def load_documents(csv_paths) -> list[Document]:
    """
    csv_paths: bitta string (bitta fayl) yoki string'lar ro'yxati (bir nechta qonun).
    Barcha manbalardan hujjatlarni yig'ib, ketma-ket doc_id beradi.
    """
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    all_documents: list[Document] = []
    for path in csv_paths:
        docs = _load_single_csv(path, start_id=len(all_documents))
        all_documents.extend(docs)
    return all_documents


if __name__ == "__main__":
    import config

    docs = load_documents(config.DATA_SOURCES)
    print(f"{len(docs)} ta modda yuklandi (barcha qonunlardan).")
    from collections import Counter
    print(Counter(d.qonun_kodi for d in docs))
    print("---- Namuna (KONST) ----")
    print(next(d for d in docs if d.qonun_kodi == "KONST").to_context_string())
    print()
    print("---- Namuna (JK) ----")
    print(next(d for d in docs if d.qonun_kodi == "JK").to_context_string())
