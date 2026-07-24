"""
Barcha sozlamalar shu yerda. Konfiguratsiyani o'zgartirish uchun
faqat shu faylni tahrirlang — boshqa fayllarga tegishning hojati yo'q.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Ma'lumot ----
# Har bir qonun uchun bitta CSV. Barchasi bir xil ustunlarga ega bo'lishi kerak:
# qonun_kodi, qonun_nomi, modda_raqami, bolim, bob_raqami, bob_nomi, matn
# Yangi qonun qo'shish uchun shu ro'yxatga yana bir CSV yo'lini qo'shing va
# `python build_index.py` ni qayta ishga tushiring.
DATA_SOURCES = [
    os.path.join(BASE_DIR, "data", "konstitutsiya_moddalar.csv"),
    os.path.join(BASE_DIR, "data", "jinoyat_kodeksi_moddalar.csv"),
]
INDEX_DIR = os.path.join(BASE_DIR, "index")
os.makedirs(INDEX_DIR, exist_ok=True)

BM25_INDEX_PATH = os.path.join(INDEX_DIR, "bm25.pkl")
DENSE_INDEX_PATH = os.path.join(INDEX_DIR, "dense_embeddings.npy")
DOCS_PATH = os.path.join(INDEX_DIR, "documents.pkl")

# ---- Dense embedding modeli ----
# Ko'p tilli model, o'zbek tilini ham qamrab oladi.
# Kichikroq/tezroq variant kerak bo'lsa: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_DIM = 768
# e5 modellari uchun maxsus prefikslar talab qilinadi
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

# ---- Hybrid retrieval sozlamalari ----
TOP_K_BM25 = 15          # BM25 dan olinadigan nomzodlar soni
TOP_K_DENSE = 15         # Dense qidiruvdan olinadigan nomzodlar soni
TOP_K_FINAL = 5          # LLM ga beriladigan yakuniy kontekst bo'laklari soni
RRF_K = 60               # Reciprocal Rank Fusion konstantasi (odatiy qiymat: 60)

# ---- Ollama (LLM) sozlamalari ----
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = "qwen3:4b-instruct"
OLLAMA_TEMPERATURE = 0.2
OLLAMA_NUM_CTX = 4096

SYSTEM_PROMPT = """Siz O'zbekiston qonunchiligi (Konstitutsiya va Jinoyat kodeksi) bo'yicha yordamchi
yuridik assistentsiz. Tizimda bir nechta qonun manbasi mavjud — har doim qaysi qonundan
foydalanayotganingizni aniq ko'rsating.

Faqat sizga berilgan KONTEKST asosida javob bering. Agar javob kontekstda bo'lmasa, buni ochiq aytib,
taxmin qilmang. Har bir da'voni tegishli QONUN NOMI va modda raqami bilan tasdiqlang
(masalan: "(Konstitutsiya, 18-modda)" yoki "(Jinoyat kodeksi, 97-modda)").
Agar foydalanuvchi qaysi qonunni nazarda tutayotgani noaniq bo'lsa va kontekstda ikkala qonundan
mos moddalar bo'lsa, ikkalasini ham ko'rsating va ularni bir-biridan aniq ajrating.
Javobni o'zbek tilida, aniq va tushunarli tarzda bering."""
