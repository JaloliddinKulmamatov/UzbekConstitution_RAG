# O'zbekiston qonunchiligi — Hybrid RAG tizimi

BM25 (leksik/sparse) + dense embeddings (semantik) qidiruvni birlashtirgan,
mahalliy Ollama modeliga asoslangan RAG tizimi.

Hozircha 2 ta qonun manbasi qo'llab-quvvatlanadi:
- **Konstitutsiya** (KONST) — 155 modda
- **Jinoyat kodeksi** (JK) — 404 modda

Tizim ko'p qonunli (multi-source) qilib qurilgan — yangi qonun qo'shish oson
(pastga qarang, "Yangi qonun qo'shish" bo'limi).

## Arxitektura

```
Savol
  │
  ├──► BM25Retriever  ──┐
  │                      ├──► Reciprocal Rank Fusion ──► Top-K modda ──► Prompt ──► Ollama (llama3.2:3b) ──► Javob
  └──► DenseRetriever ──┘
```

- **BM25** — aniq atamalar, modda raqamlari, huquqiy terminlar bo'yicha kuchli.
- **Dense embeddings** (`intfloat/multilingual-e5-base`) — ma'no bo'yicha
  o'xshashlikni topadi, sinonim/parafrazalarga chidamli.
- **RRF (Reciprocal Rank Fusion)** — ikkala reytingni ballarni normalizatsiya
  qilmasdan, faqat o'rinlar asosida birlashtiradi (sanoat standarti).
- Agar savolda aniq modda raqami ko'rsatilsa ("18-modda nima haqida"), o'sha
  modda har doim kafolatlangan tarzda kontekstga qo'shiladi.
- Har bir modda — mustaqil chunk (bo'lim/bob nomi bilan boyitilgan holda).

## O'rnatish

### 1. Ollama'ni o'rnatish va modelni yuklash

```bash
# https://ollama.com dan Ollama'ni o'rnating, so'ng:
ollama pull llama3.2:3b
ollama serve      # odatda avtomatik fonda ishlaydi, alohida ishga tushirish shart bo'lmasligi mumkin
```

### 2. Python muhitini tayyorlash

```bash
cd uzconst_rag
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Indekslarni qurish (faqat bir marta, yoki ma'lumot o'zgarganda)

```bash
python build_index.py
```

Bu buyruq:
- CSV faylni o'qiydi (`data/konstitutsiya_moddalar.csv`)
- BM25 indeksini quradi va saqlaydi (`index/bm25.pkl`)
- Har bir modda uchun dense embedding hisoblaydi va saqlaydi
  (`index/dense_embeddings.npy`) — birinchi ishga tushganda embedding
  modelini internetdan yuklab oladi (~1 GB, keyingi safar keshdan foydalanadi)

## Ishga tushirish

### Terminal chat

```bash
python cli.py
python cli.py --debug     # BM25/dense/fused reytinglarni ham ko'rsatadi
```

### Veb-interfeys

```bash
python app_web.py
```

Brauzerda oching: http://localhost:5000

Veb-interfeysda tayyor namuna savollar (chip tugmalar) mavjud — ularni bosish
orqali darhol sinab ko'rishingiz mumkin, masalan:
- "18-modda nima haqida?"
- "Prezident qanday saylanadi?"
- "Fuqarolarning asosiy huquqlari qanday?"
- "Davlat tili haqida nima deyilgan?"
- "Oliy Majlis qanday tuziladi?"

## Yangi qonun qo'shish

Tizim `config.DATA_SOURCES` ro'yxatidagi barcha CSV fayllarni avtomatik yig'ib
oladi. Yangi qonun (masalan Fuqarolik kodeksi) qo'shish uchun:

1. Yangi CSV faylni **aynan shu ustunlar bilan** tayyorlang:
   `qonun_kodi, qonun_nomi, modda_raqami, bolim, bob_raqami, bob_nomi, matn`
   - `qonun_kodi` — qisqa, noyob kod (masalan `"FK"`)
   - `modda_raqami` — matn (string) sifatida saqlansin, chunki "97-1" kabi
     qo'shilgan moddalar raqamli bo'lmasligi mumkin
2. Faylni `data/` papkasiga joylashtiring.
3. `config.py`dagi `DATA_SOURCES` ro'yxatiga yo'lini qo'shing.
4. `python build_index.py` ni qayta ishga tushiring — bu barcha manbalarni
   birlashtirib, indekslarni to'liq qayta quradi.

**Muhim**: modda raqamlari turli qonunlarda takrorlanishi mumkin (masalan
ikkalasida ham "1-modda" bor) — tizim buni `(qonun_kodi, modda_raqami)`
juftligi orqali ajratadi, shuning uchun muammo tug'dirmaydi. Foydalanuvchi
"Jinoyat kodeksining 18-moddasi" desa, faqat o'sha qonundan modda topiladi;
"18-modda" deb umumiy so'rasa, mos keladigan barcha qonunlardagi variantlar
ko'rsatiladi va LLM ikkalasini ham javobida ajratib beradi.

## Sozlamalar

Barcha parametrlar `config.py` faylida:

| Parametr | Tavsif | Standart |
|---|---|---|
| `EMBEDDING_MODEL` | Dense qidiruv uchun model | `intfloat/multilingual-e5-base` |
| `TOP_K_BM25` / `TOP_K_DENSE` | Har bir usuldan olinadigan nomzodlar soni | 15 |
| `TOP_K_FINAL` | LLM'ga beriladigan yakuniy bo'laklar soni | 5 |
| `RRF_K` | RRF fusion konstantasi | 60 |
| `OLLAMA_MODEL` | Ollama modeli | `llama3.2:3b` |
| `OLLAMA_TEMPERATURE` | Generatsiya harorati | 0.2 |

**Tezroq/yengilroq embedding kerak bo'lsa**, `config.py`da
`EMBEDDING_MODEL`ni `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
ga o'zgartiring (aniqlik biroz pasayadi, tezlik oshadi), so'ng
`python build_index.py`ni qayta ishga tushiring.

## Fayllar tuzilishi

```
uzconst_rag/
├── config.py            # barcha sozlamalar
├── ingest.py             # CSV -> Document obyektlari
├── retriever.py           # BM25 + Dense + RRF hybrid retrieval
├── build_index.py         # indekslarni qurish skripti
├── ollama_client.py        # Ollama REST API client
├── rag_pipeline.py         # retrieval + prompt + generatsiya
├── cli.py                # terminal chat
├── app_web.py              # veb-chat (Flask)
├── data/
│   ├── konstitutsiya_moddalar.csv
│   └── jinoyat_kodeksi_moddalar.csv
├── index/                # avtomatik yaratiladi (bm25.pkl, dense_embeddings.npy, documents.pkl)
└── requirements.txt
```

## Muhim eslatmalar

- Bu tizim to'liq **mahalliy** ishlaydi — Ollama orqali llama3.2:3b sizning
  kompyuteringizda ishlaydi, hech qanday ma'lumot tashqariga yuborilmaydi
  (faqat birinchi marta embedding modelini yuklab olish uchun internet kerak).
- llama3.2:3b — kichik model, shuning uchun promptda unga faqat KONTEKST
  asosida javob berish va manba modda ko'rsatish qat'iy talab qilinadi
  (`config.SYSTEM_PROMPT`). Baribir vaqti-vaqti bilan xato/gallyutsinatsiya
  bo'lishi mumkin — muhim yuridik qarorlar uchun rasmiy Konstitutsiya matni
  bilan solishtirib tekshirish tavsiya etiladi.
- CSV'da 2 ta moddada `bob_nomi`/`bob_raqami` bo'sh (`NaN`) — bular Konstitutsiya
  bo'yicha bob tashqarisidagi umumiy moddalar bo'lishi mumkin, kod bunday
  holatlarni xavfsiz qayta ishlaydi.
