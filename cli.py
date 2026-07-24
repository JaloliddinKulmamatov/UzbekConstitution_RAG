"""
Terminalda ishlaydigan chat interfeysi.

Ishlatish:
    python cli.py
    python cli.py --debug     # qidiruv natijalarini ham ko'rsatadi
"""
import sys

import config
import ollama_client
from rag_pipeline import load_hybrid_retriever, build_prompt


BANNER = """
========================================================
  O'zbekiston Konstitutsiyasi bo'yicha AI-yordamchi
  (Hybrid RAG: BM25 + Dense embeddings + llama3.2:3b)
========================================================
Savolingizni yozing. Chiqish uchun: 'chiqish', 'exit' yoki Ctrl+C
"""


def main():
    debug = "--debug" in sys.argv

    print("Indekslar yuklanmoqda...")
    try:
        retriever = load_hybrid_retriever()
    except FileNotFoundError:
        print(
            "\nXATOLIK: indekslar topilmadi. Avval quyidagini ishga tushiring:\n"
            "    python build_index.py\n"
        )
        sys.exit(1)

    if not ollama_client.check_ollama_available():
        print(
            f"\nOGOHLANTIRISH: Ollama serverga ulanib bo'lmadi ({config.OLLAMA_HOST}).\n"
            f"Boshqa terminalda quyidagini ishga tushiring:\n"
            f"    ollama pull {config.OLLAMA_MODEL}\n"
            f"    ollama serve\n"
        )

    print(BANNER)

    while True:
        try:
            query = input("Siz: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nXayr!")
            break

        if not query:
            continue
        if query.lower() in ("chiqish", "exit", "quit", "q"):
            print("Xayr!")
            break

        if debug:
            contexts, dbg = retriever.retrieve_with_debug(query)
            print(f"  [BM25 top]: {dbg['bm25_top']}")
            print(f"  [Dense top]: {dbg['dense_top']}")
            print(f"  [Fused top]: {dbg['fused_top']}")
        else:
            contexts = retriever.retrieve(query)

        prompt = build_prompt(query, contexts)

        print("\nAssistent: ", end="", flush=True)
        try:
            for chunk in ollama_client.generate(prompt, system=config.SYSTEM_PROMPT, stream=True):
                print(chunk, end="", flush=True)
        except ollama_client.OllamaError as e:
            print(f"\n[Xatolik] {e}")
        print("\n")

        used_moddas = ", ".join(str(d.modda_raqami) for d in contexts)
        print(f"  (Manba sifatida ko'rilgan moddalar: {used_moddas})\n")


if __name__ == "__main__":
    main()
