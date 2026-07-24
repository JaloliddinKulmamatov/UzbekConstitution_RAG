"""
Oddiy veb-chat interfeysi (Flask).

Ishlatish:
    python app_web.py
Keyin brauzerda: http://localhost:5000
"""
from flask import Flask, request, Response, jsonify, render_template_string
import json
 
import config
import ollama_client
from rag_pipeline import load_hybrid_retriever, build_prompt

app = Flask(__name__)
_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = load_hybrid_retriever()
    return _retriever


PAGE = """
<!DOCTYPE html>
<html lang="uz">
<head>
<meta charset="UTF-8">
<title>O'zbekiston Konstitutsiyasi — AI yordamchi</title>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 760px; margin: 40px auto; background:#f4f5f7; color:#1a1a1a;}
  h1 { font-size: 1.3rem; }
  #chat { background:white; border-radius:12px; padding:20px; min-height:400px; box-shadow:0 1px 4px rgba(0,0,0,.08); }
  .msg { margin: 12px 0; line-height:1.5; }
  .user { text-align:right; }
  .user .bubble { background:#2563eb; color:white; display:inline-block; padding:10px 14px; border-radius:14px 14px 2px 14px; max-width:80%; }
  .bot .bubble { background:#eef0f3; display:inline-block; padding:10px 14px; border-radius:14px 14px 14px 2px; max-width:85%; white-space:pre-wrap; }
  .sources { font-size:0.8rem; color:#777; margin-top:4px; }
  #inputRow { display:flex; gap:8px; margin-top:16px; }
  #q { flex:1; padding:10px 14px; border-radius:8px; border:1px solid #ccc; font-size:1rem; }
  button { padding:10px 18px; border:none; border-radius:8px; background:#2563eb; color:white; font-size:1rem; cursor:pointer; }
  button:disabled { background:#aaa; }
</style>
</head>
<body>
<h1>🇺🇿 O'zbekiston Konstitutsiyasi bo'yicha AI-yordamchi</h1>
<p style="color:#666; font-size:0.9rem;">Hybrid RAG (BM25 + dense embeddings) &middot; llama3.2:3b (Ollama)</p>
<div id="chat"></div>
<div id="inputRow">
  <input id="q" placeholder="Savolingizni yozing..." autofocus />
  <button id="sendBtn" onclick="send()">Yuborish</button>
</div>

<script>
const chat = document.getElementById('chat');
const input = document.getElementById('q');
const btn = document.getElementById('sendBtn');

input.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });

function addMsg(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = '<div class="bubble"></div>';
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div.querySelector('.bubble');
}

async function send() {
  const query = input.value.trim();
  if (!query) return;
  input.value = '';
  btn.disabled = true;
  addMsg('user', query).textContent = query;
  const botBubble = addMsg('bot', '');
  botBubble.textContent = '...';

  const resp = await fetch('/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({query})
  });

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let full = '';
  botBubble.textContent = '';
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    const chunkText = decoder.decode(value, {stream:true});
    for (const line of chunkText.split('\\n')) {
      if (!line.trim()) continue;
      const evt = JSON.parse(line);
      if (evt.type === 'token') {
        full += evt.text;
        botBubble.textContent = full;
        chat.scrollTop = chat.scrollHeight;
      } else if (evt.type === 'sources') {
        const src = document.createElement('div');
        src.className = 'sources';
        src.textContent = 'Manba moddalar: ' + evt.moddas.join(', ');
        botBubble.parentElement.appendChild(src);
      } else if (evt.type === 'error') {
        botBubble.textContent = 'Xatolik: ' + evt.text;
      }
    }
  }
  btn.disabled = false;
  input.focus();
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/chat", methods=["POST"])
def chat():
    query = request.json.get("query", "").strip()
    if not query:
        return jsonify({"error": "bo'sh so'rov"}), 400

    retriever = get_retriever()
    contexts = retriever.retrieve(query)
    prompt = build_prompt(query, contexts)
    moddas = [d.modda_raqami for d in contexts]

    def stream():
        try:
            for token in ollama_client.generate(prompt, system=config.SYSTEM_PROMPT, stream=True):
                yield json.dumps({"type": "token", "text": token}) + "\n"
        except ollama_client.OllamaError as e:
            yield json.dumps({"type": "error", "text": str(e)}) + "\n"
            return
        yield json.dumps({"type": "sources", "moddas": moddas}) + "\n"

    return Response(stream(), mimetype="application/x-ndjson")


if __name__ == "__main__":
    print("Indekslar yuklanmoqda...")
    get_retriever()
    print("Server ishga tushdi: http://localhost:5000")
    app.run(debug=False, port=5000)
