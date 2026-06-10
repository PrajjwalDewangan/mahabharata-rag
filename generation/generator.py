import os
import sys
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

sys.path.insert(0, str(ROOT))

from retrieval.retriever import MahabharataRetriever



ENVIRONMENT  = os.getenv("ENVIRONMENT", "production").lower()
IS_LOCAL     = ENVIRONMENT == "local"

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"

MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.1
MAX_HISTORY    = 3
STREAM         = IS_LOCAL



SYSTEM_PROMPT = """You are a knowledgeable guide to the Mahabharata, one of the greatest epics of ancient India. You answer questions based strictly on the provided text passages from the Kisari Mohan Ganguli (KMG) English translation.

Rules you must follow:
1. Answer only from the provided passages. Do not use outside knowledge.
2. Always cite your sources using the format: [Parva Name, Section X]
3. If the passages do not contain enough information to answer, say so clearly — do not guess or hallucinate.
4. Keep answers focused and concise. Avoid repeating the same passage multiple times.
5. When quoting directly, keep quotes short (one or two sentences maximum).
6. Maintain awareness of the conversation history when answering follow-up questions.

IMPORTANT: Never fabricate or paraphrase quotes as if they are direct.
If you quote, it must be word-for-word from the provided passages.
If the passages do not directly answer the question, say:
'The provided passages do not contain enough information to answer this fully.'"""

class ConversationHistory:
    """Sliding window of the last MAX_HISTORY (user, assistant) turn pairs."""

    def __init__(self, max_turns: int = MAX_HISTORY):
        self.max_turns = max_turns
        self.turns: list[dict] = []

    def add(self, user_message: str, assistant_message: str):
        self.turns.append({"user": user_message, "assistant": assistant_message})
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def format(self) -> str:
        if not self.turns:
            return ""
        lines = []
        for turn in self.turns:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['assistant']}")
        return "\n".join(lines)

    def clear(self):
        self.turns = []

    def __len__(self):
        return len(self.turns)


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(query: str, chunks: list[dict], history: ConversationHistory) -> str:
    passages = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[Source {i}]  {chunk['parva_name']}, Section {chunk['section_number']}"
        passages.append(f"{header}\n{chunk['text']}")
    passages_block = "\n\n".join(passages)

    history_block = history.format()

    parts = [SYSTEM_PROMPT]
    if history_block:
        parts.append(f"\n--- Conversation History ---\n{history_block}")
    parts.append(f"\n--- Retrieved Passages ---\n{passages_block}")
    parts.append(f"\n--- Current Question ---\n{query}")
    parts.append("\nAnswer (with citations in [Parva Name, Section X] format):")

    return "\n".join(parts)


# ── Ollama backend ────────────────────────────────────────────────────────────

def call_ollama(prompt: str) -> str:
    payload = {
        "model" : OLLAMA_MODEL,
        "prompt": prompt,
        "stream": STREAM,
        "options": {
            "num_predict": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p"      : 0.9,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=STREAM, timeout=120)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to Ollama. Make sure Ollama is running: `ollama serve`")
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama request timed out after 120 seconds.")

    full_response = []

    if STREAM:
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("response", "")
                print(token, end="", flush=True)
                full_response.append(token)
                if chunk.get("done", False):
                    break
        print()
    else:
        data = response.json()
        full_response.append(data.get("response", ""))

    return "".join(full_response).strip()


# ── Groq backend ──────────────────────────────────────────────────────────────

def call_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set. Add it to your .env file.")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type" : "application/json",
    }

    payload = {
        "model"      : GROQ_MODEL,
        "messages"   : [{"role": "user", "content": prompt}],
        "max_tokens" : MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p"      : 0.9,
    }

    try:
        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to Groq API. Check your internet connection.")
    except requests.exceptions.Timeout:
        raise RuntimeError("Groq API request timed out.")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Groq API error: {e.response.status_code} — {e.response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


# ── Unified LLM call ──────────────────────────────────────────────────────────

def call_llm(prompt: str) -> str:
    if IS_LOCAL:
        return call_ollama(prompt)
    else:
        return call_groq(prompt)


# ── Source formatter ──────────────────────────────────────────────────────────

def format_sources(chunks: list[dict]) -> list[str]:
    seen = set()
    sources = []
    for chunk in chunks:
        key = (chunk["parva_name"], chunk["section_number"])
        if key not in seen:
            seen.add(key)
            sources.append(
                f"{chunk['parva_name']}, Section {chunk['section_number']} "
                f"— {chunk['section_heading']}"
            )
    return sources


# ── Generator ─────────────────────────────────────────────────────────────────

class MahabharataGenerator:

    def __init__(
        self,
        chroma_path : str = "data/chroma",
        chunks_path : str = "data/chunks.json",
        use_reranker: bool = True,
    ):
        self.retriever = MahabharataRetriever(
            chroma_path  = chroma_path,
            chunks_path  = chunks_path,
            use_reranker = use_reranker,
        )
        self.history = ConversationHistory(max_turns=MAX_HISTORY)

        env_label = "Ollama" if IS_LOCAL else "Groq"
        model     = OLLAMA_MODEL if IS_LOCAL else GROQ_MODEL
        print(f"  LLM backend : {env_label} ({model})")

    def chat(self, query: str, verbose: bool = False) -> dict:
        chunks = self.retriever.retrieve(query)

        if verbose:
            print("\n--- Retrieved Chunks ---")
            for i, c in enumerate(chunks, 1):
                print(f"[{i}] {c['parva_name']}, Section {c['section_number']} "
                      f"(rerank={c.get('rerank_score', 'n/a'):.3f})")
                print(f"    {c['text'][:150]}...")
            print()

        prompt = build_prompt(query, chunks, self.history)

        if IS_LOCAL and STREAM:
            print(f"\nAssistant: ", end="", flush=True)

        answer = call_llm(prompt)

        self.history.add(user_message=query, assistant_message=answer)

        return {
            "answer" : answer,
            "sources": format_sources(chunks),
            "chunks" : chunks,
        }

    def reset(self):
        self.history.clear()
        print("Conversation history cleared.")


# ── CLI (local dev only) ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Mahabharata RAG Chatbot")
    print(f"  Environment : {ENVIRONMENT}")
    print("  Type 'quit' to exit, 'reset' to clear history")
    print("=" * 60)

    gen = MahabharataGenerator()

    while True:
        try:
            query = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() == "quit":
            print("Exiting.")
            break
        if query.lower() == "reset":
            gen.reset()
            continue

        result = gen.chat(query, verbose=False)

        if not IS_LOCAL:
            print(f"\nAssistant: {result['answer']}")

        print(f"\nSources:")
        for src in result["sources"]:
            print(f"  • {src}")
        print(f"\n[History: {len(gen.history)} turn(s) in context]")