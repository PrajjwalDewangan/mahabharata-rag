import sys
import json
import requests
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
from retrieval.retriever import MahabharataRetriever

OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_MODEL    = "gemma3:4b"
MAX_HISTORY     = 3       
STREAM          = True    
MAX_NEW_TOKENS  = 512

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
'The provided passages do not contain enough information to answer this fully.'
"""

class ConversationHistory:
 
    def __init__(self, max_turns: int = MAX_HISTORY):
        self.max_turns = max_turns
        self.turns: list[dict] = []   # each entry: {"user": str, "assistant": str}
 
    def add(self, user_message: str, assistant_message: str):
        self.turns.append({
            "user"     : user_message,
            "assistant": assistant_message,
        })
        # Drop oldest turns beyond the sliding window
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
    
def build_prompt(
    query         : str,
    chunks        : list[dict],
    history       : ConversationHistory,
) -> str:
 
    passages = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[Passage {i} | {chunk['parva_name']}, Section {chunk['section_number']}]"
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

def call_ollama(prompt: str, stream: bool = STREAM) -> str:
    payload = {
        "model" : OLLAMA_MODEL,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "num_predict": MAX_NEW_TOKENS,
            "temperature": 0.1,    
            "top_p"      : 0.9,
        },
    }
 
    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=stream, timeout=120)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure Ollama is running: `ollama serve`"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama request timed out after 120 seconds.")
 
    full_response = []
 
    if stream:
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("response", "")
                print(token, end="", flush=True)
                full_response.append(token)
                if chunk.get("done", False):
                    break
        print()  # newline after streaming finishes
    else:
        data = response.json()
        full_response.append(data.get("response", ""))
 
    return "".join(full_response).strip()

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
 
    def chat(self, query: str, verbose: bool = False) -> dict:
        """
        Process a single query through the full RAG pipeline.
 
        Args:
            query   : User's question.
            verbose : If True, print retrieved chunks before generating.
 
        Returns:
            {
                "answer" : str,         # generated response with inline citations
                "sources": list[str],   # deduplicated source references
                "chunks" : list[dict],  # raw retrieved chunks (for debugging)
            }
        """
        chunks = self.retriever.retrieve(query)
 
        if verbose:
            print("\n--- Retrieved Chunks ---")
            for i, c in enumerate(chunks, 1):
                print(f"[{i}] {c['parva_name']}, Section {c['section_number']} "
                      f"(rerank={c.get('rerank_score', 'n/a'):.3f})")
                print(f"    {c['text'][:150]}...")
            print()
 
        prompt = build_prompt(query, chunks, self.history)
 
        if STREAM:
            print(f"\nAssistant: ", end="", flush=True)
 
        answer = call_ollama(prompt, stream=STREAM)
 
        self.history.add(
            user_message      = query,
            assistant_message = answer,
        )
 
        return {
            "answer" : answer,
            "sources": format_sources(chunks),
            "chunks" : chunks,
        }
 
    def reset(self):
        self.history.clear()
        print("Conversation history cleared.")

if __name__ == "__main__":
    print("=" * 60)
    print("  Mahabharata RAG Chatbot")
    print("  Model  : gemma3:4b (Ollama)")
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
 
        print(f"\nSources:")
        for src in result["sources"]:
            print(f"  • {src}")
 
        print(f"\n[History: {len(gen.history)} turn(s) in context]")