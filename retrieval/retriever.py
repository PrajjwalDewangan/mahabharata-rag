import os
import json
import requests
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

COLLECTION_NAME   = "mahabharata"
EMBED_MODEL_NAME  = "all-mpnet-base-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
 
DEFAULT_CHROMA_PATH = "data/chroma"
DEFAULT_CHUNKS_PATH = "data/chunks.json"

DENSE_TOP_K = 20
BM25_TOP_K  = 20

RRF_TOP_K   = 20
FINAL_TOP_N = 5

RRF_K = 60

NUM_REPHRASINGS = 2

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"

REPHRASING_PROMPT = """Generate {n} alternative search queries to find relevant passages about the following question in the Mahabharata text.

Original query: {query}

Rules:
1. Use alternative names or epithets for characters if you are CERTAIN of them (e.g. Arjuna = Dhananjaya = Phalguna = Partha, Bhima = Bhimasena = Vrikodara, Krishna = Vasudeva = Keshava)
2. Vary the phrasing — try both broad and specific angles
3. Do NOT guess parva names or section numbers — only include them if they appear in the original query
4. Do NOT mix up characters — each rephrasing must be about the same character as the original query
5. Output ONLY the {n} queries, one per line, no numbering, no explanation

Alternative queries:"""

def tokenise(text: str) -> list[str]:
    return text.lower().split()

class MahabharataRetriever:

    def __init__(
        self,
        chroma_path: str = DEFAULT_CHROMA_PATH,
        chunks_path: str = DEFAULT_CHUNKS_PATH,
        use_reranker: bool = True,
    ):
        print("Initialising retriever...")

        print(f"Loading embedding model: {EMBED_MODEL_NAME}")
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)

        print(f"Connecting to ChromaDB: {chroma_path}")
        client = chromadb.PersistentClient(path=chroma_path)
        self.collection = client.get_collection(COLLECTION_NAME)

        print(f"Building BM25 index: {chunks_path}")
        with open(chunks_path, encoding="utf-8") as f:
            self.chunks: list[dict] = json.load(f)

        tokenised_corpus = [tokenise(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenised_corpus)
        print(f"BM25 index built: {len(self.chunks):,} documents")

        self.use_reranker = use_reranker
        if use_reranker:
            print(f"Loading reranker: {RERANK_MODEL_NAME}")
            self.reranker = CrossEncoder(RERANK_MODEL_NAME)
 
        print("Retriever ready.\n")

    def dense_search(self, query: str, top_k: int = DENSE_TOP_K) -> list[dict]:
        query_embedding = self.embed_model.encode(query).tolist()
 
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "id"              : None,           
                "text"            : doc,
                "score"           : 1.0 - dist,     
                "parva_number"    : meta["parva_number"],
                "parva_name"      : meta["parva_name"],
                "section_number"  : meta["section_number"],
                "section_heading" : meta["section_heading"],
                "chunk_index"     : meta["chunk_index"],
                "retriever"       : "dense",
            })

        return hits
    
    def bm25_search(self, query: str, top_k: int = BM25_TOP_K) -> list[dict]:
        tokenised_query = tokenise(query)
        scores = self.bm25.get_scores(tokenised_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
 
        hits = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            hits.append({
                "id"              : idx,
                "text"            : chunk["text"],
                "score"           : float(scores[idx]),
                "parva_number"    : chunk["parva_number"],
                "parva_name"      : chunk["parva_name"],
                "section_number"  : chunk["section_number"],
                "section_heading" : chunk["section_heading"],
                "chunk_index"     : chunk["chunk_index"],
                "retriever"       : "bm25",
            })
 
        return hits
    
    def rrf_merge(
        self,
        dense_hits: list[dict],
        bm25_hits: list[dict],
        top_k: int = RRF_TOP_K,
    ) -> list[dict]:
        
        doc_scores: dict[tuple, dict] = {}
 
        def _key(hit: dict) -> tuple:
            return (hit["parva_number"], hit["section_number"], hit["chunk_index"])
 
        for rank, hit in enumerate(dense_hits, start=1):
            k = _key(hit)
            if k not in doc_scores:
                doc_scores[k] = {"rrf_score": 0.0, "hit": hit}
            doc_scores[k]["rrf_score"] += 1.0 / (RRF_K + rank)
 
        for rank, hit in enumerate(bm25_hits, start=1):
            k = _key(hit)
            if k not in doc_scores:
                doc_scores[k] = {"rrf_score": 0.0, "hit": hit}
            doc_scores[k]["rrf_score"] += 1.0 / (RRF_K + rank)

        sorted_docs = sorted(
            doc_scores.values(), key=lambda x: x["rrf_score"], reverse=True
        )[:top_k]
 
        merged = []
        for entry in sorted_docs:
            hit = entry["hit"].copy()
            hit["rrf_score"] = round(entry["rrf_score"], 6)
            hit["retriever"] = "hybrid"
            merged.append(hit)
 
        return merged
    
    def rrf_merge_multi(
            self,
            ranked_lists: list[list[dict]],
            top_k: int = RRF_TOP_K
    ) -> list[dict]:
        doc_scores: dict[tuple, dict] = {}
 
        def _key(hit: dict) -> tuple:
            return (hit["parva_number"], hit["section_number"], hit["chunk_index"])
 
        for ranked_list in ranked_lists:
            for rank, hit in enumerate(ranked_list, start=1):
                k = _key(hit)
                if k not in doc_scores:
                    doc_scores[k] = {"rrf_score": 0.0, "hit": hit}
                doc_scores[k]["rrf_score"] += 1.0 / (RRF_K + rank)
 
        sorted_docs = sorted(
            doc_scores.values(), key=lambda x: x["rrf_score"], reverse=True
        )[:top_k]
 
        merged = []
        for entry in sorted_docs:
            hit = entry["hit"].copy()
            hit["rrf_score"] = round(entry["rrf_score"], 6)
            hit["retriever"] = "hybrid_multi"
            merged.append(hit)
 
        return merged
    
    def rerank(self, query: str, candidates: list[dict], top_n: int = FINAL_TOP_N) -> list[dict]:
        if not candidates:
            return []
 
        pairs = [(query, c["text"]) for c in candidates]
        ce_scores = self.reranker.predict(pairs)
 
        for candidate, score in zip(candidates, ce_scores):
            candidate["rerank_score"] = float(score)
 
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        
        return reranked[:top_n]
    
    def generate_rephrasings(
            self,
            query: str,
            n: int = NUM_REPHRASINGS
    ) -> list[str]:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not GROQ_API_KEY:
            print("  [multi-query] GROQ_API_KEY not set — skipping rephrasings")
            return []
 
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type" : "application/json",
        }
 
        payload = {
            "model"      : GROQ_MODEL,
            "messages"   : [{
                "role"   : "user",
                "content": REPHRASING_PROMPT.format(query=query, n=n),
            }],
            "max_tokens" : 150,
            "temperature": 0.7,  # slightly higher than generation — we WANT varied phrasing
        }
 
        try:
            response = requests.post(
                GROQ_URL, headers=headers, json=payload, timeout=15
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"].strip()
 
            rephrasings = []
            for line in raw.splitlines():
                line = line.strip()
                # Remove leading numbering like "1." or "1)" if model adds it
                if line and not line.isspace():
                    import re
                    line = re.sub(r"^\d+[\.\)]\s*", "", line)
                    if line:
                        rephrasings.append(line)
 
            return rephrasings[:n]  # cap at requested n even if model returns more
 
        except Exception as e:
            print(f"  [multi-query] Rephrasing failed: {e} — falling back to single-query")
            return []
    
    def retrieve(
        self,
        query: str,
        final_top_n: int = FINAL_TOP_N,
        dense_top_k: int = DENSE_TOP_K,
        bm25_top_k: int = BM25_TOP_K,
        rrf_top_k: int = RRF_TOP_K,
        multi_query: bool = False,
    ) -> list[dict]:
        
        if multi_query:
            rephrasings = self.generate_rephrasings(query)
 
            all_queries = [query] + rephrasings
 
            if len(all_queries) == 1:
                print("  [multi-query] No rephrasings generated — using single-query mode")
                multi_query = False
            else:
                print(f"  [multi-query] Searching with {len(all_queries)} queries:")
                for i, q in enumerate(all_queries):
                    label = "(original)" if i == 0 else f"(rephrasing {i})"
                    print(f"    {label}: {q}")
 
                all_ranked_lists = []
                for q in all_queries:
                    all_ranked_lists.append(self.dense_search(q, top_k=dense_top_k))
                    all_ranked_lists.append(self.bm25_search(q, top_k=bm25_top_k))
 
                merged = self.rrf_merge_multi(all_ranked_lists, top_k=rrf_top_k)
 
                if self.use_reranker:
                    return self.rerank(query, merged, top_n=final_top_n)
                else:
                    return merged[:final_top_n]
        
        dense_hits  = self.dense_search(query, top_k=dense_top_k)
        bm25_hits   = self.bm25_search(query, top_k=bm25_top_k)
        merged      = self.rrf_merge(dense_hits, bm25_hits, top_k=rrf_top_k)
 
        if self.use_reranker:
            results = self.rerank(query, merged, top_n=final_top_n)
        else:
            results = merged[:final_top_n]
 
        return results
    
if __name__ == "__main__":
    retriever = MahabharataRetriever()
 
    test_queries = [
        ("How did Arjuna die?",   True),
        ("How did Bhishma die?",  True),
        ("How did Krishna die?",  False),
        ("Who killed Karna?",     False),
    ]
 
    for query, use_multi in test_queries:
        mode = "MULTI" if use_multi else "SINGLE"
        print(f"\n{'='*70}")
        print(f"Query [{mode}]: {query}")
        print(f"{'='*70}")
 
        results = retriever.retrieve(query, multi_query=use_multi)
 
        for i, r in enumerate(results, 1):
            rerank_str = (
                f"  rerank={r['rerank_score']:.4f}"
                if "rerank_score" in r else ""
            )
            print(
                f"\n  [{i}] {r['parva_name']}, Section {r['section_number']} "
                f"(chunk {r['chunk_index']})  "
                f"rrf={r['rrf_score']:.5f}{rerank_str}"
            )
            print(f"  {r['text'][:250]}...")