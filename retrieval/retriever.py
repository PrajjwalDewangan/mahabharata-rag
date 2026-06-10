import json
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from typing import Optional

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
    
    def rerank(self, query: str, candidates: list[dict], top_n: int = FINAL_TOP_N) -> list[dict]:
        if not candidates:
            return []
 
        pairs = [(query, c["text"]) for c in candidates]
        ce_scores = self.reranker.predict(pairs)
 
        for candidate, score in zip(candidates, ce_scores):
            candidate["rerank_score"] = float(score)
 
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        
        return reranked[:top_n]
    
    def retrieve(
        self,
        query: str,
        final_top_n: int = FINAL_TOP_N,
        dense_top_k: int = DENSE_TOP_K,
        bm25_top_k: int = BM25_TOP_K,
        rrf_top_k: int = RRF_TOP_K,
    ) -> list[dict]:
        
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
        "Who killed Karna?",
        "What is the Bhagavad Gita about?",
        "Who was Draupadi?",
        "What happened at the dice game?",
        "Why did Karna refuse to fight under Bhishma?",
    ]
 
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"{'='*70}")
 
        results = retriever.retrieve(query)
 
        for i, r in enumerate(results, 1):
            rerank_str = f"  rerank={r.get('rerank_score', 'n/a'):.4f}" if "rerank_score" in r else ""
            print(
                f"\n  [{i}] {r['parva_name']}, Section {r['section_number']} "
                f"(chunk {r['chunk_index']})  "
                f"rrf={r['rrf_score']:.5f}{rerank_str}"
            )
            print(f"  {r['text'][:250]}...")