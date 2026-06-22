import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retrieval.retriever import MahabharataRetriever

TEST_SET_PATH = ROOT / "evaluation" / "test_set.json"
K_VALUES = [5, 10, 20]   # recall@5 = what generator sees, @10/@20 = diagnostic


def chunk_key(chunk: dict) -> tuple:
    return (chunk["parva_name"], chunk["section_number"])

def is_hit(retrieved_chunks: list[dict], relevant_chunks: list[dict]) -> bool:
    retrieved_keys = {chunk_key(c) for c in retrieved_chunks}
    relevant_keys  = {(rc["parva_name"], rc["section_number"]) for rc in relevant_chunks}
    return len(retrieved_keys & relevant_keys) > 0

def evaluate(retriever: MahabharataRetriever, test_set: list[dict]) -> dict:
    max_k = max(K_VALUES)

    per_query_results = []
    hits_at_k = {k: 0 for k in K_VALUES}

    for case in test_set:
        query = case["query"]
        relevant = case["relevant_chunks"]
        retrieved = retriever.retrieve(
            query,
            final_top_n=max_k,
            multi_query=case.get("multi_query", False),
        )

        query_result = {
            "query"   : query,
            "category": case.get("category", "uncategorized"),
            "hits"    : {},
        }

        for k in K_VALUES:
            top_k = retrieved[:k]
            hit = is_hit(top_k, relevant)
            query_result["hits"][k] = hit
            if hit:
                hits_at_k[k] += 1

        if not query_result["hits"][5]:
            query_result["top5_retrieved"] = [
                f"{c['parva_name']}, Section {c['section_number']}"
                for c in retrieved[:5]
            ]

        per_query_results.append(query_result)

    total = len(test_set)
    recall_scores = {k: round(hits_at_k[k] / total, 3) for k in K_VALUES}

    return {
        "recall"    : recall_scores,
        "total"     : total,
        "per_query" : per_query_results,
    }

def print_report(results: dict):
    print("=" * 70)
    print("  RETRIEVAL EVALUATION — Recall@k")
    print("=" * 70)

    print(f"\n  Test set size: {results['total']} queries\n")

    for k, score in results["recall"].items():
        bar_len = int(score * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  Recall@{k:<3} {bar}  {score:.1%}")

    print("\n" + "-" * 70)
    print("  Per-query breakdown")
    print("-" * 70)

    for q in results["per_query"]:
        marks = "".join(
            "✓" if q["hits"][k] else "✗"
            for k in K_VALUES
        )
        print(f"\n  [{marks}]  ({q['category']})")
        print(f"  {q['query']}")

        if not q["hits"][5]:
            print(f"  → top-5 retrieved instead:")
            for r in q["top5_retrieved"]:
                print(f"      - {r}")

    print("\n" + "=" * 70)

    categories: dict[str, list[bool]] = {}
    for q in results["per_query"]:
        categories.setdefault(q["category"], []).append(q["hits"][5])

    print("  Recall@5 by category")
    print("-" * 70)
    for cat, hits in categories.items():
        score = sum(hits) / len(hits)
        print(f"  {cat:<20} {sum(hits)}/{len(hits)}  ({score:.0%})")

    print("=" * 70)

if __name__ == "__main__":
    print("Loading test set...")
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        test_set = json.load(f)

    print(f"Loaded {len(test_set)} test queries.\n")

    retriever = MahabharataRetriever(
        chroma_path = str(ROOT / "data" / "chroma"),
        chunks_path = str(ROOT / "data" / "chunks.json"),
    )

    results = evaluate(retriever, test_set)
    print_report(results)

    output_path = ROOT / "evaluation" / "results"
    output_path.mkdir(exist_ok=True)

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_path / f"retrieval_eval_{timestamp}.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {out_file}")
    print("Run again after pipeline changes to compare recall scores over time.")