import os
import sys
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
 
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
 
from generation.generator import MahabharataGenerator

TEST_SET_PATH = ROOT / "evaluation" / "test_set.json"
 
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"

RATE_LIMIT_DELAY_SECONDS = 1.0

FAITHFULNESS_JUDGE_PROMPT = """You are evaluating a RAG system's answer for FAITHFULNESS.
 
Faithfulness means: every factual claim in the ANSWER is supported by the RETRIEVED PASSAGES. The answer should not contain information, quotes, or details that cannot be traced back to the passages.
 
Note: faithfulness is NOT about whether the answer is correct or complete - only whether it stays grounded in the given passages. An answer that says "the passages don't contain enough information" when that's true is perfectly faithful.
 
RETRIEVED PASSAGES:
{passages}
 
QUESTION: {query}
 
ANSWER: {answer}
 
Score the answer's faithfulness from 1 to 5:
1 = Answer contains major fabricated claims or quotes not in the passages
2 = Answer contains some unsupported details mixed with supported ones
3 = Answer is mostly grounded but has minor unsupported elaborations
4 = Answer is fully grounded with only trivial phrasing additions
5 = Answer is entirely supported by the passages, with accurate citations
 
Respond with ONLY valid JSON, no other text:
{{"score": <1-5>, "reasoning": "<one sentence explaining the score, citing specific evidence>"}}"""

RELEVANCE_JUDGE_PROMPT = """You are evaluating a RAG system's answer for ANSWER RELEVANCE.
 
Answer relevance means: the answer directly addresses what the user asked, without dodging, rambling, or answering a different (even related) question.
 
QUESTION: {query}
 
ANSWER: {answer}
 
Score the answer's relevance from 1 to 5:
1 = Answer does not address the question at all
2 = Answer addresses a related but different question
3 = Answer partially addresses the question with significant gaps
4 = Answer addresses the question with minor digressions
5 = Answer directly and completely addresses what was asked
 
Respond with ONLY valid JSON, no other text:
{{"score": <1-5>, "reasoning": "<one sentence explaining the score>"}}"""

def call_judge(prompt: str) -> dict:
    if not GROQ_API_KEY:
        return {"score": 0, "reasoning": "GROQ_API_KEY not set"}
 
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type" : "application/json",
    }
    payload = {
        "model"      : GROQ_MODEL,
        "messages"   : [{"role": "user", "content": prompt}],
        "max_tokens" : 150,
        "temperature": 0.0,   # judge should be deterministic, not creative
    }
 
    try:
        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        raw_text = data["choices"][0]["message"]["content"].strip()

        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`").lstrip("json").strip()
 
        parsed = json.loads(raw_text)
        return {
            "score"    : int(parsed.get("score", 0)),
            "reasoning": parsed.get("reasoning", ""),
        }
 
    except json.JSONDecodeError:
        return {"score": 0, "reasoning": f"Judge returned non-JSON: {raw_text[:100]}"}
    except Exception as e:
        return {"score": 0, "reasoning": f"Judge call failed: {str(e)}"}
    
def evaluate(generator: MahabharataGenerator, test_set: list[dict]) -> dict:
    per_query_results = []
 
    for i, case in enumerate(test_set, 1):
        query = case["query"]
        print(f"  [{i}/{len(test_set)}] {query}")

        generator.reset()
 
        result = generator.chat(query, verbose=False)
        answer = result["answer"]
        chunks = result["chunks"]

        passages_text = "\n\n".join(
            f"[{c['parva_name']}, Section {c['section_number']}]\n{c['text']}"
            for c in chunks
        )
 
        time.sleep(RATE_LIMIT_DELAY_SECONDS)
        faithfulness = call_judge(
            FAITHFULNESS_JUDGE_PROMPT.format(
                passages=passages_text, query=query, answer=answer
            )
        )
 
        time.sleep(RATE_LIMIT_DELAY_SECONDS)
        relevance = call_judge(
            RELEVANCE_JUDGE_PROMPT.format(query=query, answer=answer)
        )
 
        per_query_results.append({
            "query"       : query,
            "category"    : case.get("category", "uncategorized"),
            "answer"      : answer,
            "sources"     : result["sources"],
            "faithfulness": faithfulness,
            "relevance"   : relevance,
        })

    faithfulness_scores = [r["faithfulness"]["score"] for r in per_query_results if r["faithfulness"]["score"] > 0]
    relevance_scores    = [r["relevance"]["score"]    for r in per_query_results if r["relevance"]["score"]    > 0]

    avg_faithfulness = round(sum(faithfulness_scores) / len(faithfulness_scores), 2) if faithfulness_scores else 0
    avg_relevance    = round(sum(relevance_scores)    / len(relevance_scores),    2) if relevance_scores    else 0
 
    return {
        "avg_faithfulness": avg_faithfulness,
        "avg_relevance"   : avg_relevance,
        "judge_failures"  : len(test_set) - len(faithfulness_scores) + len(test_set) - len(relevance_scores),
        "per_query"       : per_query_results,
    }

def print_report(results: dict):
    print("\n" + "=" * 70)
    print("  GENERATION EVALUATION — Faithfulness & Relevance (LLM-as-judge)")
    print("=" * 70)
 
    print(f"\n  Avg Faithfulness: {results['avg_faithfulness']}/5")
    print(f"  Avg Relevance:    {results['avg_relevance']}/5")
    if results["judge_failures"] > 0:
        print(f"  (judge call failures excluded from average: {results['judge_failures']})")
 
    print("\n" + "-" * 70)
    print("  Per-query breakdown")
    print("-" * 70)
 
    for q in results["per_query"]:
        f_score = q["faithfulness"]["score"]
        r_score = q["relevance"]["score"]

        f_flag = " ⚠" if 0 < f_score <= 2 else ""
        r_flag = " ⚠" if 0 < r_score <= 2 else ""
 
        print(f"\n  ({q['category']})  {q['query']}")
        print(f"  Faithfulness: {f_score}/5{f_flag}  — {q['faithfulness']['reasoning']}")
        print(f"  Relevance:    {r_score}/5{r_flag}  — {q['relevance']['reasoning']}")
        print(f"  Answer (first 150 chars): {q['answer'][:150]}...")
 
    print("\n" + "=" * 70)

    categories: dict[str, list[tuple]] = {}
    for q in results["per_query"]:
        categories.setdefault(q["category"], []).append(
            (q["faithfulness"]["score"], q["relevance"]["score"])
        )
 
    print("  Average scores by category")
    print("-" * 70)
    for cat, scores in categories.items():
        f_avg = sum(s[0] for s in scores) / len(scores)
        r_avg = sum(s[1] for s in scores) / len(scores)
        print(f"  {cat:<20} faithfulness: {f_avg:.1f}/5   relevance: {r_avg:.1f}/5")
 
    print("=" * 70)

if __name__ == "__main__":
    print("Loading test set...")
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        test_set = json.load(f)
 
    print(f"Loaded {len(test_set)} test queries.\n")
    print("Initialising generator (this loads embedding + reranker models)...\n")
 
    generator = MahabharataGenerator(
        chroma_path = str(ROOT / "data" / "chroma"),
        chunks_path = str(ROOT / "data" / "chunks.json"),
    )
 
    print("\nRunning evaluation (this calls Groq 3x per query - generation + 2 judges)...\n")
 
    results = evaluate(generator, test_set)
    print_report(results)

    output_path = ROOT / "evaluation" / "results"
    output_path.mkdir(exist_ok=True)
 
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_path / f"generation_eval_{timestamp}.json"
 
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
 
    print(f"\nResults saved to: {out_file}")