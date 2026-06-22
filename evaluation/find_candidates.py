import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "chunks.json"

MAX_RESULTS_PER_QUERY = 8
PREVIEW_CHARS =300

SEARCHES = [
    {
        "test_query": "How did Arjuna die?",
        "primary_keywords": ["Arjuna", "fell"],
        "alternate_keywords": ["Arjuna", "Mahaprasthanika"],
        "hint": "Expected in Mahaprasthanika Parva - the final Himalayan ascent where the Pandavas fall one by one.",
    },
    {
        "test_query": "How did Bhima die?",
        "primary_keywords": ["Bhima", "fell"],
        "alternate_keywords": ["Bhimasena", "Mahaprasthanika"],
        "hint": "Expected in Mahaprasthanika Parva, same sequence as Arjuna.",
    },
    {
        "test_query": "How did Krishna die?",
        "primary_keywords": ["Krishna", "Jara"],
        "alternate_keywords": ["Vasudeva", "hunter"],
        "hint": "Expected in Mausala Parva - struck by a hunter named Jara who mistook him for a deer.",
    },
    {
        "test_query": "How did Bhishma die?",
        "primary_keywords": ["Bhishma", "arrows", "fell"],
        "alternate_keywords": ["Bhishma", "Shikhandi"],
        "hint": "Expected around Bhishma Parva section 119-120 (the fall) and possibly Anushasana Parva (his eventual death).",
    },
    {
        "test_query": "How did Drona die?",
        "primary_keywords": ["Drona", "Ashvatthaman", "dead"],
        "alternate_keywords": ["Drona", "weapons", "Dhrishtadyumna"],
        "hint": "Expected earlier in Drona Parva than Section 193 - the false news of Ashwatthama's death that made Drona lay down his weapons.",
    },
    {
        "test_query": "Ashwatthama the elephant",
        "primary_keywords": ["elephant", "Ashvatthaman"],
        "alternate_keywords": ["elephant", "Bhima", "slain"],
        "hint": "Look for an elephant NAMED Ashwatthama being killed by Bhima - distinct from Drona's son.",
    },
    {
        "test_query": "List the names of all the Pandavas",
        "primary_keywords": ["Yudhishthira", "Bhimasena", "Arjuna", "Nakula", "Sahadeva"],
        "alternate_keywords": ["five", "sons of Pandu"],
        "hint": "Expected in early Adi Parva - a passage that names all five brothers together.",
    },
]

def matches_all_keywords(text: str, keywords:list[str]) -> bool:
    text_lower = text.lower()
    return all(kw.lower() in text_lower for kw in keywords)

def search_chunks(chunks: list[dict], keywords: list[str]) -> list[dict]:
    return [c for c in chunks if matches_all_keywords(c["text"], keywords)]

def print_results(label: str, results: list[dict], keywords: list[str]):
    print(f"\n {label} (keywords: {', '.join(keywords)})")
    if not results:
        print(f" No matches.")
        return
    
    for c in results[:MAX_RESULTS_PER_QUERY]:
        preview = c["text"][:PREVIEW_CHARS].replace("\n", " ")
        print(f"\n {c['parva_name']}, Section {c['section_number']} (chunk_index {c['chunk_index']})")
        print(f" {preview}...")
    
    if len(results) > MAX_RESULTS_PER_QUERY:
        print(f"\n ... and {len(results) - MAX_RESULTS_PER_QUERY} more matches not shown.")

if __name__ == "__main__":
    print("Loading chunks.json...")
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks):,} chunks.\n")
 
    print("=" * 70)
    print("  CANDIDATE SEARCH — review each result and verify manually")
    print("=" * 70)

    for search in SEARCHES:
        print("\n" + "-" * 70)
        print(f"  QUERY: {search['test_query']}")
        print(f"  Hint:  {search['hint']}")
 
        primary_results = search_chunks(chunks, search["primary_keywords"])
        print_results("Primary search", primary_results, search["primary_keywords"])
 
        if not primary_results:
            alt_results = search_chunks(chunks, search["alternate_keywords"])
            print_results("Alternate search (primary found nothing)", alt_results, search["alternate_keywords"])

    print("\n" + "=" * 70)
    print("  PARVA SIZE CHECK")
    print("  (counts chunks per parva - helps explain if late parvas")
    print("   are underrepresented, which could explain multiple")
    print("   character_death failures at once)")
    print("=" * 70)

    from collections import Counter
    parva_counts = Counter(c["parva_name"] for c in chunks)
 
    # Print in a sensible reading order if possible, otherwise by count
    for parva, count in parva_counts.most_common():
        print(f"  {parva:<25} {count:>5} chunks")

    print("\n" + "=" * 70)
    print("  NEXT STEPS")
    print("=" * 70)
    print("""
  For each query above:
    1. Read the preview text of each candidate chunk
    2. If it GENUINELY describes the event (not just mentions the
       character in passing), note its parva_name + section_number
    3. Open evaluation/new_test_cases.json and replace the
       "relevant_chunks": null for that query with:
 
         "relevant_chunks": [
           {"parva_name": "...", "section_number": ...}
         ]
 
       (multiple entries if the event spans more than one section)
 
  If a query shows "No matches" for BOTH primary and alternate
  searches, that's a genuine finding: either this keyword-search
  approach needs different keywords (try synonyms - e.g. "slain"
  instead of "fell"), or the event is described in this translation
  using language that doesn't match common phrasing at all - which
  would itself help explain why semantic+BM25 retrieval also misses it.
 
  The "List the Pandavas" query is the most important to verify
  carefully: if a chunk DOES correctly list all five brothers and
  retrieval found it, but generation still produced the wrong list
  (Kunti instead of Bhima), that proves the error is in GENERATION
  not retrieval - a meaningfully different finding than if retrieval
  never found a correct enumeration at all.
""")