import json
from pathlib import Path
 
ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "chunks.json"
 
 
if __name__ == "__main__":
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
 
    print("=" * 70)
    print("  PART 1 — Krishna's death (Mausala Parva, Section 4)")
    print("  Confirmed via sacred-texts.com: hunter Jara, deer mistake,")
    print("  Krishna in Yoga meditation")
    print("=" * 70)
 
    mausala_4 = [c for c in chunks if c["parva_name"] == "Mausala Parva" and c["section_number"] == 4]
 
    if mausala_4:
        for c in mausala_4:
            print(f"\n  FOUND: Mausala Parva, Section 4, chunk_index {c['chunk_index']}")
            print(f"  {c['text'][:400]}")
    else:
        print("\n  NOT FOUND as section_number == 4.")
        print("  Listing ALL Mausala Parva sections present in chunks.json:")
        mausala_all = sorted(set(c["section_number"] for c in chunks if c["parva_name"] == "Mausala Parva"))
        print(f"  Sections present: {mausala_all}")
        print("\n  If 4 is missing from this list, the chunking/parsing step")
        print("  may have dropped or merged this section - worth checking")
        print("  data/cleaned.json or data/parsed.json for Mausala Parva.")
 
    print("\n  Cross-check: any Mausala Parva chunk containing 'Jara'?")
    mausala_jara = [c for c in chunks if c["parva_name"] == "Mausala Parva" and "Jara" in c["text"]]
    if mausala_jara:
        for c in mausala_jara:
            print(f"\n  Mausala Parva, Section {c['section_number']}, chunk_index {c['chunk_index']}")
            print(f"  {c['text'][:400]}")
    else:
        print("  None found. 'Jara' does not appear in any Mausala Parva chunk.")
        print("  This would mean the death-scene text was either not")
        print("  captured during parsing, or uses different wording than")
        print("  the sacred-texts.com version (translation edition differences).")
 
 
    print("\n\n" + "=" * 70)
    print("  PART 2 — Drona's death / Ashwatthama deception (exploratory)")
    print("  No web confirmation yet - trying alternate keyword phrasings")
    print("=" * 70)
 
    drona_chunks = [c for c in chunks if c["parva_name"] == "Drona Parva"]
    print(f"\n  Total Drona Parva chunks: {len(drona_chunks)}")
 
    alt_searches = [
        (["Drona", "Ashvatthaman", "slain"], "slain instead of dead"),
        (["Drona", "Ashvatthaman", "fallen"], "fallen instead of dead"),
        (["Drona", "weapons", "elephant"], "weapons + elephant"),
        (["Drona", "false", "Ashvatthaman"], "explicit 'false' news"),
        (["Yudhishthira", "Ashvatthaman", "elephant"], "Yudhishthira's role + elephant"),
    ]
 
    for keywords, label in alt_searches:
        matches = [c for c in drona_chunks if all(kw.lower() in c["text"].lower() for kw in keywords)]
        print(f"\n  Search ({label}): {keywords}")
        if matches:
            for c in matches[:3]:
                preview = c["text"][:250].replace("\n", " ")
                print(f"    Section {c['section_number']}, chunk_index {c['chunk_index']}")
                print(f"    {preview}...")
        else:
            print(f"    No matches.")
 
    print("\n" + "=" * 70)
    print("  NEXT STEP FOR PART 2")
    print("=" * 70)
    print("""
  If none of the above surface the right passage, the next step is the
  same as Krishna's case: web_search/web_fetch the actual KMG Drona
  Parva text for the Ashwatthama-elephant episode to get the exact
  phrasing and section number, then search chunks.json for that exact
  section directly (as PART 1 does for Mausala Parva Section 4) rather
  than guessing keywords further.
""")