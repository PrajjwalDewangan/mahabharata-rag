import json
import re
from pathlib import Path

MIN_WORDS = 40
MAX_WORDS = 300

def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def split_large_paragraph(text: str) -> list[str]:
    sentences = split_into_sentences(text)

    # Fallback: if a single sentence exceeds MAX_WORDS, split by word count
    chunks = []
    current = []
    current_words = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        # sentence itself is too large — force split by words
        if sentence_words > MAX_WORDS:
            if current:
                chunks.append(' '.join(current))
                current = []
                current_words = 0
            words = sentence.split()
            for i in range(0, len(words), MAX_WORDS):
                chunks.append(' '.join(words[i:i + MAX_WORDS]))
            continue

        if current_words + sentence_words > MAX_WORDS and current:
            chunks.append(' '.join(current))
            current = [sentence]
            current_words = sentence_words
        else:
            current.append(sentence)
            current_words += sentence_words

    if current:
        chunks.append(' '.join(current))

    return chunks

def chunk_section(section: dict) -> list[dict]:
    paragraphs = [p.strip() for p in section['text'].split('\n\n') if p.strip()]

    merged = []
    buffer = []
    buffer_words = 0

    for p in paragraphs:
        words = len(p.split())
        buffer.append(p)
        buffer_words += words
        if buffer_words >= MIN_WORDS:
            merged.append(' '.join(buffer))
            buffer = []
            buffer_words = 0

    if buffer:
        remaining = ' '.join(buffer)
        if merged:
            candidate = merged[-1] + ' ' + remaining
            if len(candidate.split()) <=MAX_WORDS:
                merged[-1] = candidate

            else:
                merged.append(remaining)

        else:
            merged.append(remaining)

    final_paragraphs = []
    for p in merged:
        if (len(p.split()) >= MAX_WORDS):
            final_paragraphs.extend(split_large_paragraph(p))
        
        else:
            final_paragraphs.append(p)

    chunks = []
    for i, text in enumerate(final_paragraphs):
        if not text.strip():
            continue
        chunks.append({
            "parva_number": section["parva_number"],
            "parva_name": section["parva_name"],
            "section_number": section["section_number"],
            "section_heading": section["section_heading"],
            "chunk_index": i,
            "text": text.strip()
        })

    return chunks

def chunk_all(input_path: str = "data/cleaned.json",
              output_path: str = "data/chunks.json"):

    with open(input_path, encoding="utf-8") as f:
        sections = json.load(f)

    all_chunks = []
    for section in sections:
        all_chunks.extend(chunk_section(section))

    word_counts = [len(c['text'].split()) for c in all_chunks]
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Min words: {min(word_counts)}")
    print(f"Max words: {max(word_counts)}")
    print(f"Avg words: {sum(word_counts) // len(word_counts)}")
    print(f"Chunks under 40 words: {sum(1 for w in word_counts if w < 40)}")
    print(f"Chunks over 300 words: {sum(1 for w in word_counts if w > 300)}")

    Path(output_path).write_text(
        json.dumps(all_chunks, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    chunk_all()