import json
import re
from pathlib import Path

ARTIFACT_PATTERNS =[
    r'^Section\s+[IVXLCDM]+$',
    r'^\(.*continued\)$',
    r'^END OF .+ PARVA$',
    r'^The End of .+ Parva$',
    r'^FOOTNOTES?$',
    r'^-+$',
    r'^\d+\.\s+.{0,100}$',
    r'^\[\(.*?\)\]$',
    r'^\d+\.\s+.{0,200}$',
    r'^\[End of.*?\]$',
    r'^The end of .+$',
    r'^Here ends .+$',
    r'^\(.*?Continued.*?\)$',
    r'^\.$',
    r'^\d+\.\s+\'.+\'.*$',
]

def is_artifact(text:str) -> bool:
    text = text.strip()
    for pattern in ARTIFACT_PATTERNS:
        if re.match(pattern, text, re.IGNORECASE):
            return True
    return False

def clean_text(text: str) -> str:
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{3,}','\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def clean_all(input_path: str = "data/parsed.json",
              output_path: str = "data/cleaned.json"):
    input_file = Path(input_path)
    output_file = Path(output_path)

    with open(input_file, encoding = "utf-8") as f:
        sections = json.load(f)

    artifact_count = 0

    for section in sections:

        paragraphs = section["text"].split('\n\n')
        filtered = []
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if is_artifact(p):
                artifact_count += 1
                continue
            filtered.append(p)
        section["text"] = '\n\n'.join(filtered)
        section["text"] = clean_text(section["text"])

    output_file.write_text(
        json.dumps(sections, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"Cleaned {len(sections)} sections")
    print(f"Artifacts removed: {artifact_count}")
    print(f"Output written to: {output_path}")

if __name__ == "__main__":
    clean_all() 