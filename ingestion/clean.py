import json
import re
from pathlib import Path

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

    for section in sections:
        section["text"] = clean_text(section["text"])

    output_file.write_text(
        json.dumps(sections, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"Cleaned {len(sections)} sections")
    print(f"Output written to: {output_path}")

if __name__ == "__main__":
    clean_all() 