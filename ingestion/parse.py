import re
import json
from pathlib import Path

PARVA_NAMES = {
    1: "Adi Parva",
    2: "Sabha Parva",
    3: "Vana Parva",
    4: "Virata Parva",
    5: "Udyoga Parva",
    6: "Bhishma Parva",
    7: "Drona Parva",
    8: "Karna Parva",
    9: "Shalya Parva",
    10: "Sauptika Parva",
    11: "Stri Parva",
    12: "Shanti Parva",
    13: "Anushashana Parva",
    14: "Ashvamedhika Parva",
    15: "Ashramavasika Parva",
    16: "Mausala Parva",
    17: "Mahaprasthanika Parva",
    18: "Svargarohana Parva",
}

MISSING_SECTIONS = [
    (7, 54), (7, 55), (7, 189), (12, 364)
]

NUMERIC_PARVAS = {8, 9, 10, 11, 16, 17, 18}

def parse_file(filepath: Path, parva_number: int) -> list[dict]:
    text = filepath.read_text(encoding="utf-8", errors="replace")
    text = '\n'.join(line.rstrip() for line in text.splitlines())

    if parva_number in NUMERIC_PARVAS:
        parts = re.split(r'\n( *\d+)\n', text)
    else:
        parts = re.split(r'\n( *SECTION\s+[IVXLCDM]+)', text)

    sections = []
    section_number = 0

    i = 1
    while i < len(parts) - 1:
        heading = parts[i].strip()
        body = parts[i + 1].strip()
        section_number += 1

        sub_heading_match = re.match(r'^\((.+?)\)', body)
        sub_heading = sub_heading_match.group(1) if sub_heading_match else None

        if sub_heading:
            body = body[sub_heading_match.end():].strip()

        if (parva_number, section_number) in MISSING_SECTIONS:
            print(f"Skipping missing section: Parva {parva_number}, Section {section_number}")
            i += 2
            continue

        sections.append({
            "parva_number": parva_number,
            "parva_name": PARVA_NAMES[parva_number],
            "section_number": section_number,
            "section_heading": heading,
            "sub_heading": sub_heading,
            "text": body
        })

        i += 2

    return sections

def parse_all(txt_dir: str = "data/txt", output_path: str = "data/parsed.json"):
    txt_path = Path(txt_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    all_sections = []
    total_skipped = 0

    for parva_number in range(1, 19):
        filename = txt_path / f"maha{parva_number:02d}.txt"
        if not filename.exists():
            print(f"WARNING: {filename} not found")
            continue

        sections = parse_file(filename, parva_number)
        all_sections.extend(sections)
        print(f"Parva {parva_number:02d} ({PARVA_NAMES[parva_number]}): {len(sections)} sections parsed")

    output.write_text(json.dumps(all_sections, indent=2, ensure_ascii=False))
    print(f"\nTotal sections parsed: {len(all_sections)}")
    print(f"Output written to: {output_path}")

if __name__ == "__main__":
    parse_all()