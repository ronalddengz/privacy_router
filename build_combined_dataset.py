# build_combined_dataset.py
# python build_combined_dataset.py --rat_path rat_benchmark.json --output_path merged.json --streaming

import json
import re
import argparse
from datasets import load_dataset


BIOGRAPHY_PATTERNS = [
    # common biographical first-sentence patterns
    r"\bwas born\b",
    r"\bborn on\b",
    r"\bborn in\b",
    r"\b\d{4}\s*[-–]\s*\d{4}\b",          # 1920–1999
    r"\b\d{4}\s*[-–]\s*present\b",
    r"\b\d{4}\s*[-–]\s*$",
    r"\bis an? [A-Z][a-z]+(?:-[A-Z][a-z]+)? (actor|actress|singer|writer|politician|scientist|artist|athlete|footballer|musician|director|producer|journalist|lawyer|professor|poet|novelist)\b",
    r"\bwas an? [A-Z][a-z]+(?:-[A-Z][a-z]+)? (actor|actress|singer|writer|politician|scientist|artist|athlete|footballer|musician|director|producer|journalist|lawyer|professor|poet|novelist)\b",
]

SENSITIVE_PATTERNS = [
    # direct identifiers / highly sensitive strings
    r"\b\d{3}-\d{2}-\d{4}\b",                         # SSN-like
    r"\b(?:\d[ -]*?){13,16}\b",                       # credit-card-like
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b", # email
    r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b", # US phone
    r"\b\d{1,5}\s+[A-Za-z0-9 .'-]+\s+"
    r"(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b", # address-like
]

DISALLOWED_TITLE_PATTERNS = [
    r"\bList of\b",
    r"\bDeaths in\b",
    r"\bBirths in\b",
    r"\b\d{4} births\b",
    r"\b\d{4} deaths\b",
]


def looks_biographical(title: str, text: str) -> bool:
    title = title or ""
    text = text or ""

    first_chunk = text[:1200]

    for pat in DISALLOWED_TITLE_PATTERNS:
        if re.search(pat, title, flags=re.IGNORECASE):
            return True

    for pat in BIOGRAPHY_PATTERNS:
        if re.search(pat, first_chunk, flags=re.IGNORECASE):
            return True

    # Many biography pages start with "Name is/was ..."
    # This is intentionally conservative.
    first_sentence = first_chunk.split(".")[0]
    if re.search(r"^[A-Z][A-Za-z .'-]{2,80}\s+(is|was)\s+(an?|the)\s+", first_sentence):
        if re.search(
            r"\b(actor|actress|politician|singer|writer|artist|athlete|scientist|composer|journalist|footballer|poet|novelist|professor|lawyer|businessman|businesswoman)\b",
            first_sentence,
            flags=re.IGNORECASE,
        ):
            return True

    return False


def looks_sensitive(text: str) -> bool:
    text = text or ""

    for pat in SENSITIVE_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True

    return False


def is_good_non_sensitive_article(example, min_chars=1000, max_chars=6000) -> bool:
    title = example.get("title", "") or ""
    text = example.get("text", "") or ""

    if len(text) < min_chars:
        return False

    if looks_biographical(title, text):
        return False

    if looks_sensitive(text):
        return False

    return True


def wiki_to_rat_format(example, new_id: str, max_chars=4000) -> dict:
    title = example.get("title", "")
    text = example.get("text", "")

    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."

    return {
        "id": new_id,
        "difficulty": 0,
        "scenario": "Wikipedia article",
        "expected_critical": False,
        "direct_identifiers": {},
        "indirect_identifiers": {},
        "features": [],
        "profile": {
            "article title": title,
            "article text": text,
            "source": "wikimedia/wikipedia/20231101.en",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rat_path", default="rat_benchmark.json")
    parser.add_argument("--output_path", default="rat_benchmark_with_wikipedia.json")
    parser.add_argument("--num_wiki", type=int, default=300)
    parser.add_argument("--wiki_split", default="train")
    parser.add_argument("--streaming", action="store_true")
    args = parser.parse_args()

    with open(args.rat_path, "r", encoding="utf-8") as f:
        rat_data = json.load(f)

    if not isinstance(rat_data, list):
        raise ValueError("Expected rat_benchmark.json to be a JSON list of records.")

    combined = list(rat_data)

    existing_ids = {str(item.get("id")) for item in rat_data}
    wiki_added = 0

    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split=args.wiki_split,
        streaming=args.streaming,
    )

    for example in ds:
        if wiki_added >= args.num_wiki:
            break

        if not is_good_non_sensitive_article(example):
            continue

        new_id = f"wiki_{wiki_added + 1}"

        while new_id in existing_ids:
            wiki_added += 1
            new_id = f"wiki_{wiki_added + 1}"

        combined.append(wiki_to_rat_format(example, new_id))
        existing_ids.add(new_id)
        wiki_added += 1

    print(f"Loaded RAT records: {len(rat_data)}")
    print(f"Added Wikipedia records: {wiki_added}")
    print(f"Total combined records: {len(combined)}")

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {args.output_path}")


if __name__ == "__main__":
    main()