import json
import argparse


def py_list_str(items):
    """
    Format a Python list in the same style as the RAT txt metadata:
    ['address', 'email']
    """
    return repr(list(items))


def get_record_text(record):
    """
    RAT records likely have a top-level 'text' field.
    Wikipedia records from the earlier script store article text in:
      profile["article text"]
    """
    if "text" in record and record["text"]:
        return record["text"]

    profile = record.get("profile", {}) or {}

    if "article text" in profile:
        title = profile.get("article title", "Untitled Wikipedia article")
        article_text = profile.get("article text", "")
        return (
            "[START OF ARTICLE]\n"
            f"Title: {title}\n\n"
            f"{article_text}\n"
            "[END OF ARTICLE]"
        )

    return ""


def make_name(record):
    record_id = str(record.get("id", "unknown"))
    scenario = record.get("scenario", "Unknown scenario")
    difficulty = record.get("difficulty", "unknown")

    if record_id.startswith("wiki_"):
        return f"WIKI-{record_id.replace('wiki_', '')} (difficulty={difficulty}, {scenario})"

    return f"RAT-{record_id} (difficulty={difficulty}, {scenario})"


def record_to_txt_block(record):
    text = get_record_text(record).strip()

    difficulty = record.get("difficulty", "")
    scenario = record.get("scenario", "")
    expected_critical = record.get("expected_critical", False)

    direct_identifiers = record.get("direct_identifiers", {}) or {}
    indirect_identifiers = record.get("indirect_identifiers", {}) or {}

    direct_ids = list(direct_identifiers.keys())
    indirect_ids = list(indirect_identifiers.keys())

    lines = []

    if text:
        lines.append(text)
        lines.append("")

    lines.append("---")
    lines.append(f"# name: {make_name(record)}")
    lines.append(f"# expected_critical: {str(expected_critical).lower()}")
    lines.append(f"# rat_difficulty: {difficulty}")
    lines.append(f"# rat_scenario: {scenario}")
    lines.append(f"# rat_direct_ids: {py_list_str(direct_ids)}")
    lines.append(f"# rat_indirect_ids: {py_list_str(indirect_ids)}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="merged.json")
    parser.add_argument("--output", default="merged.txt")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected input JSON to be a list of records.")

    blocks = [record_to_txt_block(record) for record in data]

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocks))
        f.write("\n")

    print(f"Read {len(data)} records from {args.input}")
    print(f"Wrote TXT file to {args.output}")


if __name__ == "__main__":
    main()