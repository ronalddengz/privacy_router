import json
import random
import argparse


def get_record_text(item):
    """
    RAT records usually have a top-level 'text' field.
    The Wikipedia records from the earlier script may store text in:
      item["profile"]["article text"]
    This function handles both.
    """
    if "text" in item and item["text"]:
        return item["text"]

    profile = item.get("profile", {})
    if "article text" in profile:
        title = profile.get("article title", "")
        article_text = profile.get("article text", "")

        if title:
            return f"# {title}\n\n{article_text}"
        return article_text

    return ""


def get_direct_ids(item):
    direct = item.get("direct_identifiers", {})
    if isinstance(direct, dict):
        return list(direct.keys())
    if isinstance(direct, list):
        return direct
    return []


def get_indirect_ids(item):
    indirect = item.get("indirect_identifiers", {})
    if isinstance(indirect, dict):
        return list(indirect.keys())
    if isinstance(indirect, list):
        return indirect
    return []


def record_to_txt_block(item):
    """
    Produces blocks like rat_benchmark.txt:

    ---
    # name: RAT-57 (difficulty=3, Chatbot conversation)
    # expected_critical: true
    # rat_difficulty: 3
    # rat_scenario: Chatbot conversation
    # rat_direct_ids: []
    # rat_indirect_ids: ['CIT', 'ESR']

    [START OF TRANSCRIPT]
    ...
    [END OF TRANSCRIPT]
    """

    item_id = str(item.get("id", "unknown"))
    scenario = item.get("scenario", "Unknown")
    difficulty = item.get("difficulty", 0)
    expected_critical = item.get("expected_critical", False)

    direct_ids = get_direct_ids(item)
    indirect_ids = get_indirect_ids(item)

    text = get_record_text(item)

    if item_id.startswith("wiki_") or scenario == "Wikipedia article":
        name = item_id
    else:
        name = f"RAT-{item_id}"

    # If the record text already has transcript markers, do not duplicate them.
    if "[START OF TRANSCRIPT]" in text and "[END OF TRANSCRIPT]" in text:
        body = text
    else:
        body = f"[START OF TRANSCRIPT]\n{text}\n[END OF TRANSCRIPT]"

    block = f"""---
# name: {name} (difficulty={difficulty}, {scenario})
# expected_critical: {str(expected_critical).lower()}
# rat_difficulty: {difficulty}
# rat_scenario: {scenario}
# rat_direct_ids: {direct_ids}
# rat_indirect_ids: {indirect_ids}

{body}
"""

    return block


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", default="merged.json")
    parser.add_argument("--output_json", default="merged_shuffled.json")
    parser.add_argument("--output_txt", default="merged_shuffled.txt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected input JSON to be a list of records.")

    rng = random.Random(args.seed)
    rng.shuffle(data)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    txt_blocks = [record_to_txt_block(item) for item in data]

    with open(args.output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_blocks))

    rat_count = sum(
        1 for x in data
        if not str(x.get("id", "")).startswith("wiki_")
        and x.get("scenario") != "Wikipedia article"
    )

    wiki_count = sum(
        1 for x in data
        if str(x.get("id", "")).startswith("wiki_")
        or x.get("scenario") == "Wikipedia article"
    )

    print(f"Loaded records: {len(data)}")
    print(f"RAT records: {rat_count}")
    print(f"Wikipedia records: {wiki_count}")
    print(f"Wrote shuffled JSON: {args.output_json}")
    print(f"Wrote TXT file: {args.output_txt}")


if __name__ == "__main__":
    main()