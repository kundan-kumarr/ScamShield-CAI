from utils_io import read_jsonl, write_jsonl


def build_sft():
    rows_in = list(read_jsonl("data/cai_dialogues.jsonl"))
    rows_out = []
    for r in rows_in:
        rows_out.append({
            "prompt": r["prompt"],
            "response": r["revision_response"]
        })
    write_jsonl("data/sft_dataset.jsonl", rows_out)
    print(f"SFT dataset size: {len(rows_out)}")


def build_preference():
    rows_in = list(read_jsonl("data/cai_dialogues.jsonl"))
    rows_out = []
    for r in rows_in:
        rows_out.append({
            "prompt": r["prompt"],
            "chosen": r["revision_response"],
            "rejected": r["red_response"]
        })
    write_jsonl("data/preference_dataset.jsonl", rows_out)
    print(f"Preference dataset size: {len(rows_out)}")


def main():
    build_sft()
    build_preference()


if __name__ == "__main__":
    main()
