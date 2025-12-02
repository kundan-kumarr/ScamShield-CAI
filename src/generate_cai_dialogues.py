import json
from pathlib import Path
from typing import List

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from config import load_config
from utils_io import read_jsonl, write_jsonl


def load_constitution(path: str = "data/cai_constitution.json") -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["constitutions"]


def build_conversation(prompt: str, red_response: str, critic_prompt: str, revision_prompt: str):
    """
    Return two instruction strings:
    - for critique
    - for revision
    You can adapt this to your model's chat template.
    """
    base_chat = (
        f"User: {prompt}\n\n"
        f"Assistant: {red_response}\n\n"
    )

    critique_inst = base_chat + f"User: {critic_prompt}\nAssistant:"
    revision_inst = base_chat + f"User: {revision_prompt}\nAssistant:"

    return critique_inst, revision_inst


def main():
    cfg = load_config()

    model_name = cfg.revision_model_name
    print(f"Loading model for critique + revision: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    )
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

    constitutions = load_constitution()
    rows_out = []

    in_path = "data/red_responses.jsonl"
    out_path = "data/cai_dialogues.jsonl"

    for row in tqdm(list(read_jsonl(in_path)), desc="Generating critiques + revisions"):
        prompt = row["prompt"]
        red_response = row["red_response"]

        # For simplicity, just use the first constitution entry.
        constitution = constitutions[0]
        critic_prompt = constitution["critic"]
        revision_prompt = constitution["revision"]

        critique_inst, revision_inst = build_conversation(
            prompt, red_response, critic_prompt, revision_prompt
        )

        critique = generator(
            critique_inst,
            max_new_tokens=256,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p
        )[0]["generated_text"].replace(critique_inst, "").strip()

        revision = generator(
            revision_inst,
            max_new_tokens=256,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p
        )[0]["generated_text"].replace(revision_inst, "").strip()

        rows_out.append({
            "prompt": prompt,
            "red_response": red_response,
            "critique": critique,
            "revision_response": revision
        })

    write_jsonl(out_path, rows_out)
    print(f"Saved {len(rows_out)} rows to {out_path}")


if __name__ == "__main__":
    main()
