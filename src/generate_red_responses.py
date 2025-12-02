from pathlib import Path
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from config import load_config
from utils_io import read_jsonl, write_jsonl


def main():
    cfg = load_config()
    model_name = cfg.base_model_name

    print(f"Loading base model: {model_name}")
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

    in_path = "data/red_prompts.jsonl"
    out_path = "data/red_responses.jsonl"

    rows_out = []

    for row in tqdm(list(read_jsonl(in_path)), desc="Generating red responses"):
        user_prompt = row["prompt"]

        # Simple chat template for instruct models
        chat = f"<s>[INST] {user_prompt} [/INST]"
        completion = generator(
            chat,
            max_new_tokens=256,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p
        )[0]["generated_text"]

        # Strip the prompt part if needed
        red_response = completion.replace(chat, "").strip()

        rows_out.append({
            "prompt": user_prompt,
            "red_response": red_response
        })

    write_jsonl(out_path, rows_out)
    print(f"Saved {len(rows_out)} rows to {out_path}")


if __name__ == "__main__":
    main()
