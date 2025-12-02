from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTTrainer
from transformers import TrainingArguments, AutoModelForCausalLM

from config import load_config


def main():
    cfg = load_config()

    dataset = load_dataset(
        "json",
        data_files={"train": "data/sft_dataset.jsonl"}
    )

    base_model_name = cfg.base_model_name
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_example(example):
        # Simple prompt → response concatenation
        text = f"User: {example['prompt']}\nAssistant: {example['response']}"
        return {"text": text}

    dataset = dataset.map(format_example, remove_columns=dataset["train"].column_names)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype="auto",
    )

    args = TrainingArguments(
        output_dir=cfg.sft_output_dir,
        per_device_train_batch_size=cfg.train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=10,
        save_steps=500,
        bf16=True,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        args=args,
        dataset_text_field="text",
        max_seq_length=1024
    )

    trainer.train()
    trainer.save_model(cfg.sft_output_dir)
    tokenizer.save_pretrained(cfg.sft_output_dir)
    print("SFT training complete.")


if __name__ == "__main__":
    main()
