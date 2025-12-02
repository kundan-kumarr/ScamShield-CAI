from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer
from transformers import TrainingArguments

from config import load_config


def main():
    cfg = load_config()

    dataset = load_dataset(
        "json",
        data_files={"train": "data/preference_dataset.jsonl"}
    )

    model_name = cfg.sft_output_dir  # start from SFT model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # DPO expects columns: prompt, chosen, rejected
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    )

    args = TrainingArguments(
        output_dir=cfg.dpo_output_dir,
        per_device_train_batch_size=cfg.train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=10,
        save_steps=500,
        bf16=True,
        report_to="none"
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # can load a frozen reference model if you like
        args=args,
        beta=0.1,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        max_length=1024,
        max_prompt_length=512
    )

    trainer.train()
    trainer.save_model(cfg.dpo_output_dir)
    tokenizer.save_pretrained(cfg.dpo_output_dir)
    print("DPO training complete.")


if __name__ == "__main__":
    main()
