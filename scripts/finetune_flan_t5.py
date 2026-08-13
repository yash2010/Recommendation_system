import json
import random
import yaml
from pathlib import Path
from types import SimpleNamespace
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

FINETUNE_YAML_PATH = Path("config/finetune.yaml")


def _to_namespace(d: dict) -> SimpleNamespace:
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _to_namespace(value))
        else:
            setattr(ns, key, value)
    return ns


with open(FINETUNE_YAML_PATH, "r", encoding="utf-8") as f:
    finetune_config = _to_namespace(yaml.safe_load(f))

DATA_PATH   = Path(finetune_config.data.data_path)
OUTPUT_DIR  = Path(finetune_config.output_dir)
BASE_MODEL  = finetune_config.base_model
TASK_PREFIX = finetune_config.task_prefix
SEED        = finetune_config.data.seed


class QueryExpansionDataset(Dataset):
    """Tokenizes (input, target) pairs on the fly into model-ready tensors."""

    def __init__(self, pairs: list[dict], tokenizer, max_src_len: int, max_tar_len: int):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tar_len = max_tar_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]

        model_inputs = self.tokenizer(
            TASK_PREFIX + pair["input"],
            max_length=self.max_src_len,
            truncation=True,
        )
        labels = self.tokenizer(
            text_target=pair["target"],
            max_length=self.max_tar_len,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def load_pairs() -> list[dict]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} pairs from {DATA_PATH}")
    return pairs


def split_pairs(pairs: list[dict], train_split: float) -> tuple[list[dict], list[dict]]:
    shuffled = pairs.copy()
    random.Random(SEED).shuffle(shuffled)

    split = int(len(shuffled) * train_split)
    return shuffled[:split], shuffled[split:]


def main():
    pairs = load_pairs()
    train_pairs, val_pairs = split_pairs(pairs, finetune_config.data.train_split)
    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    max_src_len = finetune_config.model.max_src_len
    max_tar_len = finetune_config.model.max_tar_len

    train_dataset = QueryExpansionDataset(train_pairs, tokenizer, max_src_len, max_tar_len)
    val_dataset = QueryExpansionDataset(val_pairs, tokenizer, max_src_len, max_tar_len)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=finetune_config.train.epochs,
        per_device_train_batch_size=finetune_config.train.batch_size,
        per_device_eval_batch_size=finetune_config.train.batch_size,
        learning_rate=finetune_config.train.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=True,
        logging_steps=20,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()

    best_val_loss = trainer.state.best_metric
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    final_dir = OUTPUT_DIR / "best_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    run_metadata = {
        "base_model": BASE_MODEL,
        "task_prefix": TASK_PREFIX,
        "data": {
            "total_pairs": len(pairs),
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
        },
        "train": {
            "epochs": finetune_config.train.epochs,
            "batch_size": finetune_config.train.batch_size,
            "learning_rate": finetune_config.train.learning_rate,
        },
        "best_val_loss": best_val_loss,
    }
    with open(OUTPUT_DIR / "config_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)
    print(f"Saved fine-tuned model to {final_dir}")

    print("\nTesting fine-tuned model:")
    test_queries = [
        "a film that makes you think",
        "something dark and unsettling",
        "a feel-good comedy",
    ]
    model.eval()
    device = next(model.parameters()).device
    for query in test_queries:
        inputs = tokenizer(TASK_PREFIX + query, return_tensors="pt").to(device)
        output_ids = model.generate(**inputs, max_new_tokens=max_tar_len, num_beams=4)
        expanded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\nQuery:    {query}")
        print(f"Expanded: {expanded}")


if __name__ == "__main__":
    main()
