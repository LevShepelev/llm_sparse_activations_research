import argparse
from pathlib import Path
from .config import load_model_config, load_train_config
from .model import load_tokenizer_and_model
from .data import load_raw_text, build_dataset, with_format
from .trainer import create_training_arguments, create_data_collator, perplexity
from .utils import set_seed, ensure_dir, count_trainable_parameters, human_readable

from transformers import Trainer

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_config", default="config/model_config.yaml")
    ap.add_argument("--train_config", default="config/train_config.yaml")
    ap.add_argument("--data_dir", default="data/raw")
    ap.add_argument("--pattern", default="shakespeare.txt")
    return ap.parse_args()

def main():
    args = parse_args()
    mc = load_model_config(args.model_config)
    tc = load_train_config(args.train_config)
    set_seed(tc.seed)
    ensure_dir(tc.output_dir)

    tokenizer, model, _ = load_tokenizer_and_model(mc.model_name, mc.add_special_tokens, mc.resize_token_embeddings)

    raw_path = Path(args.data_dir) / args.pattern
    text = load_raw_text(str(raw_path))

    train_ds, eval_ds = build_dataset(tokenizer, text, tc.block_size, tc.eval_fraction)
    train_ds = with_format(train_ds)
    eval_ds = with_format(eval_ds)

    data_collator = create_data_collator(tokenizer)
    training_args = create_training_arguments(tc)

    print(f"Train examples (blocks): {len(train_ds)}  Eval blocks: {len(eval_ds)}")
    print(f"Parameters (trainable): {human_readable(count_trainable_parameters(model))}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    trainer.add_callback(type("PerplexityCallback", (), {
        "on_evaluate": lambda self, args, state, control, metrics, **kw: perplexity(metrics)
    })())

    trainer.train()
    metrics = trainer.evaluate()
    metrics = perplexity(metrics)
    print(metrics)

    trainer.save_model(tc.output_dir)
    tokenizer.save_pretrained(tc.output_dir)

if __name__ == "__main__":
    main()
