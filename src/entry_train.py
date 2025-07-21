import argparse
from pathlib import Path
from src.config import load_model_config, load_train_config
from src.model import load_tokenizer_and_model
from src.data import load_raw_text, build_dataset, with_format
from src.trainer import create_training_arguments, create_data_collator, perplexity
from src.utils import set_seed, ensure_dir, count_trainable_parameters, human_readable
from transformers import TrainerCallback
import yaml
import mlflow, os, time
from src.callbacks import PerplexityAndMLflowCallback, EarlyStoppingCallback

from transformers import Trainer

class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            perplexity(metrics)
            
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

    raw_path = Path(args.data_dir)
    text = load_raw_text(str(raw_path))

    train_ds, eval_ds = build_dataset(tokenizer, text, tc.block_size, tc.eval_fraction)
    train_ds = with_format(train_ds)
    eval_ds = with_format(eval_ds)

    data_collator = create_data_collator(tokenizer)
    training_args = create_training_arguments(tc)

    print(f"Train examples (blocks): {len(train_ds)}  Eval blocks: {len(eval_ds)}")
    print(f"Parameters (trainable): {human_readable(count_trainable_parameters(model))}")


# Load mlflow config
    mlflow_cfg = yaml.safe_load(open('config/mlflow_config.yaml'))
    tracking_uri = mlflow_cfg.get('tracking_uri')
    if tracking_uri: mlflow.set_tracking_uri(tracking_uri)
    exp_name = mlflow_cfg.get('experiment_name', 'default')
    mlflow.set_experiment(exp_name)

    run_name = mlflow_cfg.get('run_name') or f"gpt2-shakespeare-{int(time.time())}"
    with mlflow.start_run(run_name=run_name) as run:
        # Log static params
        mlflow.log_params({
            'model_name': mc.model_name,
            'block_size': tc.block_size,
            'lr': tc.learning_rate,
            'epochs': tc.num_train_epochs,
            'batch_size_per_device': tc.per_device_train_batch_size,
            'grad_accum': tc.gradient_accumulation_steps
        })

        if tc.use_gradient_checkpointing:
            model.config.use_cache = False
            model.gradient_checkpointing_enable()

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
        )

        trainer.add_callback(PerplexityAndMLflowCallback(mlflow))
        trainer.add_callback(EarlyStoppingCallback(patience=2, min_delta=0.0))
        print(">>> total_optim_steps =", trainer._get_train_sampler().__len__() //
            (tc.gradient_accumulation_steps))
        print(">>> eval_steps        =", training_args.eval_steps)
        print(">>> evaluation_strategy =", training_args.evaluation_strategy)

        trainer.train()
        final_metrics = trainer.evaluate()
        if 'eval_loss' in final_metrics:
            mlflow.log_metric('final_eval_loss', final_metrics['eval_loss'])
        mlflow.log_artifacts(tc.output_dir)  # saves model, tokenizer

if __name__ == "__main__":
    main()
