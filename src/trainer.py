from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import math

def create_data_collator(tokenizer):
    # We already made fixed-size blocks; no dynamic padding needed.
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

def create_training_arguments(tc):
    return TrainingArguments(
        output_dir=tc.output_dir,
        overwrite_output_dir=True,
        logging_steps=tc.logging_steps,
        save_steps=tc.save_steps,
        eval_steps=tc.eval_steps,
        eval_strategy=tc.eval_strategy,
        save_total_limit=tc.save_total_limit,
        per_device_train_batch_size=tc.per_device_train_batch_size,
        per_device_eval_batch_size=tc.per_device_eval_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        learning_rate=tc.learning_rate,
        weight_decay=tc.weight_decay,
        adam_beta1=tc.adam_beta1,
        adam_beta2=tc.adam_beta2,
        adam_epsilon=tc.adam_eps,
        lr_scheduler_type=tc.lr_scheduler_type,
        warmup_steps=tc.warmup_steps,
        max_grad_norm=tc.max_grad_norm,
        num_train_epochs=tc.num_train_epochs,
        seed=tc.seed,
        report_to=None if tc.report_to == "none" else [tc.report_to],
        fp16=tc.fp16,
        bf16=tc.bf16,
        push_to_hub=tc.push_to_hub,
        logging_dir=f"{tc.output_dir}/logs",
        dataloader_pin_memory=True,
    )

def perplexity(metrics):
    if "eval_loss" in metrics:
        metrics["perplexity"] = round(math.exp(metrics["eval_loss"]), 2)
    return metrics
