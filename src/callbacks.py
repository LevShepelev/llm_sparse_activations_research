from transformers import TrainerCallback
import math, time

class PerplexityAndMLflowCallback(TrainerCallback):
    def __init__(self, mlflow, run_id=None):
        self.mlflow = mlflow
        self.run_id = run_id
        self.best_eval_loss = math.inf
        self.best_step = -1

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None: return
        if self.mlflow:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.mlflow.log_metric(k, v, step=state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None: return
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None and eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.best_step = state.global_step
        # Log perplexity explicitly
        if "eval_loss" in metrics:
            ppl = math.exp(metrics["eval_loss"]) if metrics["eval_loss"] < 20 else float('inf')
            if self.mlflow:
                self.mlflow.log_metric("perplexity", ppl, step=state.global_step)

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=2, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.bad_epochs = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        loss = metrics.get("eval_loss") if metrics else None
        if loss is None: return
        if loss < self.best - self.min_delta:
            self.best = loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                control.should_training_stop = True
                print(f"Early stopping triggered at step {state.global_step}")