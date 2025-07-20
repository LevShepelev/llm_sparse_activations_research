import math, numbers, time
from pathlib import Path
import matplotlib.pyplot as plt
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import mlflow


def _is_num(x):
    return isinstance(x, numbers.Number) and math.isfinite(x)


class PerplexityAndMLflowCallback(TrainerCallback):
    """
    • Logs train_loss every `logging_steps`
    • Logs eval_loss & perplexity every `eval_steps`
    • Builds and uploads a perplexity-vs-step PNG at the end
    """

    def __init__(self):
        self.perp_history = []        # (step, perplexity)

    # ----- Training-side logging -----
    def on_log(self,
               args: TrainingArguments,
               state: TrainerState,
               control: TrainerControl,
               logs=None, **kwargs):
        if logs is None:
            return
        loss = logs.get("loss")
        lr   = logs.get("learning_rate")
        if _is_num(loss):
            mlflow.log_metric("train_loss", loss, step=state.global_step)
        if _is_num(lr):
            mlflow.log_metric("lr", lr, step=state.global_step)

    # ----- Validation -----
    def on_evaluate(self,
                    args: TrainingArguments,
                    state: TrainerState,
                    control: TrainerControl,
                    metrics=None, **kwargs):
        if metrics is None:
            return
        ev_loss = metrics.get("eval_loss")
        if _is_num(ev_loss):
            ppl = math.exp(ev_loss)
            self.perp_history.append((state.global_step, ppl))
            mlflow.log_metric("eval_loss", ev_loss, step=state.global_step)
            mlflow.log_metric("perplexity", ppl, step=state.global_step)

    # ----- End of training -----
    def on_train_end(self,
                     args: TrainingArguments,
                     state: TrainerState,
                     control: TrainerControl, **kwargs):
        if not self.perp_history:
            return
        steps, ppl = zip(*self.perp_history)
        plt.figure()
        plt.plot(steps, ppl, linewidth=2)
        plt.xlabel("Global step")
        plt.ylabel("Perplexity (val set)")
        plt.title("GPT-2-small on Shakespeare")
        out_png = Path(args.output_dir) / "perplexity.png"
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        mlflow.log_artifact(str(out_png))
