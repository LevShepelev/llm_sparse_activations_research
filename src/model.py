# src/model.py
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def load_tokenizer_and_model(model_name: str,
                             add_special_tokens: dict,
                             resize: bool,
                             from_scratch: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add PAD or other custom tokens *before* model is created
    special_tokens = {k: v for k, v in add_special_tokens.items() if v}
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)

    if from_scratch:
        # Build an empty config (same size as GPT-2-small)
        cfg = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(cfg)  # ← **random weights**
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    if special_tokens and resize:          # match new vocab length
        model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model
