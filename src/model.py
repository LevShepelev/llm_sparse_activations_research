from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def load_tokenizer_and_model(model_name: str, add_special_tokens: dict, resize: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens = {k: v for k, v in add_special_tokens.items() if v}
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)

    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if special_tokens and resize:
        model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model, config
