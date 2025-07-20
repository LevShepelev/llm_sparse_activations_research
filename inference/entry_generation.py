import argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', default='outputs/gpt2-shakespeare')
    ap.add_argument('--prompt', default='ROMEO:')
    ap.add_argument('--max_new_tokens', type=int, default=200)
    ap.add_argument('--temperature', type=float, default=0.5)
    ap.add_argument('--top_k', type=int, default=50)
    ap.add_argument('--top_p', type=float, default=0.95)
    ap.add_argument('--num_return_sequences', type=int, default=1)
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()

@torch.inference_mode()
def generate(model, tokenizer, prompt, max_new_tokens, temperature, top_k, top_p, num_return_sequences):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in gen_ids]

if __name__ == '__main__':
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    torch.manual_seed(args.seed)
    outputs = generate(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, args.top_k, args.top_p, args.num_return_sequences)
    for i, o in enumerate(outputs):
        print(f"===== Sample {i+1} ====={o}")