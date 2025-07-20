import argparse, math, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', default='outputs/gpt2-shakespeare')
    ap.add_argument('--eval_path', default='data/raw/shakespeare_eval.txt')
    ap.add_argument('--block_size', type=int, default=512)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    text = Path(args.eval_path).read_text(encoding='utf-8')
    enc = tokenizer(text, return_tensors='pt')
    input_ids = enc['input_ids'][0]
    n_tokens = input_ids.size(0)
    stride = args.block_size
    nlls = []
    for i in range(0, n_tokens - stride, stride):
        chunk = input_ids[i:i+stride].unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(chunk, labels=chunk)
            nlls.append(out.loss * chunk.shape[1])
    ppl = math.exp(torch.stack(nlls).sum() / (len(nlls)*stride))
    print(f"Perplexity: {ppl:.2f}")

if __name__ == '__main__':
    main()