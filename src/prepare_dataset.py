from pathlib import Path
import argparse, re

def basic_clean(text: str) -> str:
    text = re.sub(r'\r\n?', '\n', text)
    return text.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/shakespeare.txt")
    ap.add_argument("--output", default="data/raw/shakespeare.txt")
    args = ap.parse_args()
    p = Path(args.input)
    cleaned = basic_clean(p.read_text(encoding="utf-8"))
    Path(args.output).write_text(cleaned, encoding="utf-8")
    print(f"Cleaned to {args.output}, {len(cleaned)} chars")

if __name__ == "__main__":
    main()
