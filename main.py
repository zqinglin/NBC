import argparse
from agent import NeuroBayesianAgent


def run_once(agent, prompt, ttt_steps, threshold, lr, max_new_tokens):
    out = agent.wake_act_learn(prompt, ttt_steps=ttt_steps, surprise_threshold=threshold, lr=lr, max_new_tokens=max_new_tokens)
    print(out["slot"], out["loss_before"], out["loss_after"], out["reward"])
    print(out["text"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--quant", default="4bit")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--force_download", action="store_true")
    p.add_argument("--resume_download", action="store_true")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--sample", action="store_true")
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--ttt_steps", type=int, default=1)
    p.add_argument("--threshold", type=float, default=1.5)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--sleep", action="store_true")
    p.add_argument("--prompt", default="Hello")
    args = p.parse_args()
    agent = NeuroBayesianAgent(model_name=args.model, quantization=args.quant, force_download=args.force_download, resume_download=args.resume_download, cache_dir=args.cache_dir, dtype=args.dtype)
    out = agent.wake_act_learn(args.prompt, ttt_steps=args.ttt_steps, surprise_threshold=args.threshold, lr=args.lr, max_new_tokens=args.max_new_tokens, sample=args.sample, top_p=args.top_p, temperature=args.temperature)
    print(out["slot"], out["loss_before"], out["loss_after"], out["reward"])
    print(out["text"])
    if args.sleep:
        agent.sleep_consolidate()


if __name__ == "__main__":
    main()
