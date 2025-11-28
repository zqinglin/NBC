import argparse
import os
from datasets import load_dataset, Dataset
from agent import NeuroBayesianAgent


DATASET_MAPPING = {
    "logic": {
        "path": "microsoft/orca-math-word-problems-200k",
        "prompt_col": "question",
        "response_col": "answer",
    },
    "coder": {
        "path": "nickrosh/Evol-Instruct-Code-80k-v1",
        "prompt_col": "instruction",
        "response_col": "output",
    },
    "persona": {
        "path": "Proj-Persona/Persona-Instruct",
        "prompt_col": "instruction",
        "response_col": "output",
    },
}


def format_examples(ds, prompt_col, response_col):
    out = []
    for ex in ds:
        p = ex.get(prompt_col, "")
        r = ex.get(response_col, "")
        if isinstance(r, list):
            if len(r) and isinstance(r[0], dict):
                r = "\n".join([str(m.get("content", "")) for m in r])
            else:
                r = "\n".join([str(x) for x in r])
        text = f"Instruction: {str(p)}\nResponse: {str(r)}"
        out.append(text)
    return out


def load_warmed_slots(agent, slots_dir):
    for s in agent.slots:
        path = os.path.join(slots_dir, s)
        if os.path.isdir(path):
            try:
                agent.model.load_adapter(path, s)
            except Exception:
                pass


def build_stream(num_each):
    items = []
    streams = []
    for s in ["logic", "coder", "persona"]:
        m = DATASET_MAPPING[s]
        ds = load_dataset(m["path"], split="train")
        ds = ds.select(range(min(num_each, len(ds))))
        texts = format_examples(ds, m["prompt_col"], m["response_col"]) 
        streams.append([(s, t) for t in texts])
    i = 0
    while True:
        done = True
        for stream in streams:
            if i < len(stream):
                items.append(stream[i])
                done = False
        if done:
            break
        i += 1
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots_dir", default="./saved_slots")
    ap.add_argument("--num_each", type=int, default=10)
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--quant", default="16bit")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--ttt_steps", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=1.5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    args = ap.parse_args()
    agent = NeuroBayesianAgent(model_name=args.model, quantization=args.quant, dtype=args.dtype)
    load_warmed_slots(agent, args.slots_dir)
    stream = build_stream(args.num_each)
    correct = 0
    total = 0
    loss_reds = []
    for idx, (gt, text) in enumerate(stream):
        out = agent.wake_act_learn(text, ttt_steps=args.ttt_steps, surprise_threshold=args.threshold, lr=args.lr, max_new_tokens=args.max_new_tokens)
        sel = out["slot"]
        total += 1
        if sel == gt:
            correct += 1
        loss_reds.append(out["loss_before"] - out["loss_after"]) 
        if (idx + 1) % 5 == 0:
            agent.sleep_consolidate()
    acc = 100.0 * correct / max(1, total)
    avg_red = sum(loss_reds) / max(1, len(loss_reds))
    print(f"Overall Routing Accuracy: {acc:.2f}%")
    print(f"Average Loss Reduction: {avg_red:.4f}")


if __name__ == "__main__":
    main()

