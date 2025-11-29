import argparse
import os
import json
import random
import re
import ast
from datasets import load_dataset, Dataset
from agent import NeuroBayesianAgent


DATASET_MAPPING = {
    "logic": {
        "path": "microsoft/orca-math-word-problems-200k",
        "subset": None,
        "prompt_col": "question",
        "response_col": "answer",
    },
    "coder": {
        "path": "nickrosh/Evol-Instruct-Code-80k-v1",
        "subset": None,
        "prompt_col": "instruction",
        "response_col": "output",
    },
    "persona": {
        "path": "proj-persona/PersonaHub",
        "subset": "instruction",
        "prompt_col": "input persona",
        "response_col": "synthesized text",
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


class MixedDataset:
    def __init__(self, num_each, stream_len, seed=42):
        self.num_each = num_each
        self.stream_len = stream_len
        self.seed = seed
        self.items = []
        random.seed(seed)
        for s in ["logic", "coder", "persona"]:
            m = DATASET_MAPPING[s]
            if m.get("subset"):
                ds = load_dataset(m["path"], m["subset"], split="train")
            else:
                ds = load_dataset(m["path"], split="train")
            ds = ds.select(range(min(num_each, len(ds))))
            for ex in ds:
                p = ex.get(m["prompt_col"], "")
                r = ex.get(m["response_col"], "")
                if isinstance(r, list):
                    if len(r) and isinstance(r[0], dict):
                        r_text = "\n".join([str(mo.get("content", "")) for mo in r])
                    else:
                        r_text = "\n".join([str(x) for x in r])
                else:
                    r_text = str(r)
                text = f"Instruction: {str(p)}\nResponse: {r_text}"
                self.items.append((s, text, r_text))
        random.shuffle(self.items)
        if self.stream_len is not None:
            self.items = self.items[:self.stream_len]
    def __iter__(self):
        for it in self.items:
            yield it

class HeuristicEvaluator:
    def __init__(self):
        pass
    def _extract_number(self, s):
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
        if not nums:
            return None
        try:
            return float(nums[-1])
        except Exception:
            return None
    def eval_logic(self, pred_text, gt_text):
        p = self._extract_number(pred_text or "")
        g = self._extract_number(gt_text or "")
        if p is None or g is None:
            return False
        return abs(p - g) < 1e-6
    def eval_coder(self, pred_text):
        if not isinstance(pred_text, str) or len(pred_text) == 0:
            return False
        code = pred_text
        if "```" in pred_text:
            blocks = re.findall(r"```\s*(?:python|py)?\s*\n([\s\S]*?)```", pred_text, re.IGNORECASE)
            if not blocks:
                blocks = re.findall(r"```([\s\S]*?)```", pred_text, re.IGNORECASE)
            if blocks:
                code = blocks[-1]
        try:
            ast.parse(code)
            has_kw = ("def" in code) or ("return" in code)
            return has_kw
        except Exception:
            return False
    def eval_persona(self, pred_text):
        return isinstance(pred_text, str) and len(pred_text) > 0
    def evaluate(self, task_type, pred_text, gt_text=None):
        if task_type == "logic":
            return self.eval_logic(pred_text, gt_text)
        if task_type == "coder":
            return self.eval_coder(pred_text)
        if task_type == "persona":
            return self.eval_persona(pred_text)
        return False

def compute_bwt(history):
    out = {}
    for t in history:
        arr = history[t]
        if len(arr) >= 2:
            out[t] = arr[-1] - arr[-2]
        else:
            out[t] = 0.0
    return out


def run_lifelong(agent_args, slots_dir, output_dir, sessions=3):
    os.makedirs(output_dir, exist_ok=True)
    irr = []
    for sid in range(sessions):
        agent = NeuroBayesianAgent(**agent_args)
        load_warmed_slots(agent, slots_dir)
        tests = []
        tests.append(("coder", "请写一个 Python 函数，变量名以 nbc_ 开头"))
        tests.append(("logic", "计算 12+5，请用 JSON 格式输出"))
        ok_count = 0
        for t, tx in tests:
            res = agent.wake_act_learn(tx, ttt_steps=0, surprise_threshold=1e9, lr=0.0, max_new_tokens=64)
            txt = res["text"]
            if t == "coder":
                ok = ("nbc_" in txt)
            else:
                ok = ("{" in txt and "}" in txt)
            ok_count += 1 if ok else 0
        irr.append(ok_count / len(tests))
        data = []
        learn_batch = []
        learn_batch.append(("coder", "请写一个 Python 函数，变量名以 nbc_ 开头，计算数组和"))
        learn_batch.append(("logic", "计算 23+7，请用 JSON 格式输出，如 {\"answer\":30}"))
        for (t, tx) in learn_batch:
            out = agent.wake_act_learn(tx, ttt_steps=1, surprise_threshold=0.5, lr=2e-4, max_new_tokens=128)
            data.append(out)
        agent.sleep_consolidate()
        try:
            if os.path.isdir(slots_dir):
                for s in agent.slots:
                    sp = os.path.join(slots_dir, s)
                    os.makedirs(sp, exist_ok=True)
                    agent.model.save_pretrained(sp)
        except Exception:
            pass
    with open(os.path.join(output_dir, "metrics_evolution.json"), "w", encoding="utf-8") as wf:
        json.dump({"irr": irr}, wf, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots_dir", default="./saved_slots")
    ap.add_argument("--num_each", type=int, default=50)
    ap.add_argument("--stream_len", type=int, default=150)
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--quant", default="16bit")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--ttt_steps", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=1.5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--every_n", type=int, default=10)
    ap.add_argument("--output_dir", default="./bench_logs")
    ap.add_argument("--baseline_mode", default="multi-stream")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    anchor_data_map = {}
    for s in ["logic", "coder", "persona"]:
        m = DATASET_MAPPING[s]
        if m.get("subset"):
            ds0 = load_dataset(m["path"], m["subset"], split="train")
        else:
            ds0 = load_dataset(m["path"], split="train")
        ds0 = ds0.select(range(min(20, len(ds0))))
        anchor_data_map[s] = [str(ex.get(m["prompt_col"], "")) for ex in ds0]
    agent = NeuroBayesianAgent(model_name=args.model, quantization=args.quant, dtype=args.dtype, anchor_data_map=anchor_data_map)
    load_warmed_slots(agent, args.slots_dir)
    mixed = MixedDataset(args.num_each, args.stream_len)
    evaluator = HeuristicEvaluator()
    try:
        loaded = list(getattr(agent.model, "peft_config", {}).keys())
        print(f"[Config] baseline_mode={args.baseline_mode}")
        print(f"[Config] loaded_slots={loaded}")
    except Exception:
        pass
    # Zero-shot baseline FWT
    baseline_acc = {"logic": 0.0, "coder": 0.0, "persona": 0.0}
    baseline_count = {"logic": 0, "coder": 0, "persona": 0}
    for (t, tx, gt) in mixed.items:
        res = agent.wake_act_learn(tx, ttt_steps=0, surprise_threshold=1e9, lr=0.0, max_new_tokens=64)
        ok = evaluator.evaluate(t, res["text"], gt)
        baseline_acc[t] += 1.0 if ok else 0.0
        baseline_count[t] += 1
    for k in baseline_acc:
        baseline_acc[k] = baseline_acc[k] / max(1, baseline_count[k])
    ra_correct = 0
    ra_total = 0
    loss_reds = []
    hist_acc = {"logic": [], "coder": [], "persona": []}
    task_counts = {"logic": 0, "coder": 0, "persona": 0}
    task_acc_sums = {"logic": 0.0, "coder": 0.0, "persona": 0.0}
    task_ra_correct = {"logic": 0, "coder": 0, "persona": 0}
    step_logs_path = os.path.join(args.output_dir, "results_stream.jsonl")
    with open(step_logs_path, "w", encoding="utf-8") as wf:
        for idx, (gt, text, gt_ans) in enumerate(mixed):
            force_slot = "working" if args.baseline_mode == "single-stream" else None
            out = agent.wake_act_learn(text, ttt_steps=args.ttt_steps, surprise_threshold=args.threshold, lr=args.lr, max_new_tokens=args.max_new_tokens, active_slot=force_slot)
            sel = out["slot"]
            ra_total += 1
            if sel == gt:
                ra_correct += 1
                task_ra_correct[gt] += 1
            loss_reds.append(out["loss_before"] - out["loss_after"]) 
            ok = evaluator.evaluate(gt, out["text"], gt_ans)
            task_counts[gt] += 1
            task_acc_sums[gt] += 1.0 if ok else 0.0
            wf.write(json.dumps({"step": idx + 1, "task_type": gt, "routed_slot": sel, "loss_before": out["loss_before"], "loss_after": out["loss_after"], "is_correct": ok}) + "\n")
            if ((idx + 1) % args.every_n) == 0:
                for t in ["logic", "coder", "persona"]:
                    samples = [x for x in mixed.items if x[0] == t][:min(10, len([x for x in mixed.items if x[0] == t]))]
                    accs = []
                    for (_, tx, gt_val) in samples:
                        res = agent.wake_act_learn(tx, ttt_steps=0, surprise_threshold=1e9, lr=args.lr, max_new_tokens=64)
                        accs.append(1.0 if evaluator.evaluate(t, res["text"], gt_val) else 0.0)
                    hist_acc[t].append(sum(accs) / max(1, len(accs)))
            if ((idx + 1) % 5) == 0:
                agent.sleep_consolidate()
    bwt = compute_bwt(hist_acc)
    ra = 100.0 * ra_correct / max(1, ra_total)
    sdr = sum(loss_reds) / max(1, len(loss_reds))
    breakdown = {}
    for t in ["logic", "coder", "persona"]:
        acc_t = task_acc_sums[t] / max(1, task_counts[t])
        racc_t = 100.0 * task_ra_correct[t] / max(1, task_counts[t])
        breakdown[t] = {"acc": acc_t, "count": task_counts[t], "routing_acc": racc_t, "baseline_acc": baseline_acc[t], "gain": acc_t - baseline_acc[t]}
    summary = {"routing_accuracy": ra, "average_loss_reduction": sdr, "bwt": bwt, "hist_acc": hist_acc, "breakdown": breakdown, "baseline_overall": sum(baseline_acc.values())/max(1, len(baseline_acc))}
    with open(os.path.join(args.output_dir, "metrics_stream.json"), "w", encoding="utf-8") as wf:
        json.dump(summary, wf, ensure_ascii=False, indent=2)
    print(f"Overall Routing Accuracy: {ra:.2f}%")
    print(f"Average Loss Reduction: {sdr:.4f}")
    run_lifelong({"model_name": args.model, "quantization": args.quant, "dtype": args.dtype}, args.slots_dir, args.output_dir, sessions=3)


if __name__ == "__main__":
    main()

