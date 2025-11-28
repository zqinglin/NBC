import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from thalamus import KAB_Thalamus


from transformers import BitsAndBytesConfig


class NeuroBayesianAgent:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct", slots=None, lora_r=16, quantization="16bit", force_download=False, resume_download=True, cache_dir=None, dtype="bf16"):
        if slots is None:
            slots = ["logic", "coder", "creative", "persona", "working"]
        self.slots = list(slots)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, force_download=force_download, resume_download=resume_download, cache_dir=cache_dir)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        load_args = {"force_download": force_download, "resume_download": resume_download}
        if cache_dir is not None:
            load_args.update({"cache_dir": cache_dir})
        if str(quantization).lower() == "4bit":
            bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
            load_args.update({"quantization_config": bnb_cfg, "device_map": "auto"})
            base = AutoModelForCausalLM.from_pretrained(model_name, **load_args)
        else:
            dt = torch.bfloat16 if str(dtype).lower() == "bf16" else torch.float16
            load_args.update({"dtype": dt, "device_map": None})
            base = AutoModelForCausalLM.from_pretrained(model_name, **load_args)
            base.to(self.device)
        base.train(False)
        for p in base.parameters():
            p.requires_grad = False
        lcfg = LoraConfig(r=lora_r, lora_alpha=32, lora_dropout=0.05, task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(base, lcfg)
        self.peft_config = lcfg
        existing_adapters = getattr(model, "peft_config", {})
        # 物理上初始化 LoRA 矩阵（Random Init）并创建各功能插槽
        for s in self.slots:
            if s not in existing_adapters:
                model.add_adapter(s, self.peft_config)
        model.set_adapter(self.slots[0])
        self.model = model
        if getattr(self.model, "generation_config", None) is not None:
            gc = self.model.generation_config
            if getattr(gc, "pad_token_id", None) is None:
                gc.pad_token_id = self.tokenizer.eos_token_id
            if getattr(gc, "eos_token_id", None) is None:
                gc.eos_token_id = self.tokenizer.eos_token_id
            if getattr(gc, "bos_token_id", None) is None and self.tokenizer.bos_token_id is not None:
                gc.bos_token_id = self.tokenizer.bos_token_id
        self.shadow_weights = {}
        for s in self.slots:
            self.model.set_adapter(s)
            try:
                sd = self.model.get_adapter_state_dict(s)
            except Exception:
                full_sd = self.model.state_dict()
                sd = {k: v for k, v in full_sd.items() if "lora_" in k}
            self.shadow_weights[s] = {k: v.detach().clone().cpu() for k, v in sd.items()}
        self.model.set_adapter(self.slots[0])
        emb_dim = int(self.model.get_input_embeddings().embedding_dim)
        anchors = {}
        prompts = {
            "logic": "Reason about rules and logical constraints",
            "coder": "Write and debug software code",
            "creative": "Compose imaginative and diverse content",
            "persona": "Adopt a consistent helpful assistant persona",
            "working": "Temporary working memory for current conversation"
        }
        for s in self.slots:
            ids = self.tokenizer(prompts.get(s, s), return_tensors="pt").to(self.device)["input_ids"]
            embs = self.model.get_input_embeddings()(ids)
            anchors[s] = embs.mean(dim=1).detach().cpu().view(-1)
        self.thalamus = KAB_Thalamus(self.slots, emb_dim, anchors=anchors)
        print("Agent initialized successfully")

    def _input_embedding(self, text):
        ids = self.tokenizer(text, return_tensors="pt").to(self.device)["input_ids"]
        embs = self.model.get_input_embeddings()(ids)
        return embs.mean(dim=1).detach().cpu().view(-1)

    def _loss(self, inputs):
        out = self.model(**inputs, labels=inputs["input_ids"]) 
        return out.loss

    def _active_params(self, slot):
        ps = []
        for n, p in self.model.named_parameters():
            use = ("lora_" in n) and (slot in n or ("." + slot) in n)
            p.requires_grad = use
            if use:
                ps.append(p)
        return ps

    def wake_act_learn(self, user_input, ttt_steps=1, surprise_threshold=1.5, lr=5e-4, max_new_tokens=128, sample=False, top_p=0.9, temperature=1.0):
        x = self._input_embedding(user_input)
        slot = self.thalamus.route(x)
        self.model.set_adapter(slot)
        enc = self.tokenizer(user_input, return_tensors="pt", padding=True).to(self.device)
        self.model.eval()
        with torch.no_grad():
            loss_before = self._loss(enc)
        reward = 0.0
        if float(loss_before.item()) > float(surprise_threshold):
            params = self._active_params(slot)
            opt = torch.optim.AdamW(params, lr=lr)
            self.model.train(True)
            for _ in range(ttt_steps):
                opt.zero_grad(set_to_none=True)
                loss = self._loss(enc)
                loss.backward()
                opt.step()
            self.model.eval()
            with torch.no_grad():
                loss_after = self._loss(enc)
            reward = max(0.0, min(1.0, float(loss_before.item() - loss_after.item())))
        else:
            loss_after = loss_before
        self.thalamus.update_belief(slot, reward)
        gen_inp = self.tokenizer(user_input, return_tensors="pt").to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(**gen_inp).logits[:, -1, :]
            if not torch.isfinite(logits).all():
                sample = False
        gen_ids = self.model.generate(**gen_inp, max_new_tokens=max_new_tokens, do_sample=sample, top_p=top_p, temperature=temperature)
        text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        return {"slot": slot, "loss_before": float(loss_before.item()), "loss_after": float(loss_after.item()), "reward": reward, "text": text}

    def sleep_consolidate(self, decay=0.1):
        for s in self.slots:
            curr = self.model.get_adapter_state_dict(s)
            curr_cpu = {k: v.detach().cpu() for k, v in curr.items()}
            if s == "working":
                zero_state = {k: torch.zeros_like(v) for k, v in curr_cpu.items()}
                self.model.load_state_dict(zero_state, strict=False)
                self.shadow_weights[s] = {k: torch.zeros_like(v) for k, v in curr_cpu.items()}
            else:
                sh = self.shadow_weights.get(s, {k: torch.zeros_like(v) for k, v in curr_cpu.items()})
                upd = {}
                for k in curr_cpu:
                    upd[k] = (1.0 - decay) * sh[k] + decay * curr_cpu[k]
                self.shadow_weights[s] = upd
                self.model.load_state_dict(upd, strict=False)
