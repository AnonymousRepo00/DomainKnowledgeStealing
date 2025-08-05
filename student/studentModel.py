import os, gc, math, copy, random, datetime as dt
from pathlib import Path
from typing import Dict

import torch
from transformers import (
    Trainer, TrainingArguments,
    DataCollatorForSeq2Seq
)
import re
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from data.dataUtils import QADataset
from typing import List, Any
import torch.nn.functional as F
try:                                   
    from peft.utils.save_and_load import (
        get_peft_state_maybe_zero_3,
        set_peft_model_state_dict,
    )
except ImportError:
    try:                                
        from peft.utils.save_and_load import (
            get_peft_state_maybe_zero_3,
            set_peft_model_state_dict,
        )
    except ImportError:                
        from peft import (
            get_peft_model_state_dict as get_peft_state_maybe_zero_3,
            set_peft_model_state_dict,
        )
        
from .askques import LLaMAQuestionGenerator, GenerationConfig

DOMAIN_CONFIGS: Dict[str, Dict[str, Any]] = {
    "medical": {
        "role": "medical expert",
    },
    "financial": {
        "role": "financial expert",
    },
}
CHK_DIR = Path("checkpoints")
CHK_DIR.mkdir(exist_ok=True)

def sanity_fail(text: str) -> bool:
    too_long = len(text) > 1000
    nonsense = bool(
        re.search(r"(O{10,}|aalborg|odense|fetisch)", text, re.I)
    )
    return too_long or nonsense

def _first_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device

class StudentModel:
    def __init__(self, model, tokenizer):
        model.gradient_checkpointing_enable()
        peft_cfg = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=16, 
            lora_alpha=32, 
            lora_dropout=0.05
        )
        model = get_peft_model(model, peft_cfg)
        self.model = model.eval()
        self.tokenizer = tokenizer
        qgen_cfg = GenerationConfig(
            model_name=None,
            temperature=0.8,
            top_p=0.92,
            repetition_penalty=1.05,
            max_new_tokens=140,
            max_retries=3,
            max_tokens_per_question=40,
            enforce_seed=42,
            device=None,
            enable_paraphrase=False,
            verbose=False
        )
        self.qgen = LLaMAQuestionGenerator(qgen_cfg, model=self.model, tokenizer=self.tokenizer)
        self.adapters = ["lora"]
        self.q_cache = {}
    
    def ask_questions(self, domain: str, entities: List[str]):
        result = self.qgen.generate_for_entities(domain, entities)
        return result

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=64, domain="medical", micro_bs=4):
        cfg  = DOMAIN_CONFIGS.get(domain, {"role": "expert"})
        role = cfg["role"]
        

        if isinstance(prompt, list):
            answers = []
            for i in range(0, len(prompt), micro_bs):
                sub_prompts = prompt[i : i + micro_bs]

                sub_with_prefix = [
                    (
                        f"As a {role}, please explain the following question in detail "
                        "without adding any labels or numbering.\n"
                        "Answer the question below in full sentences, using professional "
                        "terminology and citing guidelines where appropriate."
                        f"Question: {p}\nResponse:"
                    )
                    for p in sub_prompts
                ]

                enc = self.tokenizer(
                    sub_with_prefix,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                enc = {k: v.to(_first_device(self.model)) for k, v in enc.items()}

                outs = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                decoded = self.tokenizer.batch_decode(
                    outs, skip_special_tokens=True
                )

                ans_batch = [
                    d[len(pr):].strip() for d, pr in zip(decoded, sub_with_prefix)
                ]
                answers.extend(ans_batch)

            return answers

    def add_new_adapter(self, name, r, alpha, dropout):
        cfg = LoraConfig(task_type="CAUSAL_LM",
                         inference_mode=False,
                         r=r, lora_alpha=alpha, lora_dropout=dropout)
        self.model.add_adapter(name, cfg)
        self.adapters.append(name)

    def _save_ckpt(self, tag: str) -> Path:
        path  = CHK_DIR / f"lora_{tag}.pt"
        state = get_peft_state_maybe_zero_3(self.model)  
        torch.save(state, path)
        return path

    def _load_ckpt(self, path: Path):
        state = torch.load(path, map_location="cpu")    
        set_peft_model_state_dict(self.model, state) 
    
    def safe_finetune_current_adapter(
        self,
        qa_pairs,
        epochs: int        = 1,
        batch_size: int    = 16,
        lr: float          = 5e-5,
        clip_grad_norm: float = 1.0,
        save_every_k: int  = 0 
    ) -> bool:
        pre_ckpt = self._save_ckpt(f"pre_{dt.datetime.now():%H%M%S}")
        ds   = QADataset(qa_pairs, self.tokenizer, max_length=1024)
        coll = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        targs = TrainingArguments(
            output_dir="tmp/lora",
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=lr,
            fp16=True,
            save_strategy="no",
            optim="adamw_8bit",
            report_to="none",
            max_grad_norm=clip_grad_norm,
            logging_steps=1
        )

        trainer = Trainer(self.model, args=targs,
                          train_dataset=ds, data_collator=coll)
        result = trainer.train()
        loss_val = float(result.training_loss)

        if math.isnan(loss_val) or math.isinf(loss_val):
            print("‼️  loss=nan/inf — rollback")
            self._load_ckpt(pre_ckpt)
            ok = False
        else:
            prompt_for_test = qa_pairs[0]["prompt"]
            gen_after_ft    = self.generate(prompt_for_test, max_new_tokens=96)
            if sanity_fail(gen_after_ft):
                print("‼️  Detected corrupted answer — rollback")
                self._load_ckpt(pre_ckpt)
                ok = False
            else:
                ok = True

        if ok and save_every_k:
            if (len(qa_pairs) // batch_size) % save_every_k == 0:
                self._save_ckpt(f"step{len(qa_pairs)}")

        del trainer; torch.cuda.empty_cache(); gc.collect()
        return ok



    def final_merge_and_save(self, out_dir):
        if hasattr(self.model, "merge_and_unload"):
            self.model = self.model.merge_and_unload()
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        print(f"Saved merged student to {out_dir}")

    @torch.no_grad()
    def batch_compute_ppl(
        self,
        prompts: List[str],
        answers: List[str] | None = None,
        max_len: int = 1024,
        micro_bs: int = 8
    ) -> List[float]:
        device, tok = _first_device(self.model), self.tokenizer
        answers = answers or [""] * len(prompts)
        all_ppl = []

        for i in range(0, len(prompts), micro_bs):
            ps = prompts[i:i + micro_bs]
            ans = answers[i:i + micro_bs]

            full_inputs = [p + tok.eos_token + a for p, a in zip(ps, ans)]
            enc = tok(
                full_inputs, return_tensors="pt",
                padding=True, truncation=True, max_length=max_len
            ).to(device)

            labels = enc["input_ids"].clone()

            if any(ans): 
                prompt_lens = [
                    len(tok(p + tok.eos_token).input_ids) - 1 for p in ps
                ]
                for row, plen in enumerate(prompt_lens):
                    labels[row, :plen] = -100
            else:
                labels[enc["attention_mask"] == 0] = -100

            outputs = self.model(**enc, labels=labels)
            loss = outputs.loss       
            ppl  = math.exp(loss.item())
            all_ppl.extend([ppl] * len(ps))

            del enc, outputs
            torch.cuda.empty_cache()

        return all_ppl
    
    
    def finetune_on_pairs(
        self,
        qa_pairs: List[Dict],
        tokenizer,
        out_dir: str,
        batch_size: int = 16,
        epochs: int = 3,
        lr: float = 2e-4,
    ):
        dataset  = QADataset(qa_pairs, tokenizer, max_length=256)
        collator = DataCollatorForSeq2Seq(tokenizer, model=self.model)

        train_args = TrainingArguments(
            output_dir              = Path(out_dir)/"tmp",
            per_device_train_batch_size = batch_size,
            num_train_epochs        = epochs,
            learning_rate           = lr,
            fp16                    = True,
            report_to               = "none",
            save_strategy           = "no",
        )
        trainer = Trainer(self.model, args=train_args,
                          train_dataset=dataset, data_collator=collator)
        trainer.train()
        print("Final LoRA training done")
        self.final_merge_and_save(out_dir)