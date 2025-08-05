import os, time, gc, warnings, random, json
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoConfig
from peft import LoraConfig
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.cluster import KMeans
from teacher import query_teacher, extract_entities
from data.dataUtils import ExplorationMap, PriorityDock
from student import StudentModel
from peft import prepare_model_for_kbit_training
warnings.filterwarnings("ignore")
from pathlib import Path
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)
from followup import (
    chunk_answer,
    generate_followups_by_teacher,
)
import re
from coq import BroadAnswerDetector
from transformers.utils import logging
logging.set_verbosity_error()
from difflib import SequenceMatcher
from correct import is_bad_answer
import math
import pickle
import glob
import random
from typing import Optional, Tuple
print("running trainStudentn.py...")

QUESTION_PREFIX_RE = re.compile(r'^(what|how|why|when|where|which|who)\b', re.I)
TAG_PATTERN = re.compile(r"^\[[^\]]+\]\s*")  
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_CACHE, META_CACHE = [], []
QUESTION_SOURCES = {"followup", "cluster"}
CONCEPT_SOURCES  = {"seed", "ans_entity"}
VALID_TAGS = ["[definition]", "[categories]", "[function]", "[part_of]"]


seen_prompts = set()
ppl_cache = {}   
stu_version = 0 
all_qa_pairs = [] 
ENTITY_CACHE_DIR = "./EQCache"
os.makedirs(ENTITY_CACHE_DIR, exist_ok=True)

QA_PPL_CACHE_DIR = "./EntityQA"
os.makedirs(QA_PPL_CACHE_DIR, exist_ok=True)

def save_entity_qa(domain: str,
                   entity: str,
                   question: str,
                   stu_ans: str,
                   ppl: float):

    cache_path = os.path.join(QA_PPL_CACHE_DIR, f"{domain}_qa_ppl.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            store = json.load(f)
    else:
        store = {}

    ent_map = store.setdefault(entity, {})
    ent_map[question] = {"student_ans": stu_ans, "ppl": ppl}

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)

def load_cached_qa(domain: str, entity: str, question: str) -> Tuple[str | None, float | None]:

    cache_path = os.path.join(QA_PPL_CACHE_DIR, f"{domain}_qa_ppl.json")
    if not os.path.exists(cache_path):
        return None, None
    with open(cache_path, "r", encoding="utf-8") as f:
        store = json.load(f)
    ent_map = store.get(entity)
    if ent_map is None:
        return None, None
    rec = ent_map.get(question)
    if rec is None:
        return None, None
    return rec.get("student_ans"), rec.get("ppl")

def load_cached_entity(domain: str, entity: str) -> Optional[List[str]]:

    cache_path = os.path.join(ENTITY_CACHE_DIR, f"{domain}.json")
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    raw = cache.get(entity)
    if raw is None:
        return None

    tag_pattern = re.compile(r"^\[[^\]]+\]\s*")
    lines = [
        tag_pattern.sub("", ln.strip())
        for ln in re.split(r"[\r\n]+", raw)
        if ln.strip()
    ]
    return lines if lines else None

def save_to_entity_cache(domain: str, entity: str, question: str):
    cache_path = os.path.join(ENTITY_CACHE_DIR, f"{domain}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}

    if entity not in cache:
        cache[entity] = question
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)


TEACHER_CACHE_DIR = "./TeacherAns"
os.makedirs(TEACHER_CACHE_DIR, exist_ok=True)

def load_teacher_answer(domain: str, prompt: str) -> str | None:
    path = os.path.join(TEACHER_CACHE_DIR, f"{domain}_teacher.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        store = json.load(f)
    return store.get(prompt)

def save_teacher_answer(domain: str, prompt: str, answer: str):
    path = os.path.join(TEACHER_CACHE_DIR, f"{domain}_teacher.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            store = json.load(f)
    else:
        store = {}
    store[prompt] = answer
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)
        
def clean_candidate_label(s: str) -> str:
    s = s.strip().strip('“”"\'`')
    s = re.sub(r'[?？]+$', '', s)
    s = re.sub(r'^(what\s+is|what\s+are|define|explain|describe)\s+', '', s, flags=re.I)
    s = re.sub(r'\s+', ' ', s)
    return s

def is_question_like(s: str) -> bool:
    return bool(QUESTION_PREFIX_RE.match(s.lower()))

def safe_add_entity(exploration, raw_label: str, source: str, from_entity: str | None = None):
    c = clean_candidate_label(raw_label)
    if not c:
        return False
    if is_question_like(c):
        return False
    if exploration.is_explored(c):
        return False
    exploration.add_to_explore(c, source=source, from_entity=from_entity)
    return True
    
def get_root_nodes(domain: str,
                   teacher_model: str = "gpt-4.1",
                   n_roots: int = 30) -> List[str]:
    PROMPT = (
        f"Please generate a list of {n_roots} mutually exclusive, "
        f"non-overlapping root-level topics in the {domain} domain. "
        f"Return ONLY the concept names, separated by commas."
    )
    raw = query_teacher(PROMPT, domain=domain, model=teacher_model,
                        max_new_tokens=256)
    items = re.split(r"[,\n]+", raw)
    roots = [it.strip(" *-0123456789.") for it in items if it.strip()]
    uniq = []
    for r in roots:
        if r and r.lower() not in {u.lower() for u in uniq}:
            uniq.append(r)
        if len(uniq) >= n_roots:
            break
    return uniq

def add_qa(store: List[dict], 
           prompt: str, 
           teacher_answer: str, 
           student_answer: str,
           tag: str):
    for qa in store:
        if qa["prompt"] == prompt:
            if tag not in qa["meta"]:
                qa["meta"].append(tag)
            return  

    seen_prompts.add(prompt)
    store.append({
        "prompt": prompt,
        "teacher_answer": teacher_answer,
        "student_answer": student_answer,
        "meta": [tag]
    })

def sparse_cluster_questions(exploration: ExplorationMap, k: int = 30, top_n: int = 3) -> List[str]:
    n = len(EMBED_CACHE)
    if n < 2 * top_n:
        return [], {"n": n, "k": 0}

    k = min(k, max(2, int(np.sqrt(n))))
    k = min(k, n)
    if k < 2:
        return [], {"n": n, "k": k}

    X = np.vstack(EMBED_CACHE)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)
    counts = np.bincount(labels, minlength=k)
    sparse_ids = np.argsort(counts)[:top_n]  

    res = []
    for cid in sparse_ids:
        idxs = np.where(labels == cid)[0]
        center = X[idxs].mean(0, keepdims=True)
        rep = idxs[((X[idxs] - center) ** 2).sum(1).argmax()]
        q = META_CACHE[rep]["question"]

        if q not in exploration and not exploration.is_explored(q):
            res.append(q)
    return res, {"n": n, "k": k}


def cache_answer(ans: str, q: str, ent: str):
    v = EMBED_MODEL.encode(ans, normalize_embeddings=True)
    EMBED_CACHE.append(v)
    META_CACHE.append({"answer": ans, "question": q, "entity": ent})


def select_next(exploration, student, mix_prob=0.25, domain="medical") -> dict | None:
    queue = exploration.peek_all()
    if not queue:
        return None

    concept_entities = [e for e in queue if exploration.get_source(e) in CONCEPT_SOURCES]
    existing_questions = [e for e in queue if exploration.get_source(e) in QUESTION_SOURCES]

    def _fill_cache(q, p):
        if q not in ppl_cache and p is not None:
            ppl_cache[q] = p

    if existing_questions and concept_entities and random.random() < mix_prob:
        stu_ans_list, ppls_list, need_gen = [], [], []
        for q in existing_questions:
            a_cached, p_cached = load_cached_qa(domain, q, q)
            if a_cached is not None and p_cached is not None:
                stu_ans_list.append(a_cached)
                ppls_list.append(p_cached)
                _fill_cache(q, p_cached)
            else:
                need_gen.append(q)

        if need_gen:
            new_ans = student.generate(need_gen, max_new_tokens=512, domain=domain)
            new_ppl = student.batch_compute_ppl(need_gen, new_ans, max_len=512, micro_bs=8)
            for q_txt, a_txt, pval in zip(need_gen, new_ans, new_ppl):
                save_entity_qa(domain, q_txt, q_txt, a_txt, pval)
                stu_ans_list.append(a_txt)
                ppls_list.append(pval)
                _fill_cache(q_txt, pval)

        assert len(stu_ans_list) == len(existing_questions)
        assert len(ppls_list) == len(existing_questions)
        idx = int(np.argmax(ppls_list))
        q = existing_questions[idx]
        return {
            "entity": q,
            "prompt": q,
            "source": exploration.get_source(q),
            "ppl": ppls_list[idx],
            "templated": False,
            "stu_ans": stu_ans_list[idx],
        }
    elif concept_entities:
        ent2qtext = {}
        for ent in concept_entities:
            cached_qs = load_cached_entity(domain, ent)
            if cached_qs:                        
                ent2qtext[ent] = cached_qs
            else:                                 
                q = student.ask_questions(domain, [ent]).get(ent)
                if isinstance(q, dict):
                    q = "\n".join(f"[{k}] {v}" for k, v in q.items())
                if q:
                    save_to_entity_cache(domain, ent, q)
                    cached_qs = [
                        TAG_PATTERN.sub("", ln.strip())
                        for ln in re.split(r"[\r\n]+", q)
                        if ln.strip()
                    ]
                    print(f"\nEntity：{ent}")
                    for i, line in enumerate(cached_qs, 1):
                        print(f"{VALID_TAGS[i-1]}: {line}")
                    ent2qtext[ent] = cached_qs

        cand_pairs = []
        for ent, qlist in ent2qtext.items():
            for qtext in qlist:
                if not qtext or len(qtext.split()) < 4:
                    continue
                cand_pairs.append((ent, qtext))

        if cand_pairs:
            prompts = [q for _, q in cand_pairs]
            stu_ans, ppls = [], []
            need_gen = []
            for (ent, q_txt) in cand_pairs:
                a_cached, p_cached = load_cached_qa(domain, ent, q_txt)
                if a_cached is not None and p_cached is not None:
                    stu_ans.append(a_cached)
                    ppls.append(p_cached)
                    _fill_cache(q_txt, p_cached)
                else:
                    need_gen.append(q_txt)

            if need_gen:
                gen_ans = student.generate(need_gen, max_new_tokens=512, domain=domain)
                gen_ppl = student.batch_compute_ppl(need_gen, gen_ans, max_len=512, micro_bs=8)
                gen_iter = iter(zip(gen_ans, gen_ppl))
                for i, (ent, q_txt) in enumerate(cand_pairs):
                    if q_txt in need_gen:
                        a_txt, pval = next(gen_iter)
                        save_entity_qa(domain, ent, q_txt, a_txt, pval)
                        stu_ans.append(a_txt)
                        ppls.append(pval)
                        _fill_cache(q_txt, pval)

            assert len(stu_ans) == len(prompts)
            idx = int(np.argmax(ppls))
            ent, pr = cand_pairs[idx]
            return {
                "entity": ent,
                "prompt": pr,
                "source": exploration.get_source(ent),
                "ppl": ppls[idx],
                "templated": True,
                "stu_ans": stu_ans[idx]
            }

    elif existing_questions:
        stu_ans = student.generate(existing_questions, max_new_tokens=512, domain=domain)
        ppls = student.batch_compute_ppl(existing_questions, stu_ans, max_len=512, micro_bs=8)
        for q_txt, a_txt, pval in zip(existing_questions, stu_ans, ppls):
            save_entity_qa(domain, q_txt, q_txt, a_txt, pval)
        idx = int(np.argmax(ppls))
        q = existing_questions[idx]
        return {
            "entity": q,
            "prompt": q,
            "source": exploration.get_source(q),
            "ppl": ppls[idx],
            "templated": False,
            "stu_ans": stu_ans[idx]
        }

    return None

def run_qa(student, teacher_model, prompt, domain):
    cached = load_teacher_answer(domain, prompt)
    if cached:
        print("[TeacherCache] hit")
        return cached

    tea_ans = query_teacher(prompt, domain=domain,
                            max_new_tokens=1024,
                            model=teacher_model)
    save_teacher_answer(domain, prompt, tea_ans)
    return tea_ans

def expand_entities(teacher_answer, parent, exploration, domain, teacher_model,
                    max_terms=20, max_new=20):
    ents = extract_entities(
        text=teacher_answer,
        domain=domain,
        model=teacher_model,
        max_terms=max_terms
    )
    added = 0
    for ne in ents:
        if added >= max_new:
            break
        if safe_add_entity(exploration, ne, source="ans_entity", from_entity=parent):
            added += 1
    print(f"[Entity Expansion] from: {parent}, added: {added}, total explored: {len(exploration._visited)}")
    return added

def maybe_generate_followups(teacher_answer, 
                             parent_entity, 
                             exploration,
                             detector, 
                             domain, 
                             teacher_model,
                             enable_followup: bool = True,
                             immediate=True, 
                             max_q=3,
                             min_len=200):
    if len(teacher_answer.split()) < min_len:
        return 0, 0
    if not detector.is_broad(teacher_answer, verbose=True):
        return 0, 0
    summary = chunk_answer(teacher_answer)
    follow_qs = generate_followups_by_teacher(summary, max_q=max_q)
    gen = len(follow_qs)
    executed = 0
    if not enable_followup:
        print("[COQ] Follow‑up generation disabled by flag.")
        return gen, executed
    if immediate:
        for fq in follow_qs:
            fq_ans = query_teacher(fq, domain=domain, max_new_tokens=1024, model=teacher_model)
            add_qa(all_qa_pairs, fq, fq_ans, None, "cot")
            expand_entities(fq_ans, parent_entity, exploration, domain, teacher_model)
            safe_add_entity(exploration, fq, source="followup", from_entity=parent_entity)
            executed += 1
    else:
        for fq in follow_qs:
            safe_add_entity(exploration, fq, source="followup", from_entity=parent_entity)
    print(f"[COQ] Follow-up generated: {gen}, executed: {executed}")
    return gen, executed


def main(
    STUDENT_BASE: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    domain: str = "medical",
    MAX_QUERIES: int = 100,
    CKPT_FREQ: int = 20,          
    CLUSTER_FREQ: int = 25,
    LORA_BATCH_SIZE: int = 4,
    LORA_EPOCHS: int = 1,
    LORA_LR: float = 2e-4,
    STUDENT_DEVICE: str = "cuda:0",
    Teacher_model: str = "gpt-4.1",
    broad_answer_thresh: float = 0.38,
    sim_threshold: float = 0.6,
    with_cluster: bool = True,
    with_coq: bool = True,
    with_followup: bool = True,
    RESUME_STEP: int | None = None,
    RESUME_CKPT: str | None = None,
):
    print("Running main function...")

    # ---- Ablation flags -------------------------------------------------
    global QUESTION_SOURCES
    QUESTION_SOURCES = set(QUESTION_SOURCES)      # make a local mutable copy
    if not with_cluster:
        QUESTION_SOURCES.discard("cluster")
    if not with_followup:
        QUESTION_SOURCES.discard("followup")
    # --------------------------------------------------------------------

    ckpt_path = None
    if RESUME_STEP is not None:
        ckpt_path = f"checkpoint/{domain}_checkpoint_step{RESUME_STEP}.pkl"
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint for step {RESUME_STEP} not found: {ckpt_path}")
    elif RESUME_CKPT:
        if RESUME_CKPT.lower() == "latest":
            pattern = os.path.join("checkpoint", f"{domain}_checkpoint_step*.pkl")
            ckpts = glob.glob(pattern)
            if not ckpts:
                raise FileNotFoundError(f"No checkpoints matching pattern {pattern}")
            import re
            ckpts.sort(key=lambda p: int(re.search(r"_step(\d+)\.pkl$", p).group(1)), reverse=True)
            ckpt_path = ckpts[0]
        else:
            ckpt_path = RESUME_CKPT
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"Specified checkpoint file not found: {ckpt_path}")
    structured_logs = []
    dock = PriorityDock()
    detector = BroadAnswerDetector(
        model_name="all-MiniLM-L6-v2", 
        sim_thresh=broad_answer_thresh
    )
    base = AutoModelForCausalLM.from_pretrained(
        STUDENT_BASE, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_BASE)
    tokenizer.pad_token = tokenizer.eos_token
    student = StudentModel(base, tokenizer) 
   
    exploration = None
    start_step = 0
    if ckpt_path:
        print(f"[Resume] loading {ckpt_path}")
        with open(ckpt_path, "rb") as f:
            state = pickle.load(f)
        start_step   = state["step"]
        exploration  = state["exploration"]
        EMBED_CACHE.extend(state["EMBED_CACHE"])
        META_CACHE.extend(state["META_CACHE"])
        seen_prompts.update(state["seen_prompts"])
        all_qa_pairs.extend(state["all_qa_pairs"])
        ppl_cache.update(state.get("ppl_cache", {}))
        print(f"[Resume] restored step={start_step}, total QA={len(all_qa_pairs)}")

    if exploration is None:
        exploration = ExplorationMap()
        root_list = get_root_nodes(domain, teacher_model=Teacher_model, n_roots=30)
        print(f"[Init] fetched {len(root_list)} root nodes for '{domain}': {root_list[:30]}")
        for r in root_list:
            exploration.add_to_explore(r, source="seed")

    # ---- Fallback: resume only from latest QA JSON (if no .pkl given) ----
    if start_step == 0:
        json_snapshots = sorted(
            glob.glob(f"checkpoint/{domain}_all_qa_pairs*.json"),
            key=os.path.getmtime,
            reverse=True
        )
        if json_snapshots:
            latest_json = json_snapshots[0]
            print(f"[Resume] found QA snapshot {latest_json}")
            try:
                with open(latest_json, "r", encoding="utf-8") as f:
                    loaded_pairs = json.load(f)
                if isinstance(loaded_pairs, list) and loaded_pairs:
                    all_qa_pairs.extend(loaded_pairs)
                    for qa in loaded_pairs:
                        seen_prompts.add(qa["prompt"])
                    start_step = len(all_qa_pairs)
                    print(f"[Resume] restored {start_step} QA pairs from snapshot")
            except Exception as e:
                print(f"[Resume] failed to load snapshot: {e}")


    print("Starting exploration...")
    qa_buffer: List[Dict] = []
    step = start_step
    while step < MAX_QUERIES:
        sel = select_next(exploration, student, 0.5, domain=domain)
        if sel is None:
            print("No more entities/questions.")
            break

        gen, execd = 0, 0  

        entity  = sel["entity"]
        prompt  = sel["prompt"]
        source  = sel["source"]
        ppl_val = sel["ppl"]
        stu_ans = sel["stu_ans"]

        exploration.pop_specific(entity)
        if exploration.is_explored(entity):
            continue

        print(f"\n[{step+1}/{MAX_QUERIES}] Node ➜ {entity} (source={source}, templated={sel['templated']})")

        tic = time.time()
        tea_ans = run_qa(student, Teacher_model, prompt, domain)
        print(f"[Exploration] Remaining in queue: {len(exploration._queue)} | Explored: {len(exploration._visited)}")
        print(f"[Branch Info] Templated={sel['templated']} | Source={source}")

        qa_tag = "main" if source in CONCEPT_SOURCES else source
        if "q_tag" in sel:
            qa_tag = f"{qa_tag}:{sel['q_tag']}"
        add_qa(all_qa_pairs, prompt, tea_ans, stu_ans, qa_tag)

        if is_bad_answer(stu_ans, tea_ans, sim_threshold):
            print("[Correction] Triggered correction QA due to bad student answer.")
            print("Bad student answer -> correction")
            corr_prompt = (
                f"The student's answer was:\n«{stu_ans}»\n"
                f"The correct answer is:\n«{tea_ans}»\n\n"
                "Please kindly explain where the student was wrong and provide the correct answer again."
            )
            # print(f"[Correction] question is: {corr_prompt}")
            corr_ans = query_teacher(corr_prompt, domain=domain, max_new_tokens=1024, model=Teacher_model)
            add_qa(all_qa_pairs, corr_prompt, corr_ans, None, "correction")

        cache_answer(tea_ans, prompt, entity)
        exploration.mark_explored(entity)

        if source in CONCEPT_SOURCES or source == "followup":
            added_cnt = expand_entities(
                teacher_answer=tea_ans,
                parent=entity,
                exploration=exploration,
                domain=domain,
                teacher_model=Teacher_model,
                max_terms=20,
                max_new=20
            )
            print(f"[Entities] added {added_cnt}")

        if with_coq and source in CONCEPT_SOURCES:
            gen, execd = maybe_generate_followups(
                teacher_answer=tea_ans,
                parent_entity=entity,
                exploration=exploration,
                detector=detector,
                domain=domain,
                teacher_model=Teacher_model,
                enable_followup=with_followup,
                immediate=False,
                max_q=3,
                min_len=220
            )
            if gen:
                print(f"[Follow-up] generated={gen} executed={execd}")

        if with_cluster and CLUSTER_FREQ > 0 and (step + 1) % CLUSTER_FREQ == 0:
            extra_qs, stat = sparse_cluster_questions(exploration)
            queued = 0
            for cq in extra_qs:
                if exploration.is_explored(cq):
                    continue
                if safe_add_entity(exploration, cq, source="cluster"):
                    queued += 1
            print(f"[Cluster] queued {queued}/{len(extra_qs)} from {stat['k']} clusters")
            print(f"[COQ] Cluster trigger executed with {len(extra_qs)} extra questions.")

        step += 1
        if CKPT_FREQ > 0 and (step % CKPT_FREQ) == 0:
            os.makedirs("checkpoint", exist_ok=True)
            with open(f"checkpoint/{domain}_all_qa_pairs_step{step}_{len(all_qa_pairs)}.json",
                      "w", encoding="utf-8") as f:
                json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)
            ckpt_state = {
                "step": step,
                "exploration": exploration,
                "EMBED_CACHE": EMBED_CACHE,
                "META_CACHE": META_CACHE,
                "seen_prompts": list(seen_prompts),
                "all_qa_pairs": all_qa_pairs,
                "ppl_cache": ppl_cache,
            }
            with open(f"checkpoint/{domain}_checkpoint_step{step}.pkl", "wb") as f:
                pickle.dump(ckpt_state, f)
            print(f"[Checkpoint] Saved full state to checkpoint/{domain}_checkpoint_step{step}.pkl")
            
        structured_logs.append({
            "step": step + 1,
            "entity": entity,
            "source": source,
            "templated": sel["templated"],
            "prompt": prompt,
            "student_answer": stu_ans,
            "teacher_answer": tea_ans,
            "ppl": ppl_val if not math.isnan(ppl_val) else None,
            "is_bad_answer": is_bad_answer(stu_ans, tea_ans, sim_threshold),
            "entities_added": added_cnt if source in CONCEPT_SOURCES or source == "followup" else 0,
            "followup_generated": gen if source in CONCEPT_SOURCES else 0,
            "followup_executed": execd if source in CONCEPT_SOURCES else 0,
            "time_cost_sec": round(time.time() - tic, 2)
        })
        ppl_cache[prompt] = ppl_val if not math.isnan(ppl_val) else None
        print(f"Done in {time.time() - tic:.1f}s | PPL={ppl_val if not math.isnan(ppl_val) else '—'}")
    
    cache_size = getattr(getattr(student, "qgen", None), "cache_size", None)
    if cache_size is not None:
        print(f"[Debug] Dynamic question cache size: {cache_size} entities")
    else:
        print("[Debug] Dynamic question generation completed (no external cache).")

    os.makedirs("checkpoint", exist_ok=True)
    with open(f"checkpoint/{domain}_all_qa_pairs_{MAX_QUERIES}_{len(all_qa_pairs)}.json",\
              "w", encoding="utf-8") as f:
        json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)
    print(f"Collected {len(all_qa_pairs)} QA pairs")
    with open("results/ppl_cache.json", "w") as f:
        json.dump(ppl_cache, f, indent=2)
    with open(f"results/{domain}_structured_log.json", "w", encoding="utf-8") as f:
        json.dump(structured_logs, f, ensure_ascii=False, indent=2)

    
    final_dir = f"logs/{domain}_student_final_{MAX_QUERIES}_{len(all_qa_pairs)}_gemma"
    student.finetune_on_pairs(
        qa_pairs  = all_qa_pairs,
        tokenizer = tokenizer,
        out_dir   = final_dir,
        batch_size= LORA_BATCH_SIZE,
        epochs    = LORA_EPOCHS,
        lr        = LORA_LR,
    )
    print("student saved to", final_dir)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)