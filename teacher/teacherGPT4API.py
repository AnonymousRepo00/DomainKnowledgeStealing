from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import torch
import re
import os
from openai import OpenAI
from datasets import load_dataset
import json
import logging
from typing import List, Dict, Any, Tuple, Optional

DOMAIN_CONFIGS: Dict[str, Dict[str, Any]] = {
    "medical": {
        "role": "medical expert",
        "extract_desc": "specific medical conditions, diseases, physiological concepts, anatomical structures, diagnostic modalities, or therapeutic interventions",
        "negative": ["risk factor", "symptom", "disease", "complication", "medical condition", "health issue"]
    },
    "financial": {
        "role": "financial expert",
        "extract_desc": "specific financial instruments, economic indicators, market structure concepts, regulatory entities, accounting standards, or transaction types",
        "negative": ["financial issue", "economic condition", "market trend", "investment"]
    },

}
api_key = "" # input your openai api key
if not api_key:
    raise ValueError("❌ OpenAI API key is missing. Please set `api_key` before creating the client.")

client = OpenAI(api_key=api_key)

def gpt_teacher(
    prompt: str,
    model: str = "gpt-4.1",
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

# query teacher to get answer for a question
def query_teacher(
    question: str,
    domain: str = "w",
    max_new_tokens: int = 128,
    do_sample: bool = False,
    model: str = "gpt-4.1"
) -> str:
    cfg = DOMAIN_CONFIGS.get(domain, {"role": "expert"})
    role = cfg["role"]
    prompt = (
        # f"{few_shot}"
        f"As a {role}, please explain the following question in detail without adding any labels or numbering.\n"
        "Answer the question below in full sentences, using professional terminology and citing guidelines where appropriate."
        f"Question: {question}\n"
        "Response:"
    )
    ans = gpt_teacher(prompt, model=model, temperature=0.0, max_tokens=max_new_tokens)
    return ans

ACRONYM_RE = re.compile(r'^[A-Z]{2,}(-[A-Z]{2,})?$')
WORD_SPLIT_RE = re.compile(r'[\s]+')


GENERIC_BASE = {
    "market", "markets", "investment", "investments", "issue", "issues",
    "trend", "trends", "condition", "conditions", "risk", "risks",
    "development", "improvement", "factor", "factors", "concept",
    "problem", "problems", "analysis", "approach", "method", "methods",
    "process", "processes", "strategy", "strategies"
}

STOP_START = {"the", "a", "an"}

ALLOWED_ABSTRACT_ENDINGS = {
    "liquidity", "solvency", "materiality", "jurisdiction", "regulation",
    "governance", "compliance", "enforceability", "accountability"
}

LOWER_CONNECTORS = {"and", "of", "for", "the", "in", "on", "with", "to", "by", "or"}

def normalize_entity_surface(ent: str) -> str:
    ent = ent.strip()
    if not ent:
        return ent
    if ACRONYM_RE.fullmatch(ent):
        return ent
    parts = ent.split()
    norm_parts = []
    for w in parts:
        raw = w
        if ACRONYM_RE.fullmatch(raw):
            norm_parts.append(raw)
            continue
        if "-" in raw:
            subs = raw.split("-")
            new_sub = []
            for sp in subs:
                if ACRONYM_RE.fullmatch(sp):
                    new_sub.append(sp)
                else:
                    lowsp = sp.lower()
                    if lowsp in LOWER_CONNECTORS:
                        new_sub.append(lowsp)
                    elif sp.islower():
                        new_sub.append(sp.capitalize())
                    else:
                        new_sub.append(sp)
            norm_parts.append("-".join(new_sub))
        else:
            low = raw.lower()
            if low in LOWER_CONNECTORS:
                norm_parts.append(low)
            elif raw.islower():
                norm_parts.append(raw.capitalize())
            else:
                norm_parts.append(raw)
    return " ".join(norm_parts)


def canonical_key(ent: str) -> str:
    return re.sub(r'\s+', ' ', ent.strip().lower())

def is_acronym_like(ent: str) -> bool:
    return bool(ACRONYM_RE.fullmatch(ent))


def contains_too_many_tokens(ent: str, max_tokens: int = 8) -> bool:
    return len(WORD_SPLIT_RE.split(ent.strip())) > max_tokens


def looks_sentence(ent: str) -> bool:
    return len(ent.split()) > 1 and ent.endswith(('.', '?'))


def has_forbidden_start(ent: str) -> bool:
    first = ent.split()[0].lower()
    return first in STOP_START


def is_generic(ent: str, cfg: Dict[str, Any]) -> bool:
    low = ent.lower()
    if low in GENERIC_BASE:
        return True
    for neg in cfg.get("negative", []):
        if low == neg.lower():
            return True
    if len(low) < 3 and not is_acronym_like(ent):
        return True
    if len(ent.split()) == 1 and low in {"market", "investment", "issue", "risk"}:
        return True
    return False


def is_redundant_variant(ent: str, existing: List[str]) -> bool:

    key = canonical_key(ent)
    for e in existing:
        ek = canonical_key(e)
        if key == ek:
            return True
        if key.startswith(ek + " ") or ek.startswith(key + " "):
            return True
    return False


def abstract_penalty(ent: str) -> float:
    low = ent.lower()
    if re.search(r'(ness|ment|tion|sion|ship|ity|ance|ence|ism|ology)$', low):
        if low not in ALLOWED_ABSTRACT_ENDINGS:
            return -0.25
    return 0.0


def score_entity(ent: str) -> float:
    score = 1.0
    tokens = ent.split()
    n = len(tokens)
    if 1 <= n <= 4:
        score += 0.4
    elif n > 6:
        score -= 0.5
    if is_acronym_like(ent):
        score += 0.05
    if any(char.isdigit() for char in ent):
        score -= 0.2
    score += abstract_penalty(ent)
    return score

def acronym_candidate(full: str) -> str:
    letters = [w[0] for w in full.split() if w and w[0].isalpha()]
    ac = "".join(letters).upper()
    return ac if len(ac) >= 2 else ""


def group_acronyms(entities: List[str]) -> List[List[str]]:

    groups: List[List[str]] = []
    used = set()
    for i, e in enumerate(entities):
        if i in used:
            continue
        ac = acronym_candidate(e)
        group = [e]
        for j, f in enumerate(entities):
            if j == i or j in used:
                continue
            if is_acronym_like(f) and f == ac:
                group.append(f)
                used.add(j)
            else:
                if is_acronym_like(e) and acronym_candidate(f) == e:
                    group.append(f)
                    used.add(j)
        used.add(i)
        groups.append(group)
    return groups


def build_prompt(text: str,
                 domain: str,
                 cfg: Dict[str, Any],
                 max_terms: int,
                 few_shot: Optional[List[Tuple[str, List[str]]]] = None) -> str:
    generic_examples = ", ".join(cfg.get("negative", [])[:4]) or "generic terms"
    extract_desc = cfg.get("extract_desc", "domain-specific concepts")
    role = cfg.get("role", "domain expert")

    few_shot_block = ""
    if few_shot:
        snips = []
        for snippet, terms in few_shot:
            snips.append(
                f'Text: """{snippet}"""\nExpected JSON: {json.dumps(terms, ensure_ascii=False)}'
            )
        few_shot_block = "\n\nEXAMPLES:\n" + "\n\n".join(snips) + "\n"

    prompt = f"""
        You are a {role}.

        TASK:
        Extract the most salient *domain-specific key terms* from the given {domain} text.
        Return ONLY a JSON array of strings (no trailing commentary). 
        Each item must be a concise noun phrase (≤5 words) OR an accepted acronym (ALL CAPS).

        DOMAIN SPEC:
        Target terms are: {extract_desc}.

        DO:
        - Prefer specific, technical, canonical concepts.
        - Keep both acronym and full form if both are meaningful (e.g. "SEC", "Securities and Exchange Commission") and group them adjacently in output order.
        - Use singular form unless plural is canonical ("Generally Accepted Accounting Principles" is plural).
        - Max {max_terms} items. If fewer valid terms exist, return fewer.

        DO NOT:
        - Include generic / high-level filler words (e.g. {generic_examples}).
        - Include verbs, adjectives alone, full sentences, definitions, or overlapping near-synonyms.
        - Include temporal phrases, rhetorical phrases, or vague abstractions.
        - Add numbering or explanations.

        TEXT:
        \"\"\"{text}\"\"\"

        {few_shot_block}
        JSON array ONLY:
        """.strip()
    return prompt
def order_acronym_groups(groups: List[List[str]]) -> List[str]:
    ordered = []
    for g in groups:
        if len(g) == 1:
            ordered.append(g[0])
            continue
        full_forms = [x for x in g if not is_acronym_like(x)]
        acronyms  = [x for x in g if is_acronym_like(x)]
        if full_forms:
            full_forms.sort(key=len)
            ordered.extend(full_forms + acronyms)
        else:
            ordered.extend(g)
    return ordered

def extract_entities(
    text: str,
    domain: str,
    model: str = "gpt-4-turbo",
    max_terms: int = 20,
    few_shot: Optional[List[Tuple[str, List[str]]]] = None,
    diagnostics: bool = False,
    logger: Optional[logging.Logger] = None
) -> List[str] | Tuple[List[str], Dict[str, Any]]:

    if logger is None:
        logger = logging.getLogger("entity_extractor")
        if not logger.handlers:
            h = logging.StreamHandler()
            fmt = logging.Formatter("[%(levelname)s] %(message)s")
            h.setFormatter(fmt)
            logger.addHandler(h)
            logger.setLevel(logging.INFO)

    cfg = DOMAIN_CONFIGS.get(domain, {})
    prompt = build_prompt(text, domain, cfg, max_terms, few_shot=few_shot)

    stats = {
        "raw_text_length": len(text),
        "initial_candidates": 0,
        "after_basic_clean": 0,
        "after_generic_filter": 0,
        "after_variant_filter": 0,
        "final_count": 0,
        "filtered_reasons": {
            "empty": 0,
            "generic": 0,
            "too_long": 0,
            "sentence_like": 0,
            "forbidden_start": 0,
            "duplicate_variant": 0
        },
        "errors": []
    }

    try:
        response = query_teacher(
            prompt,
            domain=domain,
            max_new_tokens=512,
            do_sample=False,
            model=model,
        )
        raw_content = response
    except Exception as e:
        stats["errors"].append(f"LLM error: {e}")
        if diagnostics:
            return [], stats
        return []


    def parse_candidates(raw: str) -> List[str]:
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(x).strip() for x in data]
        except Exception:
            pass
        return [p.strip() for p in re.split(r'[\n,，;；、•]+', raw)]

    candidates = parse_candidates(raw_content)
    stats["initial_candidates"] = len(candidates)

    cleaned_stage1: List[str] = []
    seen_keys = set()

    for c in candidates:
        c0 = re.sub(r'^[\*\-\d]+\s*', '', c)         
        c0 = re.sub(r'[\.。]+$', '', c0)             
        c0 = c0.strip()
        if not c0:
            stats["filtered_reasons"]["empty"] += 1
            continue
        if looks_sentence(c0):
            stats["filtered_reasons"]["sentence_like"] += 1
            continue
        if contains_too_many_tokens(c0):
            stats["filtered_reasons"]["too_long"] += 1
            continue
        key = canonical_key(c0)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        cleaned_stage1.append(c0)

    stats["after_basic_clean"] = len(cleaned_stage1)

    filtered_stage2: List[str] = []
    for e in cleaned_stage1:
        if has_forbidden_start(e):
            stats["filtered_reasons"]["forbidden_start"] += 1
            continue
        if is_generic(e, cfg):
            stats["filtered_reasons"]["generic"] += 1
            continue
        filtered_stage2.append(e)

    stats["after_generic_filter"] = len(filtered_stage2)

    final_stage_pre: List[str] = []
    for e in filtered_stage2:
        if is_redundant_variant(e, final_stage_pre):
            stats["filtered_reasons"]["duplicate_variant"] += 1
            continue
        final_stage_pre.append(e)

    stats["after_variant_filter"] = len(final_stage_pre)

    groups = group_acronyms(final_stage_pre)
    grouped_linear = order_acronym_groups(groups)
    scored_items = [(ent, score_entity(ent)) for ent in grouped_linear]
    scored_items.sort(key=lambda x: x[1], reverse=True)

    
    scored_items.sort(key=lambda x: x[1], reverse=True)

    ordered: List[str] = []
    seen_final = set()
    for ent, _ in scored_items:
        k = canonical_key(ent)
        if k not in seen_final:
            seen_final.add(k)
            ordered.append(ent)

    ordered = ordered[:max_terms]

    normalized_final = [normalize_entity_surface(e) for e in ordered]

    stats["final_count"] = len(normalized_final)

    if diagnostics:
        stats["raw_response"] = raw_content
        stats["final_entities"] = normalized_final
        return normalized_final, stats
    return normalized_final
