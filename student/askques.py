import re
import json
import random
import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import time
from functools import lru_cache
from collections import deque

STOPWORDS = {"the","of","and","in","for","to","a","an","on","at","by","with","within","law","act","rule","rules"}


@dataclass
class DomainConfig:
    name: str
    example_entity: str
    example_questions: Dict[str, str]
    anchors: List[str]
    banned_cross_domain: List[str]
    disambiguation: Dict[str, str]
    max_tokens: int
    categories_hint: str
    function_hint: str
    part_of_hint: str

FINANCE_CONFIG = DomainConfig(
    name="financial",
    example_entity="Value at Risk",
    example_questions={
        "definition": "How is Value at Risk defined in terms of loss threshold, confidence level, and holding period?",
        "categories": "Which VaR methodologies (parametric, historical simulation, Monte Carlo) differ in distributional assumptions and data usage?",
        "function": "How does reported VaR influence trading limits, capital allocation, and escalation triggers for a trading desk?",
        "part_of": "Within which integrated market risk, stress testing, and Basel III capital adequacy framework is VaR embedded?"
    },
    anchors=[
        "risk","volatility","exposure","capital","liquidity","leverage","tranche",
        "discount","cash flow","yield","spread","hedge","regulatory","baseline","rating","default","duration",
        "valuation","diversification","benchmark","reporting","governance","compliance","return","optimization"
    ],
    banned_cross_domain=[
        "hearsay","precedent","jurisdiction","pathophysiology","enzyme",
        "clinical","mechanistic","physiological","etiological","staging","syndrome","biomarker","pathway"
    ],
    disambiguation={
        "Basel III": "Global banking regulatory framework for capital adequacy, leverage, and liquidity coverage.",
        "SOX": "Sarbanes-Oxley Act corporate governance & internal control reporting requirements."
    },
    max_tokens=45,
    categories_hint="instrument types, risk factor groupings, rating buckets, tranche layers, methodological classes",
    function_hint="valuation impact, risk mitigation, capital or pricing decision role",
    part_of_hint="risk management, portfolio optimization, regulatory capital or reporting architecture"
)

BIOMED_CONFIG = DomainConfig(
    name="biomedical",
    example_entity="Pulmonary Embolism",
    example_questions={
        "definition": "How is pulmonary embolism clinically defined with respect to obstructed pulmonary arteries and resulting hemodynamic compromise?",
        "categories": "Which embolus types (thrombotic, fat, air, amniotic) are distinguished for acute risk stratification algorithms?",
        "function": "How does early risk stratification modify anticoagulation, thrombolysis, or embolectomy decision pathways in pulmonary embolism?",
        "part_of": "Within which broader venous thromboembolism and cardiopulmonary emergency care pathways is pulmonary embolism evaluation embedded?"
    },
    anchors=[
        "pathway","mechanism","biomarker","physiological","pathophysiology","staging",
        "syndrome","receptor","protocol","risk stratification","diagnostic algorithm","feedback loop"
    ],
    banned_cross_domain=["statutory","precedent","capital","tranche","yield","derivative"],
    disambiguation={
        "ASA": "ASA Physical Status Classification (anesthesia risk) or acetylsalicylic acid antiplatelet — choose contextually."
    },
    max_tokens=40,
    categories_hint="subtypes, staging systems, severity scales, anatomical or etiological classes",
    function_hint="mechanism, clinical decision impact, diagnostic or therapeutic role",
    part_of_hint="organ systems, physiological pathways, integrated care protocols"
)

DOMAIN_MAP: Dict[str, DomainConfig] = {
    "financial": FINANCE_CONFIG,
    "medical": BIOMED_CONFIG
}

FIN_ENTITY_HINTS = {
    "accounting": "recognition, measurement principles, accrual vs cash basis, financial statement elements",
    "auditing": "audit assertions (existence, completeness, valuation), materiality, evidence types, risk model",
    "tax": "taxable income categories, deferred tax assets/liabilities, basis adjustments, anti-avoidance rules",
    "merger": "valuation drivers, synergy types, deal structure, integration risk, antitrust review",
    "acquisition": "valuation drivers, synergy types, deal structure, integration risk, antitrust review",
    "venture capital": "staged financing (seed/Series A/B), screening (team, PMF, traction), governance rights, exit routes",
    "private equity": "leverage structure, operational value creation, exit strategy, fee model, governance",
    "behavioral": "cognitive biases, investor sentiment, anomaly persistence, limits to arbitrage",
    "fintech": "distributed ledger, smart contracts, open banking APIs, regulatory sandbox, algorithmic transparency",
    "insurance": "underwriting, premium pricing, reserving, reinsurance structure, solvency capital",
    "portfolio": "asset allocation, risk budgeting, factor exposures, rebalancing, tracking error",
    "asset management": "mandate constraints, style drift, performance attribution, liquidity management",
    "wealth": "client risk tolerance, tax efficiency, strategic vs tactical allocation, estate interface",
    "retirement": "asset-liability matching, longevity risk, glide path allocation, funding gap",
    "estate": "asset transfer vehicles, tax minimization, beneficiary designation, liquidity planning",
    "credit": "default probability, loss given default, credit spread drivers, rating migration",
    "regulation": "Basel pillars, capital buffers, stress testing, disclosure, supervisory review",
    "capital adequacy": "risk-weighted assets, leverage ratio, liquidity coverage, internal models vs standardized",
    "risk management": "risk taxonomy, limits framework, hedging strategies, stress & scenario analysis",
    "expected shortfall": "tail loss estimation, coherence vs VaR, stress integration, capital allocation",
    "value at risk": "confidence level, holding period, distribution assumptions, backtesting, model limitations",
    "incremental risk": "default and migration risk beyond VaR, trading book capital, liquidity horizon"
}


BIOMED_ALIAS_MAP = {
    "igg": "Immunoglobulin G",
    "ig g": "Immunoglobulin G",
    "iga": "Immunoglobulin A",
    "ig a": "Immunoglobulin A",
    "igm": "Immunoglobulin M",
    "ig m": "Immunoglobulin M",
    "secretory iga": "Secretory IgA",
    "aap": "American Academy of Pediatrics"
}

MED_ENTITY_HINTS = {
    "anatomy": "structural hierarchy (cell, tissue, organ, system), anatomical planes (sagittal, coronal, axial), regional relationships",
    "biochemistry": "macromolecule classes, enzymatic kinetics, metabolic pathway regulation, structure–function relationships",
    "microbiology": "gram reaction, morphology, metabolism (aerobic/anaerobic), virulence factors, host range",
    "immunology": "innate vs adaptive components, antigen presentation, clonal expansion, effector regulation, memory formation",
    "pathology": "cell injury, inflammation, disordered growth, neoplasia, degeneration, repair processes",
    "nephrology": "glomerular vs tubular processes, filtration, reabsorption, electrolyte homeostasis, renal hemodynamics",
    "pulmonary embolism": "embolus origin, risk stratification, hemodynamic impact, diagnostic imaging, acute management",
    "american academy of pediatrics": "organizational structure, guideline development, advocacy roles, pediatric care standards",
    "world health organization": "governance structure, global health surveillance, outbreak response, normative guideline setting"
}

MED_NEUTRAL_ANCHORS = ["mechanism","pathway","biomarker","staging","syndrome","regulation","feedback loop"]
MED_ONCO_KEYWORDS = ["cancer","tumor","tumour","neoplasm","oncolog","metastasis","metastatic"]

TAG_PHRASE_POOLS = {}
NEGATIVE_EXAMPLES = {}

def _safe_json_dump(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


@dataclass
class GenerationConfig:
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    temperature: float = 0.85
    top_p: float = 0.95
    repetition_penalty: float = 1.05
    max_new_tokens: int = 220
    max_retries: int = 4
    max_tokens_per_question: int = 40         
    enforce_seed: Optional[int] = 42
    device: Optional[str] = None
    enable_paraphrase: bool = False           
    template_diversity: bool = False
    include_negative_examples: bool = False
    verbose: bool = True
    minimal_mode: bool = True
    hybrid_mode: bool = True
    hybrid_refine: bool = True


@dataclass
class FailureStats:
    counts: Dict[str, int] = field(default_factory=lambda: {
        "missing_all": 0,
        "missing_any": 0,
        "duplicate_tag": 0,
        "copy_example": 0,
        "contains_placeholder": 0,
        "contains_what_is": 0,
        "too_short": 0,
        "too_long": 0,
        "instruction_echo": 0,
        "bad_format": 0,
        "weak_form": 0,
        "cross_domain_leak": 0,
        "anchor_injected": 0,
        "var_overuse": 0,
        "truncated": 0,
        "alias": 0,
        "entity_missing": 0,
        "hint_mismatch": 0,
        "template_collapse": 0,
        "redundant_slots": 0,
        "weak_categories": 0
    })

    def inc(self, key: str):
        if key in self.counts:
            self.counts[key] += 1
        else:
            self.counts[key] = 1

    def summary(self) -> Dict[str, int]:
        return dict(self.counts)


class LLaMAQuestionGenerator:
    VALID_TAGS = ["[definition]", "[categories]", "[function]", "[part_of]"]
    EXAMPLE_ENTITY = "Pulmonary Embolism"
    EXAMPLE_ENTITY_L = EXAMPLE_ENTITY.lower()

    INSTRUCTION_ECHO_PATTERNS = [
        r"you are a .*question generator",
        r"style example",
        r"do not repeat",
        r"constraints",
    ]
    WEAK_START_RE = re.compile(r'^(are|is|does|do|can|should)\b', re.IGNORECASE)
    STOP_MARKER = "### OUTPUT_END"

    def __init__(self, cfg: GenerationConfig, model: Optional[AutoModelForCausalLM] = None, tokenizer: Optional[AutoTokenizer] = None):
        self.cfg = cfg
        if cfg.minimal_mode and not getattr(cfg, "hybrid_mode", False):
            self.model = None
            self.tokenizer = None
            self.device = cfg.device if cfg.device is not None else "cpu"
        else:
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            self.tokenizer = tokenizer
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(
                    cfg.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            self.model = model
            self.model.eval()
            self.device = cfg.device or next(self.model.parameters()).device
            if cfg.enforce_seed is not None:
                self._set_seed(cfg.enforce_seed)

        self.failure_stats = FailureStats()
        from collections import deque
        self._recent_openings = deque(maxlen=200)      
        self._anchor_injections: Dict[Tuple[str,str], int] = {}
        self._cache: Dict[Tuple[str, str], Dict[str, str]] = {}
        self._tok_cache: Dict[str, int] = {}
        self._var_avoid: set = set()
        self._biomed_seen: set = set()
        self._domain_static_prefix: Dict[str, str] = {}
        for dname, dcfg in DOMAIN_MAP.items():
            self._domain_static_prefix[dname] = (
                f"You are a domain-sensitive question generator (domain={dcfg.name}).\n"
                "Return FOUR lines ONLY, each starts with one tag then a question ending with '?'.\n"
                "Avoid yes/no starts; avoid phrase 'what is'; <= {max_tokens} tokens per question.\n"
                f"After four lines output a line: {self.STOP_MARKER}\n"
            )
        if self.cfg.minimal_mode or self.cfg.hybrid_mode:
            for dname in self._domain_static_prefix:
                self._domain_static_prefix[dname] = (
                    "Return FOUR lines ONLY; each begins with a tag and ends with '?'. "
                    "Tags: [definition] [categories] [function] [part_of]. "
                    "Avoid 'what is'. <= {max_tokens} tokens each.\n"
                )


    def generate_for_entities(self, domain, entities: List[str]) -> Dict[str, Dict[str, str]]:
        results = {}
        for ent in entities:
            qdict = self.generate_cached(domain, ent)
            results[ent] = qdict
            if self.cfg.verbose:
                print(f"\n=== {ent} ===")
                for tag in self.VALID_TAGS:
                    key = tag.strip("[]")
                    print(f"{tag} {qdict.get(key, '')}")
        if self.cfg.verbose:
            print("\n[Failure Stats]", json.dumps(self.failure_stats.summary(), indent=2))
        if self.cfg.verbose and self.failure_stats.counts.get("redundant_slots",0):
            print(f"[Info] Redundant slot rejections so far: {self.failure_stats.counts['redundant_slots']}")
        return results


    def _generate_one_entity(self, domain, entity: str) -> Tuple[Dict[str, str], Dict[str, int]]:
        if getattr(self.cfg, "hybrid_mode", False):
            base_map = self._minimal_template_fill(domain, entity)
            if self.cfg.hybrid_refine and self.model is not None:
                for attempt in range(self.cfg.max_retries + 1):
                    refined = self._hybrid_refine(domain, entity, base_map, attempt)
                    if refined:
                        return refined, {"retries": attempt, "hybrid_refined": True}
                return base_map, {"retries": self.cfg.max_retries, "hybrid_refined": False}
            return base_map, {"retries": 0, "hybrid_refined": False}
        for attempt in range(self.cfg.max_retries + 1):
            raw_text = self._raw_generate(domain, entity, attempt)
            output_block = self._extract_output_block(raw_text)
            parsed, reasons = self._parse_and_validate(domain, output_block, entity)
            if parsed is not None:
                parsed = self._post_check(domain, entity, parsed)
                return parsed, {"retries": attempt}

            for r in reasons:
                self.failure_stats.inc(r)
            if self.cfg.verbose:
                print(f"[Retry {attempt}] entity={entity} reasons={reasons}")
            if "var_overuse" in reasons:
                self._var_avoid.add(entity.lower())

        self.failure_stats.inc("missing_all")
        fallback = self._fallback_questions(domain, entity)
        return fallback, {"retries": self.cfg.max_retries, "fallback_used": True}

    def _dynamic_tag_instructions(self, domain_name: str) -> str:
        if True: 
            dcfg = DOMAIN_MAP[domain_name]
            return (
                f"[definition] core discriminative criteria (avoid 'what is')\n"
                f"[categories] classification axes ({dcfg.categories_hint})\n"
                f"[function] decision / procedural impact ({dcfg.function_hint})\n"
                f"[part_of] higher-level framework ({dcfg.part_of_hint})\n"
            )

    def _build_prompt(self, domain_name, entity: str) -> str:
        dcfg = DOMAIN_MAP[domain_name]
        static_prefix = self._domain_static_prefix[domain_name]
        disamb_line = ""
        for key, desc in dcfg.disambiguation.items():
            if key.lower() in entity.lower():
                disamb_line = f"Clarify: {desc}\n"
                break
        tag_instruction_block = self._dynamic_tag_instructions(domain_name)
        prompt = (
            f"{static_prefix.replace('{max_tokens}', str(dcfg.max_tokens))}"
            f"{tag_instruction_block}"
            f"Entity: {entity}\n"
            f"{disamb_line}"
            "Output the four tagged questions only.\n"
            "### OUTPUT_START\n"
        )
        return prompt
    def _token_len(self, text: str) -> int:
        """
        Fast approximate token length with caching.
        """
        if text in self._tok_cache:
            return self._tok_cache[text]
        length = len(self.tokenizer.encode(text, add_special_tokens=False))
        self._tok_cache[text] = length
        return length

    def _entity_core_tokens(self, entity: str) -> List[str]:
        base = re.findall(r"[a-zA-Z]+", entity.lower())
        return [t for t in base if t not in STOPWORDS]

    def _has_entity_token(self, text: str, entity_tokens: List[str]) -> bool:
        low = text.lower()
        return any(t in low for t in entity_tokens)

    def _hint_keywords(self, domain_name: str, entity: str) -> List[str]:
        el = entity.lower()
        if domain_name == "financial":
            for k,h in FIN_ENTITY_HINTS.items():
                if k in el:
                    return [w for w in re.findall(r"[a-zA-Z]+", h.lower()) if w not in STOPWORDS][:12]
        if domain_name == "medical":
            for k,h in MED_ENTITY_HINTS.items():
                if k in el:
                    return [w for w in re.findall(r"[a-zA-Z]+", h.lower()) if w not in STOPWORDS][:12]
        return []

    def _slot_redundancy_score(self, mapping: Dict[str,str]) -> float:

        import itertools
        sets = []
        for k,v in mapping.items():
            toks = [t.lower() for t in re.findall(r"[A-Za-z]+", v) if t.lower() not in STOPWORDS]
            sets.append(set(toks))
        if len(sets) < 2:
            return 0.0
        pair_scores = []
        for a,b in itertools.combinations(sets, 2):
            if not a and not b:
                continue
            inter = len(a & b)
            union = len(a | b) or 1
            pair_scores.append(inter/union)
        return sum(pair_scores)/len(pair_scores) if pair_scores else 0.0

    def _categories_axis_count(self, text: str) -> int:

        parts = [p.strip() for p in re.split(r",|;|/|\\band\\b", text) if p.strip()]
        parts = [p for p in parts if len(p.split()) >= 1]
        norm = []
        for p in parts:
            key = re.sub(r"[^a-zA-Z]+"," ",p).strip().lower()
            if key and key not in norm:
                norm.append(key)
        return len(norm)


    @torch.no_grad()
    def _raw_generate(self, domain, entity: str, attempt: int = 0) -> str:
        prompt = self._build_prompt(domain,entity)
        temp = max(0.6, self.cfg.temperature - attempt * 0.1)
        top_p = max(0.7, self.cfg.top_p - attempt * 0.05)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out_ids = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            repetition_penalty=self.cfg.repetition_penalty,
            max_new_tokens=self.cfg.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        decoded = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        if decoded.startswith(prompt):
            decoded = decoded[len(prompt):]
        dcfg = DOMAIN_MAP[domain]
        raw_lower = decoded.lower()
        if any(b.lower() in raw_lower for b in dcfg.banned_cross_domain):
            return "" 
        return decoded

    def _extract_output_block(self, text: str) -> str:
        block = text.split("### OUTPUT_START", 1)[-1]
        if self.STOP_MARKER in block:
            block = block.split(self.STOP_MARKER, 1)[0]
        return block.strip()

    def _parse_and_validate(self, domain_name,  block: str, entity: str) -> Tuple[Optional[Dict[str, str]], List[str]]:
        reasons = []
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        tag_lines = [l for l in lines if any(l.startswith(t) for t in self.VALID_TAGS)]

        if len(tag_lines) == 0:
            reasons.append("missing_any")
            return None, reasons

        cleaned = []
        for l in tag_lines:
            l = re.sub(r"[`\"']+$", "", l)
            l = re.sub(r"\?+\s*$", "?", l)
            cleaned.append(l)

        mapping = {}
        for l in cleaned:
            head = l.split("]", 1)[0] + "]"
            body = l[len(head):].strip()
            tag_key = head.strip("[]")
            if head not in self.VALID_TAGS:
                reasons.append("bad_format")
                continue
            if tag_key in mapping:
                reasons.append("duplicate_tag")
                continue
            mapping[tag_key] = body

        dcfg = DOMAIN_MAP[domain_name]
        banned_terms = [b.lower() for b in dcfg.banned_cross_domain]
        joined_lower = " ".join(mapping.values()).lower()
        if any(bt in joined_lower for bt in banned_terms):
            reasons.append("cross_domain_leak")

        entity_tokens = self._entity_core_tokens(entity)
        if "definition" in mapping and not self._has_entity_token(mapping["definition"], entity_tokens):
            reasons.append("entity_missing")
        if not any(self._has_entity_token(v, entity_tokens) for v in mapping.values()):
            reasons.append("entity_missing")
        hint_keys = self._hint_keywords(domain_name, entity)
        if hint_keys:
            if not any(any(h in v.lower() for h in hint_keys) for v in mapping.values()):
                reasons.append("hint_mismatch")

        for kq, vq in mapping.items():
            lowvq = vq.lower()
            tokens = re.findall(r"[a-zA-Z]+", lowvq)
            tset = set(tokens)
            if any(b.lower() in tset for b in dcfg.banned_cross_domain):
                reasons.append("cross_domain_leak")

        if domain_name == "financial":
            if any(b in ("clinical","mechanistic","physiological","etiological","staging","syndrome","biomarker","pathway") for b in tokens):
                reasons.append("cross_domain_leak")
            joined = " ".join(mapping.values()).lower()
            var_terms = ["value at risk"," var ","expected shortfall","incremental risk charge","irc "]
            var_hits = sum(joined.count(t) for t in var_terms)
            ent_low = entity.lower()
            risk_core = any(k in ent_low for k in ["value at risk","expected shortfall","incremental risk","var "])
            if var_hits > 1 and not risk_core:
                reasons.append("var_overuse")

        for k, v in mapping.items():
            if self.WEAK_START_RE.match(v):
                reasons.append("weak_form")

        if len(mapping) != 4:
            reasons.append("missing_any")

        lower_block = block.lower()
        if self.EXAMPLE_ENTITY_L in lower_block:
            reasons.append("copy_example")

        for pat in self.INSTRUCTION_ECHO_PATTERNS:
            if re.search(pat, lower_block):
                reasons.append("instruction_echo")
                break

        for k, v in mapping.items():
            lv = v.lower()
            if "..." in v:
                reasons.append("contains_placeholder")
            if "what is" in lv:
                reasons.append("contains_what_is")
            tok_len = self._token_len(v)
            if tok_len < 5:
                reasons.append("too_short")
            if tok_len > dcfg.max_tokens:
                reasons.append("too_long")
        if domain_name == "medical":
            for v in mapping.values():
                body = v.rstrip("?").strip()
                last_word = body.split()[-1].lower() if body.split() else ""
                if body.endswith(" to mod") or last_word in {"mod"}:
                    reasons.append("truncated")
                    break

        openings = [re.findall(r"^[A-Za-z]+(?:\s+[A-Za-z]+)?", mapping[k]) for k in mapping]
        flat_open = [o[0].lower() for o in openings if o]
        if len(set(flat_open)) <= 2:  # too repetitive
            reasons.append("template_collapse")
        else:
            for o in flat_open:
                self._recent_openings.append(o)
        if len(mapping) == 4:
            red_score = self._slot_redundancy_score(mapping)
            if red_score >= 0.55:
                reasons.append("redundant_slots")
            cat_txt = mapping.get("categories","")
            if self._categories_axis_count(cat_txt) < 3:
                reasons.append("weak_categories")

        if not reasons:
            for k, v in mapping.items():
                mapping[k] = v.replace(self.STOP_MARKER, "").strip()
            if self.cfg.verbose:
                pass  

        if reasons:
            return None, reasons

        return mapping, reasons

    def generate_cached(self, domain: str, entity: str) -> Dict[str, str]:
        if domain == "medical":
            lowe = entity.lower()
            if lowe in BIOMED_ALIAS_MAP:
                canonical = BIOMED_ALIAS_MAP[lowe]
                if self.cfg.verbose and canonical != entity:
                    print(f"[Alias] Canonicalizing '{entity}' -> '{canonical}'")
                entity = canonical
            if entity in self._biomed_seen and (domain, entity) in self._cache:
                return self._cache[(domain, entity)]
            
        key = (domain, entity)
        if key not in self._cache:
            qdict, _ = self._generate_one_entity(domain, entity)
            self._cache[key] = qdict
            if domain == "medical":
                self._biomed_seen.add(entity)
        return self._cache[key]
    def clear_cache(self):
        self._cache.clear()

    def _post_check(self, domain_name, entity: str, mapping: Dict[str, str]) -> Dict[str, str]:
        if getattr(self.cfg, "hybrid_mode", False):
            openings = [re.split(r"\s+", mapping[t], 2)[0].lower() for t in mapping]
            if len(set(openings)) < 4:
                used = set()
                for t in mapping:
                    head = mapping[t].split(None,1)[0].lower()
                    if head in used:
                        mapping[t] = "In what ways " + mapping[t][0].lower() + mapping[t][1:]
                    used.add(mapping[t].split(None,1)[0].lower())
            return mapping
        dcfg = DOMAIN_MAP[domain_name]
        if self.cfg.minimal_mode and not getattr(self.cfg, "hybrid_mode", False):
            return mapping
        anchor_key = (domain_name, entity)
        current_injections = self._anchor_injections.get(anchor_key, 0)
        MAX_ANCHOR_PER_ENTITY = 1
        for tag, q in list(mapping.items()):
            lower_q = q.lower()
            if current_injections >= MAX_ANCHOR_PER_ENTITY:
                continue
            has_domain_anchor = any(a.lower() in lower_q for a in dcfg.anchors)
            has_include_phrase = "include analysis of" in lower_q or "(consider" in lower_q
            if (not has_domain_anchor) and (not has_include_phrase) and random.random() < 0.20:
                if domain_name == "medical":
                    anchor = random.choice(MED_NEUTRAL_ANCHORS)
                    q = q.rstrip("?")
                    q = f"{q} (address {anchor})?"
                else:
                    anchor = random.choice(dcfg.anchors)
                    q = q.rstrip("?")
                    q = f"{q} (consider {anchor})?"
                mapping[tag] = q
                current_injections += 1
        self._anchor_injections[anchor_key] = current_injections

        if domain_name == "medical":
            ent_low = entity.lower()
            is_onco_entity = any(k in ent_low for k in MED_ONCO_KEYWORDS)
            if not is_onco_entity:
                for tag, q2 in list(mapping.items()):
                    lowq2 = q2.lower()
                    if any(k in lowq2 for k in MED_ONCO_KEYWORDS):
                        cleaned = re.sub(r"(cancer|tumou?r|neoplasm|metastatic|metastasis)", "condition", q2, flags=re.IGNORECASE)
                        mapping[tag] = cleaned
        if domain_name == "law":
            ent_low = entity.lower()
            admin_keys = ("administrative", "chevron", "agency", "deference")
            is_admin = any(k in ent_low for k in admin_keys)
            if not is_admin:
                for tag, qv in list(mapping.items()):
                    if "chevron" in qv.lower():
                        cleaned = re.sub(r"(?i)chevron (deference )?", "appropriate doctrinal", qv)
                        mapping[tag] = cleaned
        if domain_name == "financial":
            ent_low = entity.lower()
            risk_keys = ("value at risk","expected shortfall","var ","incremental risk","market risk","risk management")
            is_risk = any(k in ent_low for k in risk_keys)
            if not is_risk:
                for tag, qv in list(mapping.items()):
                    low = qv.lower()
                    if any(t in low for t in ["value at risk","expected shortfall","incremental risk charge","var/ES","var "]):
                        cleaned = re.sub(r"(?i)value at risk|expected shortfall|incremental risk charge|VaR/ES|VaR", "relevant metrics", qv)
                        mapping[tag] = cleaned
        for tag, qv in list(mapping.items()):
            mapping[tag] = self._light_paraphrase(domain_name, qv, tag)
        openings = [re.split(r"\s+", mapping[t], 2)[0].lower() for t in mapping]
        if len(set(openings)) < 4:
            used = set()
            for t in mapping:
                first = mapping[t].split(None,1)
                if not first:
                    continue
                head = first[0].lower()
                if head in used:
                    if not mapping[t].lower().startswith(("in what","across which","under what","by which")):
                        mapping[t] = "In what ways " + mapping[t][0].lower() + mapping[t][1:]
                used.add(mapping[t].split(None,1)[0].lower())
        return mapping

    def _minimal_template_fill(self, domain: str, entity: str) -> Dict[str, str]:

        ent = entity.strip()
        low = ent.lower()
        OPEN_DEF = ["Which", "What core", "How do the key", "By what principal"]
        OPEN_CAT = ["Across which", "Along which", "By which", "Across what"]
        OPEN_FUN = ["How does", "In what ways does", "Through what mechanisms does", "To what extent does"]
        OPEN_PART = ["Within which", "In which broader", "Under which", "Across which higher-level"]

        def pick(pool, salt):
            return pool[hash((salt, low)) % len(pool)]

        axis_map = {
            "medical": ["subtypes", "staging/severity scales", "anatomical regions", "etiologic classes"],
            "law": ["scrutiny tiers", "doctrinal tests", "rights domains", "procedural stages"],
            "financial": ["measurement bases", "statement elements", "valuation methods", "risk groupings"]
        }
        impact_map = {
            "medical": ["diagnostic selection", "therapeutic choice", "prognostic stratification", "monitoring strategy"],
            "law": ["burden allocation", "motion strategy", "remedy scope", "adjudicative outcomes"],
            "financial": ["valuation comparability", "capital allocation", "risk limit setting", "earnings quality assessment"]
        }
        framework_map = {
            "medical": ["integrated care pathways", "physiological systems", "clinical governance processes"],
            "law": ["litigation lifecycle", "governance structure", "appellate review hierarchy"],
            "financial": ["reporting architecture", "governance & control systems", "regulatory capital framework"]
        }
        axes = ", ".join(axis_map.get(domain, axis_map["medical"]))
        impacts = ", ".join(impact_map.get(domain, impact_map["medical"])[:3])
        frames = ", ".join(framework_map.get(domain, framework_map["medical"])[:3])

        qdef = f"{pick(OPEN_DEF,'def')} discriminative features define {ent}?"
        qcat = f"{pick(OPEN_CAT,'cat')} axes ({axes}) is {ent} organized?"
        qfun = f"{pick(OPEN_FUN,'fun')} {ent} influence {impacts}?"
        qpart = f"{pick(OPEN_PART,'part')} frameworks ({frames}) does {ent} operate?"

        return {
            'definition': qdef,
            'categories': qcat,
            'function': qfun,
            'part_of': qpart
        }

    def _hybrid_refine(self, domain: str, entity: str, base_map: Dict[str, str], attempt: int = 0) -> Optional[Dict[str, str]]:

        if self.model is None or self.tokenizer is None:
            return None
        dcfg = DOMAIN_MAP[domain]
        hint_tokens = self._hint_keywords(domain, entity)
        hint_line = ""
        if hint_tokens:
            hint_line = "Hint keywords: " + ", ".join(hint_tokens[:8]) + "\n"

        template_lines = "\n".join(f"{tag} {base_map[tag.strip('[]')]}" for tag in self.VALID_TAGS)
        prompt = (
            "Refine the following four template questions.\n"
            "Requirements:\n"
            "- Output EXACTLY four lines, preserve the exact tags in order: [definition],[categories],[function],[part_of].\n"
            "- Each line must remain a single interrogative sentence ending with '?'.\n"
            "- Add concise domain-specific detail; avoid excessive length (≤ "
            f"{dcfg.max_tokens} tokens per line).\n"
            "- Do NOT introduce cross-domain jargon (medical vs law vs financial mixing).\n"
            "- Avoid the phrase 'what is'.\n"
            "- Keep the entity token present in the definition line.\n"
            f"Domain: {domain}\n"
            f"Entity: {entity}\n"
            f"{hint_line}"
            "Template:\n"
            f"{template_lines}\n"
            "### OUTPUT_START\n"
        )
        temp = max(0.4, self.cfg.temperature - 0.1 * attempt)
        top_p = max(0.7, self.cfg.top_p - 0.05 * attempt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out_ids = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            repetition_penalty=self.cfg.repetition_penalty,
            max_new_tokens=dcfg.max_tokens * 6,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        decoded = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        if "### OUTPUT_START" in decoded:
            decoded = decoded.split("### OUTPUT_START", 1)[1]
        parsed, reasons = self._parse_and_validate(domain, decoded, entity)
        if parsed is None:
            return None
        refined = self._post_check(domain, entity, parsed)
        return refined

    def _light_paraphrase(self, domain_name: str, q: str, tag: str) -> str:

        if not self.cfg.template_diversity:
            return q
        lowers = q.lower()
        replacements = [
            (r"^how does ", "In what way does "),
            (r"^how do ", "In what ways do "),
            (r"^which ", "What "),
            (r"^along which ", "Across which "),
        ]
        for pat, repl in replacements:
            if re.match(pat, lowers):
                q = re.sub(pat, repl, q, count=1, flags=re.IGNORECASE)
                break
        q = re.sub(r"\s{2,}", " ", q)
        return q

    def generate_batch(self, domain: str, entities: List[str], batch_size: int = 1) -> Dict[str, Dict[str, str]]:
        results = {}
        for ent in entities:
            results[ent], _ = self._generate_one_entity(domain, ent)
        return results

    def _fallback_questions(self, domain_name, entity: str) -> Dict[str, str]:
        dcfg = DOMAIN_MAP[domain_name]
        if domain_name == "law":
            fb = {
                "definition": f"Which foundational doctrinal elements or structural principles delineate {entity}'s scope (tests, tiers, structural limits)?",
                "categories": f"Across which axes (scrutiny tiers, doctrinal tests, rights or structural domains, justiciability doctrines) is {entity} organized?",
                "function": f"In what ways does {entity} shift burdens, shape motion strategy, influence remedies, or constrain governmental action?",
                "part_of": f"Within which overarching legal architecture (review hierarchy, procedural sequence, governance or enforcement framework) does {entity} operate?"
            }
            return fb

        if domain_name == "financial":
            fb = {
                "definition": f"Which measurement and recognition principles (accrual basis, valuation basis, revenue/cost recognition) characterize {entity}?",
                "categories": f"Across which axes (statement elements, measurement bases, reporting or disclosure layers, planning vs performance horizon) is {entity} classified?",
                "function": f"How does {entity} influence valuation comparability, earnings quality assessment, capital allocation, or internal control decisions?",
                "part_of": f"Within which broader financial reporting, governance, audit, or regulatory framework (IFRS/GAAP, SOX, Basel if relevant) is {entity} embedded?"
            }
            return fb

        fb = {
            "definition": f"How is {entity} delimited across structural/functional levels (cell, tissue, organ, system) and key distinguishing processes?",
            "categories": f"Across which structural or functional axes (hierarchical levels, regional/system groupings, tissue or subtype classes, developmental or etiologic distinctions) is {entity} organized?",
            "function": f"In what ways does {entity} inform diagnostic interpretation, therapeutic choice, monitoring strategy, or prognostic stratification?",
            "part_of": f"Within which broader physiological pathways, integrated organ-system interactions, or coordinated care protocols is {entity} contextualized?"
        }
        return fb

    def _set_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
