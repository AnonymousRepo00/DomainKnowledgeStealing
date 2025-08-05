from typing import Optional, Sequence
import re, openai
import json
from teacher import query_teacher

BASE_TEMPLATES: Sequence[str] = [
    "You mentioned \"{phrase}\"—please explain it further from a scientific or mechanistic perspective.",
    "Please provide a concrete example to illustrate how \"{phrase}\" works in practice."
]

class FollowUpGenerator:
    def __init__(self,
                 ppl_scorer,
                 broad_detector,
                 *,
                 tau_wide_hi: float = 0.6,
                 tau_wide_mid: float = 0.4,
                 tau_ppl_low: float = 20.0,
                 tau_ppl_hi: float = 40.0,
):
        
        self.ppl_scorer = ppl_scorer
        self.broad_detector = broad_detector
        self.tau_wide_hi = tau_wide_hi
        self.tau_wide_mid = tau_wide_mid
        self.tau_ppl_low = tau_ppl_low
        self.tau_ppl_hi  = tau_ppl_hi


    def ask(self, 
            student_answer: str,
            *,
            phrase: str,
            verbose: bool = False) -> Optional[str]:
        w = self._width(student_answer)
        p = self.ppl_scorer.compute(student_answer)

        if verbose:
            print(f"[Debug] width={w:.3f}, ppl={p:.1f}")

        method = self._select_method(w, p)
        if method is None:
            return None
        if method == "template":
            return self._fill_template(phrase)
        return self._gpt_followup(student_answer)

    def _select_method(self, w: float, p: float) -> Optional[str]:
        if w >= self.tau_wide_hi and p <= self.tau_ppl_low:
            return "template"
        if w >= self.tau_wide_mid or p >= self.tau_ppl_hi:
            return "gpt"
        return None

    def _width(self, ans: str) -> float:
        return self.broad_detector.width_score(ans)

    def _fill_template(self, phrase: str) -> str:
        return BASE_TEMPLATES[0].format(phrase=phrase)

    def _gpt_followup(self, base_answer: str) -> str:
        system = "You are an expert tutor."
        user = (f'The student gave this answer:\n"{base_answer}"\n\n'
                "Ask exactly ONE concise follow-up question that would guide the student to a deeper understanding. "
                "Do not explain, only ask the question.")
        rsp = openai.ChatCompletion.create(
            model=self.gpt_model,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            temperature=0.7,
            max_tokens=64
        )
        return rsp.choices[0].message.content.strip()
    
    

def chunk_answer(ans: str, max_chunks: int = 3) -> str:
    sents = re.split(r"[.!?；;]", ans)
    sents = [s.strip() for s in sents if s.strip()]
    return " | ".join(sents[:max_chunks])

def generate_followups_by_teacher(answer: str,
                                  max_q: int = 3,
                                  gpt_model: str = "gpt-4o-mini") -> list[str]:

    prompt = (
        "You just produced the following answer:\n"
        f"\"\"\"\n{answer}\n\"\"\"\n\n"
        "Think step by step and list up to "
        f"{max_q} precise follow-up questions that would probe deeper into "
        "the key points you mentioned. "
        "Return ONLY a JSON array of strings."
    )
    rsp = query_teacher(prompt, domain="general", max_new_tokens=256)
    try:
        qs = json.loads(rsp)
        if isinstance(qs, list):
            return [q.strip() for q in qs if q.strip()]
    except Exception:
        pass
    return re.findall(r"\"([^\"]+\?)\"", rsp)[:max_q]
