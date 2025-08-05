from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from datetime import datetime
from collections import deque
from datetime import datetime
from typing import Dict, Optional, List
import json, os
import heapq, itertools


class ExplorationMap:
    def __init__(self):
        self.state = {
            "qa_log": [],           
            "explored_entities": [],  
            "to_explore": []          
        }
        self._queue   = deque()      
        self._visited = set()       
        self._queued  = set()         
        self.entity_meta = {}       

    def __contains__(self, item: str) -> bool:
        return item in self._queue or item in self._visited
    
    def is_explored(self, entity: str) -> bool:
        return entity in self._visited

    def already_queued(self, entity: str) -> bool:
        return entity in self._queued

    def add_to_explore(
        self,
        entity: str,
        source: str,
        from_entity: Optional[str] = None,
        why_added: Optional[str] = None,
        priority: bool = False,
    ):
        if self.is_explored(entity) or self.already_queued(entity):
            return

        item = {
            "entity": entity,
            "source": source,
            "from_entity": from_entity,
            "why_added": why_added,
            "timestamp": datetime.now().isoformat(),
        }
        if priority:
            self._queue.appendleft(item)
        else:
            self._queue.append(item)

        self.state["to_explore"].append(item)

        self._queued.add(entity)

        meta = self.entity_meta.setdefault(
            entity, {"parent": from_entity, "children": set(), "sources": set()}
        )
        meta["sources"].add(source)
        if from_entity:
            parent_meta = self.entity_meta.setdefault(
                from_entity, {"parent": None, "children": set(), "sources": set()}
            )
            parent_meta["children"].add(entity)

    def pop_next_entity(self) -> Optional[Dict]:
        if not self._queue:
            return None
        item = self._queue.popleft()
        self._queued.remove(item["entity"])
        self.state["to_explore"].pop(0)
        return item

    def log_qa(
        self,
        question: str,
        student_answer: str,
        teacher_answer: str,
        difference: str = "",
        followup_question: str = "",
        final_note: str = "",
    ):
        self.state["qa_log"].append(
            {
                "question": question,
                "student_answer": student_answer,
                "teacher_answer": teacher_answer,
                "difference": difference,
                "followup_question": followup_question,
                "final_note": final_note,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def mark_explored(self, entity: str, source: str = "unknown"):
        if self.is_explored(entity):
            return
        self._visited.add(entity)
        self.state["explored_entities"].append(
            {
                "entity": entity,
                "source": source,
                "timestamp": datetime.now().isoformat(),
            }
        )
    def peek_all(self):
        return [item["entity"] for item in self._queue]
    
    def pop_specific(self, entity: str):
        for i, item in enumerate(self._queue):
            if item["entity"] == entity:
                self._queue.rotate(-i)
                popped = self._queue.popleft()
                self._queue.rotate(i)
                return popped
        return None
    
    def save(self, filepath: str = "exploration_map.json"):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def load(self, filepath: str = "exploration_map.json"):
        if not os.path.isfile(filepath):
            return
        with open(filepath, "r", encoding="utf-8") as f:
            self.state = json.load(f)
        self._queue.clear()
        self._visited.clear()
        self._queued.clear()
        self.entity_meta.clear()

        for item in self.state["explored_entities"]:
            self._visited.add(item["entity"])
        for item in self.state["to_explore"]:
            self._queue.append(item)
            self._queued.add(item["entity"])
    def get_source(self, entity: str) -> Optional[str]:
        for item in reversed(self.state["to_explore"]): 
            if item["entity"] == entity:
                return item.get("source", None)
        return None


class PriorityDock:
    def __init__(self):
        self._heap = []            
        self._counter = itertools.count()
        self._in_heap = set()       

    def push(self, question, student_answer, ppl):
        if question in self._in_heap:
            return
        heapq.heappush(self._heap, (-ppl, next(self._counter),
                                    {"q": question, "ans": student_answer, "ppl": ppl}))
        self._in_heap.add(question)

    def pop(self):
        if not self._heap:
            return None
        _, _, item = heapq.heappop(self._heap)
        self._in_heap.remove(item["q"])
        return item                 

    def __len__(self):
        return len(self._heap)


class QADataset(Dataset):
    def __init__(self, qa_pairs: List[Dict], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for qa in qa_pairs:
            prompt = qa["prompt"]
            answer = qa["teacher_answer"]
            input_text = prompt + " " + answer
            tokenized = self.tokenizer(
                input_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            input_ids = tokenized.input_ids.squeeze()
            attention_mask = tokenized.attention_mask.squeeze()
            labels = input_ids.clone()
            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {
            "input_ids": self.examples[idx]["input_ids"],
            "attention_mask": self.examples[idx]["attention_mask"],
            "labels": self.examples[idx]["labels"]
        }


