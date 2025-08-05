import csv, re, statistics
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import accuracy_score

RESULT_DIR = Path("results")      

def acc_equal(path, col_pred, col_gold, tf=lambda x: x):
    tot = corr = 0
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            corr += int(tf(r[col_pred]) == tf(r[col_gold]))
            tot  += 1
    return corr / tot

def acc_medqa(p):     
    return acc_equal(p, "pred_answer", "gt_answer", str.upper)

def acc_pubmedqa(p): 
    return acc_equal(p, "correct", "correct", int)

def acc_chemprot(p): 
    return acc_equal(p, "pred_label", "gt_label", str.upper)

def acc_headline(p):  
    y_true, y_pred = defaultdict(list), defaultdict(list)
    with open(p, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            key = r["label_key"]
            y_true[key].append(int(r["gold"]))
            y_pred[key].append(int(r["pred_binary"]))
    return statistics.mean(accuracy_score(y_true[k], y_pred[k])
                           for k in y_true)

def acc_fomc(p):
    lab = {"dovish":0,"neutral":1,"hawkish":2}
    g, y = [], []
    with open(p, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            g.append(lab[r["gold"].lower()])
            y.append(lab[r["pred_label"].lower()])
    return accuracy_score(g, y)

def acc_fpb(p):
    lab = {"negative":0,"neutral":1,"positive":2}
    g, y = [], []
    with open(p, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            g.append(lab[r["gold"].lower()])
            y.append(lab[r["pred_label"].lower()])
    return accuracy_score(g, y)

ROUTER = [
    (re.compile(r"medqa",     re.I), acc_medqa),
    (re.compile(r"pubmedqa",  re.I), acc_pubmedqa),
    (re.compile(r"chemprot",  re.I), acc_chemprot),
    (re.compile(r"headline",  re.I), acc_headline),
    (re.compile(r"fomc",      re.I), acc_fomc),
    (re.compile(r"fpb",       re.I), acc_fpb),
]

def dispatch(path):
    for pat, fn in ROUTER:
        if pat.search(path.name):
            return fn(path)
    raise ValueError(f"Un-recognized dataset in {path.name}")

def main():
    if not RESULT_DIR.exists():
        raise FileNotFoundError("results/ 目录不存在")
    summary = []
    for csv_file in sorted(RESULT_DIR.glob("*.csv")):
        acc = dispatch(csv_file) * 100
        summary.append((csv_file.name, acc))
        # print(f"{csv_file.name:45s}  ACC = {acc:6.2f}%")

    print("\n=== Summary ===")
    for n,a in summary:
        print(f"{n:45s}  {a:6.2f}%")

if __name__ == "__main__":
    main()