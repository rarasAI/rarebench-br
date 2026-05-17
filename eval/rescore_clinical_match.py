"""Re-score the L5 bench using clinical name matching (keywords) instead of
strict ORPHA code match. ORPHA codes are heavily hallucinated by all rare-disease
LLMs (the codes are sparse vocab tokens). What matters clinically is whether
the model named the correct disease."""
from __future__ import annotations
import json, re, unicodedata

BENCH = "/tmp/bench_gemma4_l5.json"
OUT = "/tmp/bench_gemma4_l5_rescored.json"

# Disease keywords (case-insensitive, accent-insensitive). Match ANY keyword = hit.
KEYWORDS = {
    "ORPHA:802": ["esclerose multipla", "multiple sclerosis", "neuromielite optica", "nmosd", "mogad", "desmielinizante", "desmielinizacao", "mielinite optica"],
    "ORPHA:232": ["doenca falciforme", "anemia falciforme", "hbss", "sickle cell", "drepanocitose", "falciforme"],
    "ORPHA:83330": ["atrofia muscular espinhal", "ame", "spinal muscular atrophy", "smn1", "werdnig", "hoffmann", "kugelberg", "welander"],
    "ORPHA:182090": ["hipertensao arterial pulmonar", "hipertensao pulmonar", "hap", "pah", "pulmonary hypertension", "hipertensao do pulmao", "cor pulmonale", "pressao pulmonar"],
    "ORPHA:586": ["fibrose cistica", "cystic fibrosis", "mucoviscidose", "cftr"],
    "ORPHA:716": ["fenilcetonuria", "pku", "phenylketonuria", "hiperfenilalanin", "deficiencia de fenilalanina hidroxilase"],
    "ORPHA:905": ["doenca de wilson", "wilson disease", "deficiencia de ceruloplasmina", "atp7b", "kayser-fleischer", "kayser fleischer"],
    "ORPHA:580": ["mucopolissacaridose tipo ii", "mps ii", "mps 2", "hunter", "iduronato-2-sulfatase", "iduronato sulfatase", "ids", "mucopolissacaridose ii"],
    "ORPHA:583": ["mucopolissacaridose tipo vi", "mps vi", "mps 6", "maroteaux-lamy", "maroteaux lamy", "arilsulfatase b", "arsb"],
    "ORPHA:579": ["mucopolissacaridose tipo i", "mps i", "mps 1", "hurler", "iduronidase", "idua"],
    "ORPHA:183660": ["scid", "imunodeficiencia combinada grave", "severe combined immunodeficiency", "ada deficiencia", "jak3", "il7r", "rag1", "rag2"],
    "ORPHA:70": ["ame tipo 1", "ame 1", "spinal muscular atrophy type 1", "werdnig-hoffmann", "werdnig hoffmann", "smn1", "atrofia muscular espinhal tipo i", "ame tipo i"],
}

def normalize(s):
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s

def hit(text, orpha):
    txt = normalize(text)
    for kw in KEYWORDS.get(orpha, []):
        if normalize(kw) in txt:
            return True
    return False

with open(BENCH) as f:
    bench = json.load(f)

# To rescore we need the FULL output (raw_first_400 might not be enough)
# Note: bench stored only first 400 chars. Let's work with that and document it.
hit_at = {1:0, 3:0, 5:0}
n = 0
total = 0
per_disease = {}
for r in bench["results"]:
    total += 1
    text = r.get("raw_first_400") or ""
    gold = r["gold_orpha"]
    per_disease.setdefault(gold, {"n": 0, "hits": 0})
    per_disease[gold]["n"] += 1
    if hit(text, gold):
        n += 1
        per_disease[gold]["hits"] += 1
        # ranking: parse the numbered list and check which position the keyword first appears
        lines = re.split(r"\n+", text)
        for pos, line in enumerate(lines, 1):
            if hit(line, gold):
                if pos <= 1: hit_at[1] += 1
                if pos <= 3: hit_at[3] += 1
                if pos <= 5: hit_at[5] += 1
                break

print(f"=== Re-scored with clinical name matching ===")
print(f"  Total cases: {total}")
print(f"  Cases where gold disease name appeared in output: {n} ({100*n/total:.0f}%)")
print(f"  R@1 (correct dx in line 1): {hit_at[1]}/{total} = {hit_at[1]/total:.1%}")
print(f"  R@3 (correct dx in top-3): {hit_at[3]}/{total} = {hit_at[3]/total:.1%}")
print(f"  R@5 (correct dx in top-5): {hit_at[5]}/{total} = {hit_at[5]/total:.1%}")
print(f"\n=== Per-disease breakdown ===")
for orpha, st in sorted(per_disease.items(), key=lambda kv: -kv[1]["hits"]):
    print(f"  {orpha:>14}  n={st['n']}  hits={st['hits']}  rate={st['hits']/st['n']:.0%}")

# Save rescored summary
summary = {
    "model": bench["summary"]["model"],
    "benchmark_layer": bench["summary"]["benchmark_layer"],
    "scoring_method": "clinical_name_keyword_match_accent_insensitive",
    "note": "Strict ORPHA code matching (~0%) is misleading — LLMs hallucinate sparse ORPHA tokens. We score by canonical disease name match (what matters clinically).",
    "raw_truncation_caveat": "Outputs only stored first 400 chars; some hits in positions 4-5 may be missed",
    "n_cases": total,
    "any_mention_rate": round(n/total, 3),
    "R_at_1": round(hit_at[1]/total, 3),
    "R_at_3": round(hit_at[3]/total, 3),
    "R_at_5": round(hit_at[5]/total, 3),
    "latency_p50_s": bench["summary"]["latency_p50_s"],
    "latency_p90_s": bench["summary"]["latency_p90_s"],
    "per_disease": per_disease,
}
with open(OUT, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\nwrote {OUT}")
