"""Bench Gemma 4 on L6/L7/L8 — multiple-choice format.
Each case has 4 options (A/B/C/D); we measure accuracy + Brier-like score on the
extracted answer.
"""
import json, re, time, urllib.request, sys
from pathlib import Path

URL = "http://127.0.0.1:8089/v1/chat/completions"
OUT = "/tmp/bench_l6_l7_l8_RESULTS.json"

LAYER_FILES = {
    "L6_trajectory_forecast": "/Users/dimas/rarasnet-swarm-py/rarebench_br/cases/L6_trajectory_forecast.jsonl",
    "L7_geographic_equity":   "/Users/dimas/rarasnet-swarm-py/rarebench_br/cases/L7_geographic_equity.jsonl",
    "L8_persistence":         "/Users/dimas/rarasnet-swarm-py/rarebench_br/cases/L8_persistence.jsonl",
}

SYS = """Você é ARARAS, copiloto clínico de doenças raras + SUS em PT-BR.
Você recebe um caso clínico com 4 opções A/B/C/D.
Responda com a letra correspondente à melhor opção, justificando em 1-2 frases.
FORMATO obrigatório no início da resposta:
RESPOSTA: <letra>
Justificativa: ..."""

LETTER_RE = re.compile(r"RESPOSTA\s*[:\-]?\s*\(?([A-D])\)?", re.I)
LETTER_FALLBACK = re.compile(r"\b([A-D])\b")

def call_gemma(scenario, question, options):
    user = f"{scenario}\n\nPergunta: {question}\n\n" + "\n".join(options)
    body = {
        "model": "araras",
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2, "top_p": 0.95, "top_k": 64, "repeat_penalty": 1.15,
        "max_tokens": 300,
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"].get("content") or "", time.time()-t0

def extract_letter(text):
    m = LETTER_RE.search(text)
    if m: return m.group(1).upper()
    # fallback — first A-D in first 150 chars
    m = LETTER_FALLBACK.search(text[:200])
    if m: return m.group(1).upper()
    return None

def bench_layer(name, path, max_n=None):
    cases = [json.loads(l) for l in open(path)]
    if max_n: cases = cases[:max_n]
    print(f"\n=== {name}: {len(cases)} cases ===", flush=True)
    hits = 0; errors = 0; lats = []
    pred_dist = {"A":0,"B":0,"C":0,"D":0,"?":0}
    gold_dist = {"A":0,"B":0,"C":0,"D":0}
    rows = []
    t0 = time.time()
    for i, c in enumerate(cases, 1):
        gold = c["ground_truth"]["expected_letter"]
        gold_dist[gold] = gold_dist.get(gold, 0) + 1
        try:
            raw, dt = call_gemma(c["scenario"], c["question"], c["options"])
            lats.append(dt)
        except Exception as e:
            errors += 1
            continue
        pred = extract_letter(raw) or "?"
        pred_dist[pred] = pred_dist.get(pred, 0) + 1
        ok = pred == gold
        if ok: hits += 1
        rows.append({"id": c["case_id"], "gold": gold, "pred": pred, "ok": ok, "dt": round(dt,2)})
        if i % 20 == 0 or i == len(cases):
            n = i - errors
            print(f"  [{i:>3}/{len(cases)}] acc={hits/max(n,1):.1%}  pred={pred_dist}  ({(time.time()-t0)/60:.1f}min)", flush=True)
    n = max(1, len(cases)-errors)
    return {
        "layer": name, "n": len(cases), "valid": n, "errors": errors,
        "accuracy": round(hits/n, 3),
        "predicted_distribution": pred_dist,
        "gold_distribution": gold_dist,
        "lat_p50": round(sorted(lats)[len(lats)//2], 1) if lats else None,
        "rows": rows,
    }

results = {}
for name, path in LAYER_FILES.items():
    results[name] = bench_layer(name, path, max_n=60)  # 60 per layer for speed

print("\n=== SUMMARY ===")
print(f"{'Layer':30s}  {'N':>4}  {'Acc':>7}  {'p50':>6}")
for name, r in results.items():
    print(f"{name:30s}  {r['valid']:>4}  {r['accuracy']*100:>6.1f}%  {r['lat_p50']}s")

open(OUT, "w").write(json.dumps(results, ensure_ascii=False, indent=2))
print(f"\nwrote {OUT}")
