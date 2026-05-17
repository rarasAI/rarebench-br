"""Roda Gemma 4 (llama-server local) em sub-amostra do L5_realsus.
Métricas: R@1, R@3, R@5 (matching ORPHA code OR primary_name_pt fuzzy).
Output: /tmp/bench_gemma4_l5.json
"""
from __future__ import annotations
import json, re, time, urllib.request, random
from pathlib import Path

random.seed(7)

L5 = "/Users/dimas/rarasnet-swarm-py/rarebench_br/cases/L5_realsus.jsonl"
OUT = "/tmp/bench_gemma4_l5_full.json"
URL = "http://127.0.0.1:8089/v1/chat/completions"
N_SAMPLES = 999  # full L5

SYS = """Você é ARARAS, copiloto clínico de doenças raras em PT-BR.
Para o caso clínico abaixo, liste o TOP-5 diferencial em ordem de probabilidade.
FORMATO OBRIGATÓRIO (uma linha por diagnóstico):
1. Nome da doença (ORPHA:XXXX) — Justificativa breve
2. Nome da doença (ORPHA:XXXX) — Justificativa breve
3. ...
NÃO escreva análise extensa, vá direto ao formato."""

ORPHA_RE = re.compile(r"ORPHA[:\s]*(\d{1,7})", re.I)
NUM_LINE = re.compile(r"^\s*(\d+)\.\s*([^(]+)\((ORPHA:\d+)", re.M)

def call(case):
    body = {
        "model": "araras",
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": case["free_text_pt"]},
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "repeat_penalty": 1.15,
        "max_tokens": 600,
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        resp = json.loads(r.read())
    dt = time.time() - t0
    msg = resp["choices"][0]["message"]
    content = msg.get("content") or ""
    finish = resp["choices"][0].get("finish_reason")
    return content, dt, finish

def extract_orphas(text):
    """Extract ranked list of ORPHA codes from model output."""
    out = []
    seen = set()
    # numbered lines first
    for m in NUM_LINE.finditer(text):
        c = m.group(3)
        if c not in seen:
            seen.add(c); out.append(c)
    # fallback: all ORPHA mentions in order
    for m in ORPHA_RE.finditer(text):
        c = f"ORPHA:{m.group(1)}"
        if c not in seen:
            seen.add(c); out.append(c)
    return out[:10]

# Load + subsample (3 per disease)
cases = [json.loads(l) for l in open(L5)]
by_d = {}
for c in cases:
    by_d.setdefault(c["ground_truth"]["primary_orphanet"], []).append(c)
sample = cases  # full L5
print(f"[start] {len(sample)} cases across {len(by_d)} diseases")

results = []
hit_at = {1:0, 3:0, 5:0}
n_valid = 0
errors = 0
latencies = []
for i, case in enumerate(sample, 1):
    gold = case["ground_truth"]["primary_orphanet"]
    gold_name = case["ground_truth"]["primary_name_pt"]
    try:
        content, dt, finish = call(case)
        latencies.append(dt)
    except Exception as e:
        errors += 1
        print(f"  [{i:>2}/{len(sample)}] {case['case_id']} ERROR: {e}")
        continue
    ranked = extract_orphas(content)
    name_match = gold_name.lower() in content.lower() if gold_name else False
    hit1 = (gold in ranked[:1]) or (name_match and not ranked)
    hit3 = (gold in ranked[:3]) or (name_match and not ranked)
    hit5 = (gold in ranked[:5]) or (name_match and not ranked)
    if ranked or name_match:
        n_valid += 1
        hit_at[1] += int(hit1); hit_at[3] += int(hit3); hit_at[5] += int(hit5)
    mark = "✓" if hit1 else ("~" if hit5 else "✗")
    print(f"  [{i:>2}/{len(sample)}] {case['case_id']} gold={gold} pred={ranked[:3]} {mark} {dt:.1f}s")
    results.append({
        "case_id": case["case_id"],
        "gold_orpha": gold,
        "gold_name": gold_name,
        "predicted_ranked": ranked,
        "name_in_text": name_match,
        "hit_at_1": hit1,
        "hit_at_3": hit3,
        "hit_at_5": hit5,
        "latency_s": dt,
        "finish_reason": finish,
        "raw_full": content,
    })

n = max(1, len(sample) - errors)
summary = {
    "model": "araras-gemma4-e4b-v4-Q4_K_M (llama.cpp / Metal / Apple M4 Pro)",
    "benchmark_layer": "L5_realsus (DataSUS APAC-anchored)",
    "n_cases": len(sample),
    "n_valid_responses": n_valid,
    "errors": errors,
    "R_at_1": round(hit_at[1] / n, 3),
    "R_at_3": round(hit_at[3] / n, 3),
    "R_at_5": round(hit_at[5] / n, 3),
    "latency_p50_s": round(sorted(latencies)[len(latencies)//2], 1) if latencies else None,
    "latency_p90_s": round(sorted(latencies)[int(len(latencies)*0.9)], 1) if latencies else None,
    "generation_params": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "repeat_penalty": 1.15, "max_tokens": 600},
}
out = {"summary": summary, "results": results}
Path(OUT).write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(f"\n=== SUMMARY ===")
print(json.dumps(summary, indent=2))
print(f"\nwrote {OUT}")
