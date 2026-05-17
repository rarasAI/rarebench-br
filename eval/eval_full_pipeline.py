"""A/B benchmark: araras-pipeline (HPO + Gemma + ORPHA lookup + PCDT) vs Gemma 4 raw.

Roda em 24 casos do L5 (2 por doença x 12 doenças). Mede:
- R@1 / R@3 via canonical name match (raw output)
- R@1 / R@3 via ORPHA lookup post-processor (pipeline strict)
- Track B preview: SUS-conduta correta (top-1 drug em expected_pcdt_therapy_ceaf)
"""
from __future__ import annotations
import sys, json, time, random, re, unicodedata
sys.path.insert(0, "/tmp")
import araras_pipeline as P

random.seed(11)
L5 = "/Users/dimas/rarasnet-swarm-py/rarebench_br/cases/L5_realsus.jsonl"
OUT = "/tmp/bench_pipeline_l5.json"

def norm(s):
    s = (s or "").lower()
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")

cases = [json.loads(l) for l in open(L5)]
by_d = {}
for c in cases:
    by_d.setdefault(c["ground_truth"]["primary_orphanet"], []).append(c)
sample = []
for d, lst in by_d.items():
    sample += random.sample(lst, min(2, len(lst)))
print(f"[start] {len(sample)} cases across {len(by_d)} diseases")

results = []
metric_raw = {"R1": 0, "R3": 0}
metric_pipeline_strict = {"R1": 0, "R3": 0}
track_b_hits = 0
track_b_valid = 0
latencies = {"hpo": [], "gemma": [], "total": []}

for i, case in enumerate(sample, 1):
    gold_orpha = case["ground_truth"]["primary_orphanet"]
    gold_name = case["ground_truth"]["primary_name_pt"]
    expected_drugs = case["ground_truth"]["expected_pcdt_therapy_ceaf"] or []

    try:
        r = P.inference(case["free_text_pt"])
    except Exception as e:
        print(f"  [{i:>2}] {case['case_id']} ERROR: {e}")
        continue

    raw = r["stage2_gemma_raw"]
    ranked_orphas = r["stage3_ranked_orphas"]
    pred_top1 = ranked_orphas[0]["orpha"] if ranked_orphas else None
    pred_top3 = [o["orpha"] for o in ranked_orphas[:3]]

    # Raw name match (any line)
    name_hit = any(kw in norm(raw) for kw in [norm(gold_name), gold_orpha.lower()])
    if name_hit:
        metric_raw["R1"] += 1  # imprecise but consistent w/ earlier bench
        metric_raw["R3"] += 1

    # Pipeline strict ORPHA match
    if pred_top1 == gold_orpha: metric_pipeline_strict["R1"] += 1
    if gold_orpha in pred_top3: metric_pipeline_strict["R3"] += 1

    # Track B: SUS conduta — did pipeline's top-1 ceaf_drugs intersect expected?
    if ranked_orphas:
        pred_drugs = ranked_orphas[0].get("ceaf_drugs", [])
        if expected_drugs and pred_drugs:
            track_b_valid += 1
            # any expected drug substring-matched in any predicted drug
            if any(norm(ed).split("-")[0] in norm(pd) or norm(pd).split("-")[0] in norm(ed)
                   for ed in expected_drugs for pd in pred_drugs):
                track_b_hits += 1

    latencies["hpo"].append(r["timings_s"]["hpo"])
    latencies["gemma"].append(r["timings_s"]["gemma"])
    latencies["total"].append(r["timings_s"]["total"])

    mark_pipeline = "✓" if pred_top1 == gold_orpha else ("~" if gold_orpha in pred_top3 else "✗")
    mark_name = "name+" if name_hit else "name-"
    print(f"  [{i:>2}/{len(sample)}] {case['case_id']} gold={gold_orpha}({gold_name[:20]:>20}) | pipeline_top1={pred_top1} {mark_pipeline} | {mark_name} | total={r['timings_s']['total']}s")

    results.append({
        "case_id": case["case_id"],
        "gold_orpha": gold_orpha,
        "gold_name": gold_name,
        "expected_drugs_pcdt": expected_drugs,
        "stage1_hpo_extracted": len(r["stage1_hpo_normalized"]),
        "stage1_hpo_codes": [h["hpo_id"] for h in r["stage1_hpo_normalized"]],
        "stage3_top1_orpha": pred_top1,
        "stage3_top3_orphas": pred_top3,
        "stage3_top1_drugs": ranked_orphas[0].get("ceaf_drugs") if ranked_orphas else None,
        "stage4_sus_conduta": r["stage4_sus_conduta"],
        "match_raw_name": name_hit,
        "match_pipeline_top1": pred_top1 == gold_orpha,
        "match_pipeline_top3": gold_orpha in pred_top3,
        "match_track_b": track_b_hits > 0 and ranked_orphas and any(
            norm(ed).split("-")[0] in norm(pd) or norm(pd).split("-")[0] in norm(ed)
            for ed in expected_drugs for pd in (ranked_orphas[0].get("ceaf_drugs") or [])
        ) if (expected_drugs and ranked_orphas) else None,
        "timings_s": r["timings_s"],
        "raw_first_500": raw[:500],
    })

n = len(results)
summary = {
    "pipeline": "araras-hpo-brasil + araras-gemma4-e4b Q4_K_M (llama.cpp Metal) + canonical ORPHA lookup + PCDT overlay",
    "benchmark_layer": "L5_realsus (DataSUS APAC-anchored)",
    "n_cases": len(sample), "n_valid": n,
    "track_a_diagnosis": {
        "raw_name_match_rate": round(metric_raw["R1"] / max(1, n), 3),
        "pipeline_strict_R1": round(metric_pipeline_strict["R1"] / max(1, n), 3),
        "pipeline_strict_R3": round(metric_pipeline_strict["R3"] / max(1, n), 3),
    },
    "track_b_sus_conduta": {
        "n_evaluable": track_b_valid,
        "n_correct": track_b_hits,
        "pcdt_correct_rate": round(track_b_hits / max(1, track_b_valid), 3),
    },
    "latency_s": {
        "hpo_p50": round(sorted(latencies["hpo"])[len(latencies["hpo"])//2], 2) if latencies["hpo"] else None,
        "gemma_p50": round(sorted(latencies["gemma"])[len(latencies["gemma"])//2], 2) if latencies["gemma"] else None,
        "total_p50": round(sorted(latencies["total"])[len(latencies["total"])//2], 2) if latencies["total"] else None,
    },
    "compare_to_baseline": "Earlier non-pipeline run on 36 cases: R@1 = 58.3%, R@3 = 75% (name-match scoring)",
}
out = {"summary": summary, "results": results}
open(OUT, "w").write(json.dumps(out, ensure_ascii=False, indent=2))
print(f"\n=== SUMMARY ===")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"\nwrote {OUT}")
