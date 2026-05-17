"""Analisa /tmp/datasus_patient_trajectories_v2.json — extrai padrões reais SUS
pra ancorar geração de casos sintéticos do RareBench-BR L5.

Output: /tmp/sus_patterns_v1.json com per-ORPHA statistics.
"""
from __future__ import annotations
import json, os, statistics, time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

SRC = "/tmp/datasus_patient_trajectories_v2.json"
OUT = "/tmp/sus_patterns_v1.json"

t0 = time.time()
print(f"[1/4] loading {SRC} (337MB)...", flush=True)
with open(SRC) as f:
    pts = json.load(f)
print(f"  loaded {len(pts)} trajectories in {time.time()-t0:.1f}s", flush=True)

# Per-ORPHA aggregation
by_orpha = defaultdict(lambda: {
    "n_patients": 0,
    "ages_at_first_event": [],
    "sex_counts": Counter(),
    "uf_counts": Counter(),
    "cids": Counter(),
    "procedure_codes": Counter(),
    "n_events_per_patient": [],
    "trajectory_span_months": [],
    "is_orphan_drug_count": 0,
    "monthly_costs_brl": [],
    "auth_years": Counter(),
    "patient_examples": [],  # store small sample for case template
})

print("[2/4] aggregating per-ORPHA...", flush=True)
for p in pts:
    orphas = p.get("orphas") or []
    if not orphas:
        continue
    for orpha in orphas:
        s = by_orpha[orpha]
        s["n_patients"] += 1
        s["sex_counts"][p.get("sex", "?")] += 1
        # estimate age at first event from birth_year + first_year
        if p.get("birth_year") and p.get("first_year"):
            age_first = p["first_year"] - p["birth_year"]
            if 0 <= age_first <= 110:
                s["ages_at_first_event"].append(age_first)
        s["n_events_per_patient"].append(p.get("n_events", 0))
        if p.get("first_year") and p.get("last_year"):
            s["trajectory_span_months"].append((p["last_year"] - p["first_year"] + 1) * 12)
        for ev in p.get("events", []) or []:
            s["uf_counts"][ev.get("uf_code") or "?"] += 1
            if ev.get("cid"):
                s["cids"][ev["cid"]] += 1
            if ev.get("procedure_code"):
                s["procedure_codes"][ev["procedure_code"]] += 1
            if ev.get("is_orphan_drug"):
                s["is_orphan_drug_count"] += 1
            cost = ev.get("monthly_cost_brl")
            if cost is not None and cost > 0:
                s["monthly_costs_brl"].append(cost)
            if ev.get("year"):
                s["auth_years"][ev["year"]] += 1
        # store one anonymized template per ORPHA (drop CNS hashes)
        if len(s["patient_examples"]) < 3:
            ex = {
                "sex": p.get("sex"),
                "birth_year": p.get("birth_year"),
                "n_events": p.get("n_events"),
                "first_year": p.get("first_year"),
                "last_year": p.get("last_year"),
                "events_summary": [
                    {k: v for k, v in ev.items() if k not in ("cns_hash", "patient_id")}
                    for ev in (p.get("events", [])[:5] or [])
                ],
            }
            s["patient_examples"].append(ex)

print(f"  {len(by_orpha)} unique ORPHAs encountered", flush=True)

# Build final stats
print("[3/4] computing percentiles + ranking...", flush=True)
def pct(arr, p):
    if not arr: return None
    arr = sorted(arr)
    k = max(0, min(len(arr)-1, int(p * (len(arr)-1))))
    return arr[k]

final = {}
for orpha, s in by_orpha.items():
    if s["n_patients"] < 5:  # too rare to extract pattern
        continue
    final[orpha] = {
        "orpha": f"ORPHA:{orpha}",
        "n_patients_in_sus": s["n_patients"],
        "age_at_first_event": {
            "p10": pct(s["ages_at_first_event"], 0.10),
            "p50": pct(s["ages_at_first_event"], 0.50),
            "p90": pct(s["ages_at_first_event"], 0.90),
            "median": int(statistics.median(s["ages_at_first_event"])) if s["ages_at_first_event"] else None,
        },
        "sex_distribution": dict(s["sex_counts"]),
        "top_5_ufs": dict(s["uf_counts"].most_common(5)),
        "top_5_cids": dict(s["cids"].most_common(5)),
        "top_5_procedures": dict(s["procedure_codes"].most_common(5)),
        "median_events_per_patient": int(statistics.median(s["n_events_per_patient"])) if s["n_events_per_patient"] else None,
        "median_trajectory_months": int(statistics.median(s["trajectory_span_months"])) if s["trajectory_span_months"] else None,
        "orphan_drug_event_rate": round(s["is_orphan_drug_count"] / max(1, sum(s["n_events_per_patient"])), 3),
        "monthly_cost_brl_median": int(statistics.median(s["monthly_costs_brl"])) if s["monthly_costs_brl"] else None,
        "auth_years_active": sorted(s["auth_years"].keys()),
        "template_cases": s["patient_examples"],
    }

# Rank by patient count (top 50)
ranked = sorted(final.items(), key=lambda kv: -kv[1]["n_patients_in_sus"])

print(f"[4/4] writing {OUT}", flush=True)
result = {
    "source": "datasus_apac_cns_linkage_v2",
    "n_trajectories_total": len(pts),
    "n_orphas_with_5plus_patients": len(final),
    "generation_date": datetime.now().isoformat(),
    "top_50_orphas_by_volume": [
        {"orpha": k, "n_patients": v["n_patients_in_sus"], "top_uf": list(v["top_5_ufs"].keys())[0] if v["top_5_ufs"] else None}
        for k, v in ranked[:50]
    ],
    "per_orpha_patterns": dict(ranked),
}
with open(OUT, "w") as f:
    json.dump(result, f, indent=2, default=str, ensure_ascii=False)
print(f"DONE in {time.time()-t0:.1f}s — wrote {OUT}", flush=True)
print(f"  {len(final)} ORPHAs with ≥5 patients")
print(f"  TOP 10:")
for k, v in ranked[:10]:
    print(f"    ORPHA:{k} — {v['n_patients_in_sus']} pacientes — top UF: {list(v['top_5_ufs'].keys())[0] if v['top_5_ufs'] else '?'} — age p50: {v['age_at_first_event']['p50']}")
