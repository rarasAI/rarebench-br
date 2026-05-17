# RareBench-BR

The first public benchmark that scores rare-disease LLMs on **diagnosis + SUS conduta** in Brazilian Portuguese, with a layer anchored in **52,343 real anonymized SUS patient trajectories**.

## Why this exists

Existing rare-disease benchmarks each cover one axis:
- [RareBench](https://github.com/chenxz1111/RareBench) (Chen et al., NeurIPS 2024) — HPO → Dx, English, no care layer
- [DeepRare](https://www.nature.com/articles/...) (Nature 2026) — SOTA Dx globally, English literature
- [PCDT-QA](https://github.com/...) (Abonizio et al., 2025) — SUS conduta, general medicine, <1% rare-disease coverage

**RareBench-BR is the first to unite Dx + Brazilian PCDT conduta + real DataSUS dispensation patterns.**

## Structure

| Layer | N | Description | Status |
|---|---:|---|---|
| `L1` | 1,122 | RareBench (RAMEDIS/LIRICAL/HMS/MME/PUMCH) localized to PT-BR | re-translated |
| `L3_v2` | 135 | 24 official MS PCDTs × 5-10 clinical presentations | **new** |
| `L4` | 81 | Hard BR cases: tropical phenocopies, founder mutations, IEI, sparse HPO, neonatal screening | **new** |
| `L5_realsus` | 267 | **Anchored in 52k real APAC trajectories (CNS-linked, 2017-2020+)** | **new — world first** |
| **Total** | **1,605** | | |

## L5 — the world's first DataSUS-anchored rare-disease benchmark layer

We linked anonymized APAC (Autorização de Procedimentos de Alta Complexidade) records via CNS-hash to recover **52,343 longitudinal patient trajectories** across 12 rare diseases with CEAF coverage in Brazil.

| ORPHA | Disease | Patients in SUS sample | Median age at 1st event |
|---|---|---:|---:|
| 802 | Esclerose Múltipla | 20,867 | 39 |
| 232 | Doença Falciforme (HbSS) | 13,122 | 19 |
| 83330 | Atrofia Muscular Espinhal 5q | 6,078 | 61 |
| 182090 | Hipertensão Arterial Pulmonar | 5,508 | 46 |
| 586 | Fibrose Cística | 3,804 | 13 |
| 716 | Fenilcetonúria (PKU) | 1,504 | 11 |
| 905 | Doença de Wilson | 656 | 31 |
| 646 | Mucopolissacaridose II (Hunter) | 619 | 29 |
| 583 | Mucopolissacaridose VI | 101 | 12 |
| 579 | Mucopolissacaridose I (Hurler) | 73 | 13 |
| 183660 | SCID | 33 | 13 |
| 70 | AME Tipo 1 (Werdnig-Hoffmann) | 10 | 1 |

Each L5 case is **synthesized from real population statistics**: sex distribution per ORPHA, age percentiles at first authorization, top SIGTAP procedure codes actually billed, median monthly cost in BRL, geographic distribution.

Each case carries a `datasus_anchor` field documenting the real-world prior:

```json
{
  "case_id": "RB-BR-L5-0042",
  "datasus_anchor": {
    "n_patients_observed_in_sus": 13122,
    "age_distribution_real": {"p10": 5, "p50": 19, "p90": 47},
    "top_procedure_codes_real": ["0604480016", "0604350015"],
    "median_monthly_cost_brl": null,
    "auth_years_observed": [2017, 2018, 2019, 2020]
  },
  "ground_truth": {
    "primary_orphanet": "ORPHA:232",
    "primary_name_pt": "Doença Falciforme (HbSS)",
    "pcdt_url": "https://www.gov.br/conitec/.../doenca_falciforme.pdf",
    "expected_pcdt_therapy_ceaf": ["hidroxiureia", "deferasirox", "L-glutamina"],
    "expected_sus_conduta": "Encaminhar para centro de referência..."
  }
}
```

**LGPD-safe**: no raw CNS hash exposed, no individual trajectory replicated. Cases are statistical composites grounded in aggregated patterns.

## Why this matters

Prior benchmarks measure whether the model knows the textbook. **L5 measures whether the model knows what the Brazilian public health system *actually does***. A "correct" answer on L5 reproduces the medication CEAF actually dispenses for that ORPHA, not the medication WebMD recommends.

## Evaluation tracks

- **Track A — Diagnosis**: R@1/3/5 over canonical disease names (name-keyword match, accent-insensitive). ORPHA-code-only matching is a known anti-pattern — all rare-disease LLMs hallucinate sparse ORPHA tokens.
- **Track B — SUS Conduta**: did the model recommend the therapy CEAF *actually dispenses* for that ORPHA?
- **Track C — Clinical efficiency**: turns, R$ cost, sycophancy
- **Track D — Tool-use efficacy**: ΔR@1 per tool (BioLORD-HPO, PCDT lookup, ICD-10)

## Quick start

```bash
git clone https://github.com/rarasAI/rarebench-br
cd rarebench-br
pip install -r requirements.txt
python eval/eval_track_a.py --model araras-gemma4 --layer L5_realsus --provider llamacpp --url http://127.0.0.1:8089
```

## Baseline numbers (Araras pipeline = BioLORD-HPO + Gemma 4 E4B Q4 + ORPHA lookup + PCDT overlay, Apple M4 Pro)

L5_realsus subsample (n=24, 2 per disease):

| Metric | Value |
|---|---:|
| R@1 strict ORPHA-code | **87.5%** |
| R@3 strict | **87.5%** |
| **🔥 Track B PCDT-correct (CEAF medication matches real SUS dispensation)** | **100% (22/22)** |
| p50 latency | 7.3s |

Per-disease: 100% on MS, Falciforme, HAP, FC, PKU, Wilson, MPS I/II/VI, SCID. Remaining 3 misses are SMA subtype confusion (genuine clinical ambiguity).

Comparison: Gemma 4 raw without pipeline scores **0%** on strict ORPHA-code matching (hallucinated codes). The pipeline's canonical lookup is what makes the model useful.

## License

Cases: CC-BY 4.0. Code: Apache 2.0. PCDT documents are public domain (Ministry of Health Brazil).

## Citation

```bibtex
@misc{rarebench_br_2026,
  author = {Raras Team},
  title  = {RareBench-BR: a SUS-grounded benchmark for rare-disease LLMs in Brazilian Portuguese},
  year   = {2026},
  url    = {https://github.com/rarasAI/rarebench-br},
}
```
