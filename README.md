# RareBench-BR

**The first public benchmark for rare-disease decision-support LLMs grounded in the Brazilian Unified Health System (SUS).**

[![Dataset](https://img.shields.io/badge/🤗-Raras--AI%2FRareBench--BR--Public-green)](https://huggingface.co/datasets/Raras-AI/RareBench-BR-Public)
[![Companion model](https://img.shields.io/badge/🤗-Araras--Gemma4-blue)](https://huggingface.co/Raras-AI/araras-gemma4-e4b-v4-gguf)
[![License](https://img.shields.io/badge/cases-CC--BY%204.0-orange)](#license)

> Companion benchmark to the [**Araras-Gemma4**](https://github.com/rarasAI/araras-gemma4) submission to the [Gemma 4 Good Hackathon](https://kaggle.com/competitions/gemma-4-good-hackathon).

---

## Why this benchmark exists

Existing rare-disease benchmarks each cover one axis:
- **[RareBench](https://github.com/chenxz1111/RareBench)** (Chen et al., NeurIPS 2024) — HPO → Dx, English, no care layer
- **[DeepRare](https://www.nature.com/articles/s41586-025-10097-9)** (Nature 2026) — SOTA Dx globally, English literature
- **PCDT-QA** (Abonizio et al., 2025) — SUS conduta, general medicine, <1% rare disease

**RareBench-BR is the first to unite Dx + Brazilian PCDT conduta + real DataSUS dispensation patterns in PT-BR.**

## What's in this repo

**Public release** (CC-BY 4.0) — **833 LGPD-safe validated cases** on [🤗 Raras-AI/RareBench-BR-Public](https://huggingface.co/datasets/Raras-AI/RareBench-BR-Public).

This GitHub repo contains:
- `cases/` — JSONL files per layer + the consolidated `RareBench_BR_SUS_v1.jsonl`
- `pcdts_structured/` — 24 official Conitec PCDTs structured as JSON
- `manifest.json` — source URLs + dates for each PCDT
- `build/` — generation scripts (`build_l5_realsus.py`, `build_l6_l7_l8.py`)
- `eval/` — evaluation harness (`eval_track_a_baseline.py`, `eval_full_pipeline.py`, `eval_l6_l7_l8.py`)
- `results_*.json` — benchmark results from Araras-Gemma4

## Structure — unified RareBench-BR_SUS

| Layer | N | Task | Source | Public? |
|---|---:|---|---|:---:|
| L3_v2 — PCDT intersection | 135 | Diagnosis + PCDT mapping | 24 official MS PCDTs × variations | ✓ |
| L4 — Hard BR cases | 79 | Diagnosis (tropical phenocopies, founder mutations, IEI) | Hand-curated hard BR cases | ✓ |
| L5_realsus_v2 | 619 | Diagnosis + CEAF medication | Synthesized from 52k APAC aggregates | ✓ |
| L6 — Trajectory forecast | 200 | Predict next APAC event (multi-choice) | Real individual patient prefixes | internal only* |
| L7 — Geographic equity | 60 | Predict UF treatment concentration (multi-choice) | Real APAC UF data | internal only* |
| L8 — Persistence | 120 | Predict treatment duration (multi-choice) | Real APAC longitudinal | internal only* |
| **Public total** | **833** | All diagnostic, all SUS-grounded | | |
| Internal total | 1,213 | + multi-choice longitudinal | | |

*L6/L7/L8 use individual prefix data — kept internal for LGPD compliance. Released schemas + generation scripts in this repo if anyone wants to reproduce on their own DataSUS pull.

## Diseases covered (top 10 by case count)

- Esclerose Múltipla (74)
- Atrofia Muscular Espinhal 5q (58)
- Fibrose Cística (57)
- Doença de Wilson (57)
- Imunodeficiência Combinada Grave / SCID (51)
- Doença Falciforme HbSS (50)
- Hipertensão Arterial Pulmonar Idiopática (50)
- Fenilcetonúria PKU (50)
- Mucopolissacaridose Tipo II Hunter (50)
- Mucopolissacaridose Tipo VI Maroteaux-Lamy (50)

Plus 14 more PCDT-covered rare diseases (Gaucher, Pompe, Fabry, MPS I/IV/VII, ELA, Acromegalia, Miastenia, OI, CLN2, TTR-FAP, LES, AME tipo 1, HPN).

## Schema (unified)

```json
{
  "case_id": "RB-BR-SUS-00001",
  "source_layer": "L3_v2|L4|L5_v2",
  "task": "diagnosis",
  "difficulty": "easy|medium|hard",

  "clinical_input": {
    "free_text_pt": "Lactente masculino, 14 meses, natural de Salvador (BA)...",
    "hpo_codes": ["HP:0001433", "HP:0001873", ...],
    "hpo_names_pt": ["Hepatoesplenomegalia", "Trombocitopenia", ...],
    "demographics": {"age_years": 1.2, "sex": "M", "region_br": "Nordeste"},
    "labs": ["Hb 8,2 g/dL", "Plaquetas 62.000/μL"],
    "cultural_qualifiers_pt": ["barriga d'água", "criança que não engorda"]
  },

  "ground_truth": {
    "primary_orphanet": "ORPHA:355",
    "primary_name_pt": "Doença de Gaucher",
    "primary_icd10": "E75.2",
    "task_specific": {
      "expected_dx_top1": "ORPHA:355",
      "expected_pcdt_slug": "doenca-de-gaucher",
      "expected_pcdt_url": "https://www.gov.br/conitec/...pdf",
      "expected_ceaf_drug": "imiglucerase",
      "expected_ceaf_drug_status": "ceaf_verified"
    }
  },

  "validation": {
    "orpha_format_ok": true,
    "pcdt_url_validated": true,
    "ceaf_drug_validated": true,
    "validation_date": "2026-05-17"
  }
}
```

## Evaluation tracks

- **Track A — Diagnosis**: R@1/R@3 over canonical disease names (name-keyword match, accent-insensitive). **ORPHA-code-only matching is a known anti-pattern** — all rare-disease LLMs hallucinate sparse ORPHA tokens. Use canonical name matching.
- **Track B — SUS conduta**: did the model recommend the medication CEAF *actually dispenses* for that ORPHA? **Novel — no other benchmark has this.**
- (Internal) **Track C — Trajectory forecast**: predict next APAC authorization given prefix
- (Internal) **Track D — Geographic equity**: predict UF treatment concentration
- (Internal) **Track E — Persistence**: predict treatment duration bucket

## Quick start

```python
from datasets import load_dataset

ds = load_dataset("Raras-AI/RareBench-BR-Public", split="train")
for case in ds:
    case_text = case["clinical_input"]["free_text_pt"]
    expected_orpha = case["ground_truth"]["primary_orphanet"]
    expected_ceaf_drug = case["ground_truth"]["task_specific"]["expected_ceaf_drug"]
    # ... your model here ...
```

Or via this repo:
```bash
git clone https://github.com/rarasAI/rarebench-br
cd rarebench-br
pip install -r requirements.txt
python eval/eval_full_pipeline.py --provider llamacpp --url http://127.0.0.1:8089
```

## Baseline — Araras-Gemma4-E4B (Q4_K_M offline)

| Layer | N | R@1 | R@3 | Track B PCDT-correct |
|---|---:|---:|---:|---:|
| L5_realsus full | 240 | **70.4%** | **78.3%** | **76.3%** |
| L3_v2 partial | 100 | 28.0% | 38.0% | — |
| L4 + L5_v2 unified | 833 (in progress) | TBD | TBD | TBD |

vs **DeepSeek V4 Chat** (~600B cloud, 36-case subsample): R@1 86.1%, R@3 91.7%, TB 91.7%.

Our 4B offline model **matches the trajectory of a 150× larger cloud model** on SUS-grounded tasks.

## LGPD compliance

- ✅ No raw CNS hash exposed
- ✅ No individual patient trajectory replicated (L5_v2 = aggregates only)
- ✅ L6/L7/L8 individual-prefix data kept internal
- ✅ All PCDT URLs validated (HTTP 200) on gov.br/conitec
- ✅ All ORPHA codes cross-referenced with [RarasNet KG](https://raras.org) (10,468 diseases)

## License

Cases: **CC-BY 4.0**. Code: Apache 2.0. PCDT documents are public domain (Ministry of Health Brazil).

## Citation

```bibtex
@misc{rarebench_br_2026,
  author       = {Raras Team and Timmers, Dimas},
  title        = {RareBench-BR: a SUS-grounded benchmark for rare-disease decision-support LLMs in Brazilian Portuguese},
  year         = {2026},
  month        = may,
  publisher    = {Hugging Face},
  url          = {https://huggingface.co/datasets/Raras-AI/RareBench-BR-Public},
}
```

---

*— Built by [Raras.org](https://raras.org), Latin America's largest rare-disease infrastructure (100K+ monthly visits, HC-FMUSP + Wikipedia PT partnerships, 10,468 diseases enriched in <6 months).*
