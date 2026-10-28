# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Ali Almasri
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/AliAlmasri7/FlyRank-ML
- **Date:** 2026-10-28

## 1. Problem framing

This project supports the decision of which content records a human editor should inspect first. The unit of analysis is a pseudonymous client/content record aggregated over a monthly feature window. The output is a ranked priority list with reason codes and suggested review actions.

An editor uses the ranking to decide whether to review a search-result snippet, inspect content relevance, investigate a performance trend, or continue monitoring.

A wrong high-priority call can waste editorial time, while a wrong low-priority call can delay a useful review. The ranking is therefore a triage aid, not an automatic instruction to change a page. Data and machine learning help summarize historical signals consistently across many records, while a transparent heuristic provides an interpretable baseline.

## 2. Data safety

The analysis uses the gated `FlyRank/internship-warehouse` dataset, release build `flyrank_pseudonymized_warehouse_release_v20260703`, exported on 2026-07-03. The primary table is `fact_content_daily_performance`.

- Reported table size: 78,835,655 rows.
- Feature window: March 2026.
- Outcome window: April 2026.
- Recorded March-to-April join: 331,436 matched records.
- Recorded model-ready population: 137,742 records after requiring at least 10 April impressions and complete required fields.

The model uses March-only features: `gsc_impressions`, `gsc_clicks`, `observed_ctr`, `avg_position`, and one-hot position-group indicators (`1-3`, `4-10`, `11-20`, `21-50`, `50+`).

April clicks, impressions, and CTR are used only for the outcome and retrospective evaluation. `ctr_gap` and `baseline_score` are excluded from the Random Forest features because they are derived priority fields. Pseudonymous client/content identifiers are used for joins and grouped validation only, never as model features.

Label-derived fields such as `trend_direction` and `trend_pct` are leakage risks if calculated using future-window data. They must not be included as predictors when they depend on the outcome or information unavailable at decision time.

The April-impression threshold is a retrospective selection rule based on future information. It is disclosed as a limitation and must not be reused as a live March eligibility condition.

This report contains aggregate results only. Before publishing, inspect every notebook, output, chart, and artifact under `work/`; this report alone cannot certify that every repository file is free of identifying data.

## 3. Baseline

The baseline is a transparent position-peer priority score. For each March position group, expected CTR is calculated as aggregate clicks divided by aggregate impressions. The score combines a normalized positive gap between peer expected CTR and observed CTR with normalized `log1p(impressions)`:

`baseline_score = 0.7 × gap_score + 0.3 × impression_score`

This is a priority score, not a CTR forecast. It is a fair operational comparator because it uses historical March information and ranks candidates without training a predictive model.

The recorded same-split Spearman results are:

| Method | Spearman rank correlation |
|---|---:|
| Position-peer baseline | -0.250472 |
| Random Forest | 0.343931 |

Both methods were evaluated on the same recorded random-row validation split of 27,549 records.

Recorded mean April CTR among the top-ranked validation records:

| Top K | Baseline | Random Forest |
|---:|---:|---:|
| 20 | 0.0578% | 6.0012% |
| 50 | 0.0610% | 3.7988% |
| 100 | 0.0487% | 2.9967% |
| 500 | 0.1091% | 1.4608% |

These are observed outcomes for selected groups, not causal uplift. CTR is sparse and heavy-tailed, so the top-20 result may be sensitive to a small number of records. Baseline MAE, RMSE, and R² were not available in the recorded same-split comparison, so they are not invented here.

## 4. Model / analysis

The method is a `RandomForestRegressor` with 100 trees, `max_depth=12`, `min_samples_leaf=20`, `random_state=42`, and `n_jobs=-1`.

It fits this lane because it can model nonlinear relationships among historical visibility, clicks, CTR, position, and position group without requiring a linear relationship to be specified in advance.

**Feature list:** March `gsc_impressions`, March `gsc_clicks`, March `observed_ctr`, March `avg_position`, and one-hot position-group indicators.

Client/content IDs, April outcome columns, `ctr_gap`, `baseline_score`, and future-derived trend fields are excluded from model features.

**Target definition:** `future_ctr` is April clicks divided by April impressions, evaluated only for joined records with at least 10 April impressions and complete required fields.

The main experiment uses an 80/20 random row split with `random_state=42`. A separate `GroupShuffleSplit` groups by pseudonymous client so that no client appears in both training and validation.

The recorded client-held-out audit used 29,797 validation rows from nine held-out clients and reported Spearman 0.378473, MAE 0.002357, RMSE 0.006052, and R² 0.123012. This is a separate validation design and must not be treated as the same-split baseline comparison.

## 5. Evaluation

The primary recorded same-split results are:

| Metric | Position-peer baseline | Random Forest |
|---|---:|---:|
| Spearman rank correlation | -0.250472 | 0.343931 |
| MAE | Not recorded | 0.002425 |
| RMSE | Not recorded | 0.006364 |
| R² | Not recorded | 0.140516 |

Spearman correlation is the primary ranking metric because the intended use is prioritization. The Random Forest's positive value indicates that higher predictions tended to be associated with higher observed April CTR in this validation sample. The R² of 0.140516 indicates that much of the variation in future CTR remains unexplained.

CTR is highly skewed, and many observations have low or zero CTR. A small number of records can therefore materially affect averages in small top-K groups. The recorded April CTR distribution had a median of zero and a maximum of 1.0 before any additional display clipping.

This is a regression/ranking task rather than a majority-class classification task, so a majority-class percentage is not applicable. Top-K mean CTR is not accuracy and should not be described as causal lift.

The notebook must be run against the gated dataset to confirm that the current data access, preprocessing, and split reproduce the recorded values. If results differ, investigate the discrepancy and update the paper and report together.

## 6. Interpretation

The recorded same-split result is directionally favorable to the Random Forest for ranking future CTR: its Spearman correlation was positive while the baseline's was negative. The recorded top-K means also favor the model on this validation sample.

The separate client-held-out audit reported a positive Spearman correlation of 0.378473. This suggests that the ranking association persisted on that particular held-out-client split, but one split with nine held-out clients is not enough to establish broad generalization.

Feature-importance values and a formal error breakdown by position group or client were not included in the recorded results used to draft this report. They should not be claimed until computed and checked in a fresh run.

No claim is made that the model predicts Google's algorithm or establishes causal effects.

## 7. Recommendation

Use the ranking as a human-in-the-loop review queue.

1. **Review high-visibility content with CTR below position peers.** Inspect search intent and the actual result presentation before changing a title or snippet.
2. **Review content around positions 11–20 with meaningful March impressions.** Check intent match, completeness, internal links, and position stability.
3. **Investigate model/observation discrepancies.** Treat large differences as trend or data-quality review candidates, not estimates of potential uplift.
4. **Use evidence levels cautiously.** The action workflow treats at least 1,000 impressions as higher observed visibility and 100–299 impressions as more limited evidence. These are workflow heuristics, not confidence intervals.
5. **Require editorial review and document interventions.** Record the hypothesis, change, date, and outcome window. Avoid untracked simultaneous edits.
6. **Monitor after changes.** Use explicit review windows and account for seasonality and query-mix changes.
7. **Audit queue quality and model drift.** Monitor ranking metrics, prediction errors, benchmark coverage, missing values, duplicate keys, score validity, reason-code distribution, and results across position groups and clients.

The recorded action-queue export contained 23,496 rows and 19 columns, with ranks 1–23,496, nonnegative priority scores, and no missing peer benchmark values in the audited export. Operational queue eligibility uses at least 100 March impressions. This differs from the retrospective April-impression filter used for model evaluation.

**Confidence:** moderate that the recorded experiment ranks this validation sample better by Spearman than the heuristic baseline; limited confidence in generalization and no causal confidence. The ranking should guide investigation, not dictate edits.

## 8. Reproducibility

- **Repository:** https://github.com/AliAlmasri7/FlyRank-ML
- **Dataset:** https://huggingface.co/datasets/FlyRank/internship-warehouse
- **Capstone notebook:** `work/notebooks/capstone.ipynb`

From a fresh clone, create a Python environment and install the dependencies used by the notebook:

```bash
git clone https://github.com/AliAlmasri7/FlyRank-ML.git
cd FlyRank-ML

python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install duckdb numpy pandas matplotlib scikit-learn scipy jupyter nbconvert huggingface_hub
```

Set `HF_TOKEN` privately in the shell or notebook environment. Do not put the token in the notebook or repository.

```bash
# macOS/Linux example:
export HF_TOKEN="YOUR_TOKEN"

jupyter nbconvert --to notebook --execute work/notebooks/capstone.ipynb --output capstone_executed.ipynb
```

Recorded random seeds are `random_state=42` for the row split, Random Forest, and client-held-out split. The Random Forest uses 100 estimators, maximum depth 12, minimum leaf size 20, and `n_jobs=-1`. The main split is 80/20; the client-held-out audit uses `GroupShuffleSplit(test_size=0.20, n_splits=1, random_state=42)`.

Supporting notebooks:

- `work/notebooks/w01_research_question.ipynb`
- `work/notebooks/w02_ml_task_framing.ipynb`
- `work/notebooks/w03_data_contract.ipynb`
- `work/notebooks/w03_feature_leakage_check.ipynb`
- `work/notebooks/w04_baseline_score.ipynb`
- `work/notebooks/w05_model.ipynb`
- `work/notebooks/w06_validation_audit.ipynb`
- `work/notebooks/w07_action_playbook.ipynb`

The notebook has not yet been confirmed executing end-to-end against the gated dataset in a fresh environment. Before submission, run it top-to-bottom, record actual package versions (for example, with `python -m pip freeze`), confirm the data query paths, and ensure every metric in this report matches the fresh run. Do not commit credentials or publish raw warehouse data.

---

**Claims checklist:** Findings are described as observed, directional, and decision-support. No causal claim is made. The model does not predict Google's algorithm. Public artifacts must not contain client-identifying details. Metrics must be refreshed from a successful run before final submission.
