# Capstone Report — Lane 2: Refresh / Content Opportunity Scoring

- **Author:** Quang Pham
- **Lane:** 2 — Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/quangptt0910/flyrank-ml-internship-starter
- **Paper:** https://quangptt0910.github.io/flyrank-ml-internship-starter/
- **Date:** 2026-11-15

Every number below comes from a receipt in `work/outputs/` written by `work/notebooks/capstone.ipynb`
(unless another notebook is named), and `tests/test_capstone_report.py` checks the key ones against those receipts.

## 1. Problem framing

**Decision supported:** which pages an editor should review first in the weekly refresh slot (typically 10–50 pages).

- **Unit:** one page (one row of the starter CSV).
- **Output:** a probability that the page is in decline, P(decline), blended into a 0–100 priority. Each page also gets
  reason codes and one suggested action (Week 7 playbook).
- **Human action:** the editor opens the page and decides whether to refresh it, and how. Nothing changes automatically.
- **Cost of a wrong call:** a false positive wastes 1.5–6 hours of editor time on a healthy page. A false negative leaves a
  declining page unattended for another cycle. Neither is catastrophic, so this is decision support, not automation.
- **Why ML helps:** the obvious rule (look at the biggest pages in striking distance) puts healthy pages at the top
  (Section 3). Many weak signals combined, and checked on unseen clients, do better than any single one (ML-06).

## 2. Data safety

- **Data:** the anonymized starter slice `data/raw/content_refresh_anonymized.csv`: 30,000 pages, 32 clients, trailing
  90-day GSC + GA4 metrics. The ~79M-row warehouse was **not** used for modelling (see Section 7, limits).
- **Label:** `trend_direction == 'down'` (last-30-day impressions more than 20% below the previous 30 days). Base rate 0.542.
- **Excluded on purpose:**
  - label sources `trend_direction` and `trend_pct` (`trend_pct` alone scores AUC 0.955; ML-05)
  - every last-30 / prev-30 column (with them the model reaches AUC 0.998, the positive control in ML-05)
  - `client_id` and `content_id`: grouping only, never features
  - tooling metadata
- **Leakage I could not remove:** the 90-day features overlap the label's 60-day window. I report a **strict** model as
  well, which uses only days 61–90 (fully before the label window) plus static fields.
- **Public safety:** no client names, URLs, titles or queries appear in `work/`. IDs are not printed in any notebook.
  Held-out clients are shown as Client A–G, ordered by size. The CSV is not committed beyond the starter file, and CI
  blocks datasets.

## 3. Baseline

**Week 4 rule:** `[4 ≤ position ≤ 20] × [days_with_impressions ≥ 45] × [impressions_90d ≥ 300] × impressions_90d`.
It means "established pages in striking distance, biggest first", the queue a busy editor would build by hand.

It is a fair comparison because it is scored on the same held-out pages, with the same metrics, as the model.

| Held-out clients, 20 client-grouped splits (mean) | P@10 | P@50 | ROC-AUC |
|---|---|---|---|
| Week 4 rule | 0.40 | 0.47 | 0.54 |
| Test base rate (random pick) | 0.49 | 0.49 | 0.50 |

The rule does **worse than picking at random** at the top. ML-06 explains why: among striking-distance pages, the
biggest ones are the least likely to be declining, so ordering by impressions fills the top with healthy pages.

## 4. Model / analysis

**Method:** `HistGradientBoostingClassifier(max_depth=5, min_samples_leaf=25, class_weight="balanced", random_state=42)`.
Numeric features are scaled and categorical ones one-hot encoded. I chose gradient boosting on the seed-42 split
against logistic regression, a depth-3 tree and a random forest (W05). It handles mixed, skewed tabular features without
much tuning.

**Target (one sentence):** a page is positive if its impressions in the last 30 days fell more than 20% versus the
previous 30 days.

**Features (24):**
- numeric (18):
  - `log_impressions_90d`, `log_clicks_90d`, `log_sessions_90d`
  - `ctr`, `days_with_impressions`, `days_with_sessions`, `avg_position`, `has_position_data`
  - `search_volume`, `cpc`, `competition`
  - `content_age_days`, `days_since_last_update`, `word_count`, `has_word_count`
  - `engagement_rate`, `scroll_rate`, `ai_traffic_pct`
- categorical (6): `content_type`, `main_intent`, `competition_level`, `freshness_tier`, `position_tier`, `word_count_tier`

**Strict variant (10 + 5):**
- traffic only from days 61–90: `log_{impressions,clicks,sessions}_d61_90`
- `search_volume`, `cpc`, `competition`, `content_age_days`, `days_since_last_update`, `word_count`, `has_word_count`
- `content_type`, `main_intent`, `competition_level`, `freshness_tier`, `word_count_tier`

**Priority:** `100 × (0.60·P(decline) + 0.25·log-impressions share + 0.15·position leverage)`. The weights are a
judgment call, not fitted.

## 5. Evaluation

**Split:** `GroupShuffleSplit(test_size=0.2)` on `client_id`, repeated 20 times (seed 42), so each test set holds 7
unseen clients. A random row split would let the model learn each client's patterns: it scores AUC 0.78 vs 0.70
grouped. Grouping matches the real use, which is scoring a new client's catalog.

| Held-out clients, 20 splits (mean) | P@10 | P@50 | ROC-AUC | Beats rule on P@50 |
|---|---|---|---|---|
| Model, 90-day features | 0.89 | 0.81 | 0.70 | 20 / 20 splits |
| Model, strict features | 0.72 | 0.68 | 0.63 | 17 / 20 splits |
| Logistic regression | 0.82 | 0.78 | 0.67 | — |
| Week 4 rule | 0.40 | 0.47 | 0.54 | — |

- The test base rate is 0.49 on average, with a range of 0.25–0.61, so P@K must be read against it. The model's P@50
  is above its split's base rate in 20 of 20 splits.
- Strict P@10 swings from 0.40 to 1.00 across splits. With 7 clients per test set, one split proves little.
- **Queue as shipped:** on the seed-42 held-out clients the blended priority queue reaches P@50 0.66 (traffic weight
  trades some precision for impact), and **16 of the top 20 are really declining**.

**Error analysis** (seed-42 held-out clients, `work/outputs/error_analysis.json`):
- **Wrong picks are healthy pages, not near-misses.** Of the 24 wrong picks in the model's top 100, 13 were stable
  and 10 growing.
- **Skill depends on position.** Within positions 1–3 the AUC is 0.82, at 4–10 it is 0.65, and past position 20 it is
  about coin-flip (0.53, then 0.50). Part of the headline comes from separating groups of pages, not from ranking within
  a group.
- **Skill depends on client.** Per-client AUC is 0.56–0.71. One client (Client B) takes 16 of the 20 global top slots.
- **Label noise.** 20.6% of declining pages had under 10 impressions in the previous 30 days. Without them the AUC is
  0.597 instead of 0.620, so on pages with real traffic the model is a little weaker than the headline.
- **W05 case studies** show the same thing. The worst false positive was a visible but unclicked page (CTR 0%, zero
  engaged sessions): a risk signal, not a guarantee. The worst false negative was a dormant 1-impression page labelled
  "down", which is label noise.

## 6. Interpretation

- **What drives the score** (W05 permutation importance, average precision on the test set): `days_with_impressions`
  (0.070) ≫ `avg_position` (0.034) > `content_age_days` (0.021) > `scroll_rate` (0.016) > `ctr` (0.013). Raw traffic
  volume matters little (≤ 0.004). The model reads *consistency* of visibility and *where* a page ranks, not how big it is.
- **Part of the gain is the overlapping window.** The strict model loses about a third of the lift (AUC 0.63 vs 0.70)
  but still beats the rule on AUC in 20 of 20 splits. Treat 0.70 as an upper bound and 0.63 as the cautious estimate.
- **Negative results (ML-06):** no popular single belief held cleanly client by client. Staleness came closest, but
  only within keyword articles. Thin content, old content and striking distance were reversed or mostly client
  differences. That is a valid finding: single-signal rules do not transfer.
- **Surprises:**
  - The shipped `position_tier` puts 1,205 no-data pages in `top_3`.
  - The by-hand top 20 is almost all one action (`snippet_and_ctr_refresh`).
  - Hits and misses carry the same reason codes, so the codes describe pages but do not explain the model.

## 7. Recommendation

How an editor would use it tomorrow:
1. **Build the queue per client** (or per editor team), not one global list. Otherwise one client gets almost all the work.
2. **Work from the top of the queue, not from the biggest pages.** Measured: top-10 precision 0.72–0.89 vs about 0.40
   for the impressions-ordered rule.
3. **Do cheap snippet / CTR fixes first** on striking-distance pages, and leave strong top-10 pages alone unless they
   show their own warning signs.
4. **Review every page by hand.** The tool says where to look, not what to change.
5. **Log every refresh** and compare it with similar unrefreshed pages over 30–60 days. That is the only route to
   evidence that refreshing helps.

**Confidence:** moderate for "the model ranks declining pages above the rule on unseen clients" (it does on 20 of 20
splits, 17 of 20 strict on P@50). Low for anything about the future or about cause.

**Limits, stated explicitly:**
- It uses one snapshot and was not tested forward in time.
- The 90-day features overlap the label.
- Only 32 clients were available, 7 per test set.
- Catalog-wide W07 scores include training clients, so they are in-sample.
- The model family was chosen on the seed-42 split.
- The label is noisy for small pages.
- It was not compared with FlyRank production flags (none are in the data).
- A warehouse rebuild with forward-in-time labels (features from month M, decline in month M+1) is the natural next step.

## 8. Reproducibility

From a fresh clone (the starter CSV ships with the repo):

```bash
pip install -r requirements.txt
cd work/notebooks
for nb in w03_feature_leakage_check w04_baseline_score w04_signal_audit w05_model w06_validation_audit w07_action_playbook capstone; do
  python -m nbconvert --to notebook --execute --inplace "$nb.ipynb"
done
cd ../.. && pytest tests
```

- **Seeds:**
  - every model uses `random_state=42`
  - the seed-42 split is `GroupShuffleSplit(test_size=0.2, random_state=42)`
  - the repeat uses `GroupShuffleSplit(n_splits=20, test_size=0.2, random_state=42)`, and its split 0 is the seed-42 split
- **Environment used:** Python 3.13, scikit-learn 1.7.1, pandas 2.3.2, numpy 2.4.3, matplotlib 3.10.5.
  `requirements.txt` pins only lower bounds. Small numeric differences are possible on other scikit-learn versions.
- **Receipts:**
  - `work/outputs/robustness_summary.json` (20 splits)
  - `model_comparison.json` (seed-42)
  - `leakage_check_summary.json` (ML-05)
  - `action_playbook_summary.json` (W07)
  - `error_analysis.json`
  - The model card is `work/model_card.md`.
- Rerunning the notebooks reproduces the receipts byte for byte on this environment.

---

**Claims checklist:**
- [x] observed / measured / directional / decision-support wording
- [x] base rate next to every P@K (0.49 mean, 0.25–0.61)
- [x] AUC and the rule comparison reported as the discrimination numbers
- [x] no causal claims
- [x] nothing about predicting Google's algorithm
- [x] no client-identifying details
- [x] numbers match a fresh rerun (checked by `tests/test_capstone_report.py`)
