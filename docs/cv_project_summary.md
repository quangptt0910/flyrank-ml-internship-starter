# FlyRank ML Internship — Project Summary & CV Guide

## 1. Executive Summary (The "Elevator Pitch" for Interviews)

> Built an enterprise-grade content opportunity and decay ranking engine on 30,000 multi-tenant content items across 32 client domains (Google Search Console & GA4 data). Replaced an existing volume-biased heuristic rule that failed on unseen domains (20% Precision@10) with modern tree ensembles and linear benchmarks, achieving **90.0% Precision@10** and **80.0% Precision@50** on held-out client domains (a **4.5× improvement**). Diagnosed and eliminated a **16.3-point memorization leakage gap** using client-grouped holdout validation, implemented positive-control leakage verification, and extracted interpretable reason codes for human editorial decision-support.

---

## 2. Ready-to-Use CV / Resume Bullet Points

### Option A: Impact & Engineering-Focused (Recommended for ML Engineer / Data Scientist roles)

* **Machine Learning & System Design:** Engineered an end-to-end content decay prioritization engine across 30,000 content items from 32 enterprise domains, framing decision-support ranking as probabilistic binary classification.
* **Model Progression & Benchmarking:** Implemented a rigorous progression from transparent heuristics (W04 baseline) to L2 Logistic Regression, printable depth-3 Decision Trees, Random Forests, and HistGradientBoosting, boosting **Precision@10 from 20.0% to 90.0%** (4.5× lift) and **Precision@50 from 42.0% to 80.0%** on held-out domains.
* **Honest Validation & Leakage Prevention:** Uncovered and eliminated a **16.3-point optimistic memorization gap** (ROC-AUC 0.783 → 0.620) by replacing naive random splits with client-grouped holdout validation (`GroupShuffleSplit`), proving models generalize to novel customer domains rather than memorizing domain authority.
* **Positive-Control Audit Harness:** Architected a programmatic leakage audit with positive-control injection (injecting ground-truth target arithmetic to verify test sensitivity at AUC >0.999), enforcing strict temporal feature-window isolation.
* **Model Interpretability & Error Analysis:** Conducted test-set permutation importance (scoring on Average Precision) and examined individual false positives/negatives, demonstrating that activity consistency (`days_with_impressions`), striking-distance rank (positions 4–20), and staleness predict decay far more accurately than raw traffic scale.

---

### Option B: Concise 3-Bullet Format (Compact Resume Space)

* **Enterprise Ranking Engine:** Developed a content refresh opportunity model on 30k multi-tenant pages across 32 domains, boosting top-10 triage precision from **20.0% to 90.0%** over the production heuristic rule.
* **Rigorous Validation & Leakage Defense:** Designed client-grouped holdout splits to eliminate a **16.3% domain-memorization bias**, and built an automated audit harness with temporal window isolation and positive-control leakage testing.
* **Explainability & Editorial Decision-Support:** Extracted transparent decision rules and permutation importances across 24 features (GSC + GA4), providing interpretable reason codes that decouple decay probability from raw impression volume.

---

## 3. Project Technical Specifications & Keywords

| Category | Details |
|---|---|
| **Domain** | Search Engine Optimization (SEO), Search Performance, Decision-Support Systems |
| **Data Scope** | 30,000 rows × 44 columns, 32 pseudonymized client domains, GSC + GA4 signals |
| **ML Models** | Heuristic Baseline, Logistic Regression (L2), Decision Tree (depth 3), Random Forest (200 trees), HistGradientBoosting |
| **Validation Design** | Client-Grouped Holdout (`GroupShuffleSplit` on `client_id`, 80/20 train/test) |
| **Key Metrics** | Precision@10 (90.0%), Precision@20 (90.0%), Precision@50 (80.0%), Precision@100 (76.0%), ROC-AUC (0.620), PR-AUC (0.618) |
| **Tech Stack** | Python, scikit-learn, pandas, numpy, Jupyter, nbconvert, Git |
| **Methodologies** | Data contracts, leakage auditing, positive-control testing, permutation importance, subgroup analysis, claim ladder |

---

## 4. Interview Talking Points (STAR Method)

### Situation
Enterprise SEO teams managing thousands of URLs face severe operational bottlenecks: they only have the bandwidth to review 20–50 pages per week. The incumbent rule sorted pages by raw impression volume and striking-distance rank, but it failed on unseen domains because it prioritized massive, actively defended cornerstone pages rather than decaying content.

### Task
Frame, train, validate, and interpret an honest machine learning system that accurately prioritizes pages facing genuine decay risk, ensuring the model generalizes to completely new, unseen client websites without data leakage.

### Action
1. **Data Contract & Window Isolation:** Isolated trailing 90-day search/traffic aggregates from recent 30-day trend computation windows, strictly excluding label-derived fields.
2. **Model Progression:** Built a baseline rule, then trained linear and non-linear models with custom handling for missingness-by-content-type and sentinel position values (`avg_position = 0`).
3. **Validation Audit:** Tested models under both random 80/20 and client-grouped 80/20 splits to quantify domain memorization, discovering a 16.3-point AUC inflation in the naive split.
4. **Leakage Attack:** Executed a positive-control test by deliberately injecting `trend_pct` to verify the test harness caught leakage before stripping it.

### Result
Achieved **90.0% Precision@10** and **80.0% Precision@50** on held-out domains (compared to 20.0% and 42.0% for the baseline rule). Permutation feature importance revealed that page activity consistency, position tier, and staleness are the primary predictors of decay, producing an actionable queue with transparent reason codes for human editors.
