# FlyRank ML Internship — Project Summary & CV Guide

All numbers are from held-out clients and come from receipts in `work/outputs/`. Unless marked, they are means over
20 client-grouped splits. Quote them as written; don't round them up.

## 1. Elevator Pitch

> I built a page-refresh ranking for SEO editors on 30,000 anonymized pages from 32 clients (Search Console + GA4).
> On clients the model never saw, its top-10 list held **72–89% declining pages**, against **40%** for an
> impressions-ordered rule (test base rate 49%). The range runs from a strict, leakage-cautious feature set up to the
> full 90-day features. I showed that random splits inflate AUC by about **0.075** (0.78 vs 0.70) compared with
> client-grouped splits. I ran positive-control leakage checks, and the ranking is shipped with reason codes and a
> human-review workflow.

---

## 2. CV Bullet Points

### Option A: Detailed

* **Ranking system:** framed weekly refresh prioritization as probability ranking over 30,000 pages from 32 clients.
  The model's scores are blended with traffic and position into a 0–100 review queue with reason codes and suggested actions.
* **Model vs baseline:** compared a transparent rule, logistic regression, a decision tree, a random forest and
  gradient boosting. Over 20 client-grouped splits, gradient boosting raised top-10 precision from **0.40 to 0.89**
  (**0.72** with strict pre-label features) and ROC-AUC from 0.54 to 0.70 (0.63 strict).
* **Honest validation:** measured a **+0.075 AUC** inflation from random vs client-grouped splits, averaged over many
  splits. An early single-split estimate of 0.163 was corrected after the repeat.
* **Leakage auditing:** ran positive controls (AUC 0.998 when the label's 30-day windows are added back). Found and
  quantified an overlapping-window leak, then reported a strict model that keeps about two thirds of the lift.
* **Error analysis:** found that wrong picks are healthy pages rather than borderline ones, that ranking skill
  concentrates near the top of Google, and that one client dominated a global queue. That led to the recommendation
  of per-client queues.

### Option B: Three Bullets

* **Refresh ranking:** built a page-decline ranking on 30k pages / 32 clients. Top-10 precision was 0.72–0.89 on unseen
  clients vs 0.40 for an impressions-ordered rule (base rate 0.49).
* **Validation & leakage:** used client-grouped repeated splits (random splits inflate AUC by about 0.075), positive-control
  leakage tests, and a strict pre-label-window feature set as a cautious estimate.
* **Decision support:** shipped a 0–100 priority with reason codes and actions, a by-hand top-20 review (16/20
  declining), a model card, and a public write-up with stated limits.

---

## 3. Technical Specifications

| Category | Details |
|---|---|
| **Domain** | SEO content refresh, decision support |
| **Data scope** | 30,000 rows × 44 columns, 32 pseudonymized clients, GSC + GA4 (starter slice, not the 79M-row warehouse) |
| **Models** | Rule baseline, logistic regression, decision tree (depth 3), random forest, HistGradientBoosting |
| **Validation** | `GroupShuffleSplit` on `client_id`, 80/20, repeated 20 times (seed 42) |
| **Key metrics (20-split mean)** | P@10 0.89 / P@50 0.81 / ROC-AUC 0.70; strict 0.72 / 0.68 / 0.63; rule 0.40 / 0.47 / 0.54 |
| **Tech stack** | Python, scikit-learn, pandas, numpy, matplotlib, Jupyter, nbconvert, pytest, Git |
| **Methods** | Data contract, leakage audit with positive controls, permutation importance, per-client error analysis, claim ladder |

---

## 4. Interview Talking Points (STAR)

### Situation
SEO editors can review about 10–50 pages a week. A natural rule (striking-distance pages, biggest first) mostly
surfaces healthy pages.

### Task
Rank pages by likelihood of decline in a way that holds up on clients the model has never seen, without leakage, and
make the output usable by an editor.

### Action
1. Wrote a data contract that excludes label-derived and 30-day-window columns, and tested it with positive controls.
2. Audited popular single signals client by client. None held cleanly, which motivated a multi-signal model.
3. Built the rule baseline, then compared four model families on held-out clients.
4. Repeated the client-grouped split 20 times and added a strict feature set to bound the overlapping-window leak.
5. Turned probabilities into a review queue with reason codes and actions, and checked the top 20 by hand.

### Result
On unseen clients, top-10 precision was 0.72–0.89 vs 0.40 for the rule, and AUC 0.63–0.70 vs 0.54. These are
observational, single-snapshot results, framed as decision support. There is no causal claim. The next step is
forward-in-time validation on the warehouse.
