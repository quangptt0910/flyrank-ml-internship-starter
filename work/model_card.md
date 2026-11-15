# Model card — Refresh priority model (Lane 2)

*Quang Pham, FlyRank ML Internship, November 2026. Built on the FlyRank ML Internship dataset (https://flyrank.ai).*

## What it is
A gradient-boosted classifier (`HistGradientBoostingClassifier`, depth 5, min leaf 25, balanced weights) that scores each page's
probability of being in decline, P(decline). A fixed formula blends that probability with traffic and position into a 0-100 priority.
Each page also gets reason codes and one suggested action (Week 7 playbook).

## Intended use
- **Who:** an editor or SEO strategist planning the weekly review of 10-50 pages.
- **What for:** deciding *which pages to look at first*. A human reviews every page before anything changes.

## Not for
- Publishing, rewriting, redirecting or deleting pages automatically.
- Claiming that a refresh will bring traffic back (nothing causal was tested).
- Explaining or predicting how Google ranks pages.
- Scoring clients or content types very unlike the 32 clients it was built on, without re-validating first.

## Data
- Starter slice of the FlyRank ML Internship dataset: 30,000 anonymized pages, 32 clients, trailing 90-day GSC + GA4 metrics.
- **Label:** last-30-day impressions more than 20% below the previous 30 days (`trend_direction == 'down'`). Base rate 54.2%.
- **Features:** 24 (18 numeric, 6 categorical). Excluded: label sources, every 30-day window column, IDs, tooling metadata.
  No FlyRank product flags exist in the data.

## How well it works (held-out clients, 20 client-grouped splits, mean)
| | Top-10 precision | Top-50 precision | ROC-AUC |
|---|---|---|---|
| Model, 90-day features | 0.89 | 0.81 | 0.70 |
| Model, strict features (before label window) | 0.72 | 0.68 | 0.63 |
| Week 4 rule (impressions-ordered) | 0.40 | 0.47 | 0.54 |

Test base rate 0.49 on average (range 0.25-0.61).
Treat the 90-day row as an upper bound and the strict row as a more cautious estimate.
The blended priority queue reaches top-50 precision 0.66 on the seed-42 held-out clients,
trading some precision for traffic at stake. Top-20 review on those clients: 16/20 declining.

## Where it breaks
- **Overlapping windows.** The 90-day features include the label's last 30 days, so part of the measured skill comes from that overlap (see the strict row).
- **One snapshot.** It has not been tested forward in time. Pages that *look like* recent decliners are not guaranteed to decline next month.
- **Few clients.** Each test set holds 7 clients, and top-10 precision swings from 0.80 to 1.00 across splits (0.40-1.00 strict).
- **Client mix.** Popular single signals (staleness, depth, age, striking distance) did not hold consistently client by client (ML-06).
  The model can inherit client-level patterns that will not transfer.
- **Label blind spots.** Pages labelled `new` or `flat` can never be "declining", and small pages cross ±20% from ordinary noise.
- **In-sample catalog scores.** The Week 7 catalog-wide queue scores pages from training clients with a model that saw them.
  Only held-out numbers are evaluation.
- **Model selection.** The model family was chosen on the seed-42 split. The 20-split repeat reduces but does not remove that optimism.

## Keep it honest in use
- Each week, log whether the reviewed top 20 were really declining. If that share falls below the client's base rate for a month, stop and re-validate.
- Retrain and re-check on held-out clients when new clients are onboarded, or at least quarterly.
- Log every refresh and compare it with similar unrefreshed pages over 30-60 days. That is the only route to evidence of impact.

## Reproduce
`work/notebooks/capstone.ipynb`, seed 42. Receipts: `work/outputs/*.json`. Leakage checks: `work/notebooks/w03_feature_leakage_check.ipynb`
(positive control AUC 0.998).
