# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Youssef Abozaid
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/youssefabozaidyou/flyrank-ml-internship
- **Date:** 2026-08-02

> Copy this file to `work/capstone_report.md` and fill it in as you build. The eight
> sections mirror the Pass / Needs-Work rubric axes, so nothing here is optional.

## 1. Problem framing

**Unit of analysis:** one content item (page), scored once per month.

**Output:** a ranked list — every content item gets a `decline_risk_score` (0–1) plus a
`reason_code` explaining *why* it scored the way it did (e.g. "impressions falling, position
stable" vs "position falling, impressions still up").

**The action a human takes:** a FlyRank content editor works down the ranked list each week,
starting from the top, and decides whether to refresh, merge, or leave each page.

**Cost of a wrong call:**
- **False positive** (flagged as declining, but it wasn't really) → wasted editor hours on a
  page that didn't need it.
- **False negative** (a genuinely declining page not flagged) → a real opportunity is missed
  and the page keeps losing visibility for another month before anyone notices.

**Why ML helps here:** a simple "flag anything whose impressions dropped" rule ignores *how
much* history a page has, whether the drop is noise or a trend, and how position and traffic
move together. Those signals interact in ways that are hard to write as a single if-statement —
that's where a model earns its place over the baseline rule.

## 2. Data safety

**Data used:**
- `fact_content_daily_performance` — the daily time series that features and the label are
  built from (report_date × client × content grain), for a mid-panel month (`month = '2026-03'`)
- `dim_clients` — used only to build the client-holdout split, never as a feature

**Deliberately excluded columns (and why):**
- `trend_direction`, `trend_pct` — label-derived. These are literally how the label is defined,
  so using them as features would be leakage. This was proven experimentally in
  `w03_data_contract.ipynb`: accuracy jumped from 0.796 to 1.0 the moment a label-derived
  column was added, then dropped back down once it was removed.
- `client_hash_id`, `content_hash_id` — pseudonymous identifiers. Used only for grouping and
  the train/test split, never as model inputs (memorizing an ID doesn't generalize to new
  pages).
- Any column from **inside the label window** (the later half of the analysis month) —
  only columns from **before** the decision point are used as features, so the label can't
  leak into the inputs.

**Leakage risks considered:** the past/future split is enforced by construction — features are
built exclusively from the "prior" half of the month and the label from the "later" half, with
no column crossing that boundary. The client-holdout split additionally prevents a page from
one client "teaching" the model about a page from that same client if it ends up in the test
set.

**Confirmed:** no client names, domains, URLs, or raw exports appear anywhere in `work/` —
only pseudonymous hash IDs used for grouping.

## 3. Baseline

**The transparent rule:** rank pages by the percentage drop in impressions between the second
half of the prior window and the first half of the prior window — the same logic the
warehouse's own `trend_pct` column uses, but computed by hand from raw daily rows rather than
read off the pre-built column, so it stays an independent, fair baseline rather than a
shortcut through the label itself.

**Why it's a fair comparison:** it's built from the exact same raw daily-performance numbers
the model sees, over the same time window, and produces a rank — so it can be compared to the
model on identical terms (Precision@K, on the same holdout pages).

**Its numbers, on the same split and metric as the model:**

| Method | Precision@50 |
|---|---|
| Baseline rule | **0.680** |

(Full comparison table in Section 5.)

## 4. Model / analysis

**Method:** Logistic Regression as the transparent first model, then Random Forest as a
non-linear comparison model. This fits the lane because the decision (refresh or not) is
binary, and editors benefit from being able to see *why* a page scored high — Logistic
Regression coefficients give that directly; Random Forest feature importances give a second,
non-linear read on the same question.

**Feature list (all knowable before the label window):**
1. `impr_prior_avg` — average daily impressions, prior half — historical only
2. `position_prior_avg` — average search position, prior half — historical only
3. `sessions_prior_avg` — average GA4 sessions, prior half — historical only
4. `n_days_present_prior` — days the page had any data in the prior half — historical only
5. `clicks_prior_avg` — average daily clicks, prior half — historical only

**Deliberately left out:** `trend_direction`, `trend_pct` (label-derived — see Section 2), and
any column from the later half of the month (the label window).

**Target/proxy, in one sentence:** `is_declining` = 1 if a page's total impressions in the
later half of the analysis month fell versus the prior half, else 0 — an observed outcome, not
a human-defined rule.

## 5. Evaluation

**Split:** client-holdout, grouped by `client_hash_id` (`GroupShuffleSplit`, `test_size=0.25`,
`random_state=42`) — not a random row split, and not time-aware within a client, because the
risk here is a page from the same client appearing in both train and test and leaking
client-specific patterns. Verified: 39 train clients / 13 test clients, 260,873 train pages /
59,051 test pages, **0 client overlap** between train and test.

**Base rate (majority class, "not declining"): 0.224** — reported next to every score below so
a high number can't be mistaken for an artifact of an easy base rate.

**Model vs baseline, on the same split:**

| Method | Precision@50 | ROC-AUC |
|---|---|---|
| Baseline rule | 0.680 | — |
| Logistic Regression | **0.700** | 0.818 |
| Random Forest | 0.480 | **0.834** |

**What the errors look like:** the model's most confident mistakes are concentrated in pages
with a low `n_days_present_prior` (thin prior history). With only a handful of days to average
over, a single unusually good or bad day swings the feature values more than it should,
inflating the model's confidence beyond what the underlying signal supports. This is why
`low_confidence_thin_history` pages are flagged separately in the recommendation output
(Section 7) rather than trusted at face value.

**A genuine, honest finding from this comparison:** Random Forest has the higher overall
ROC-AUC (0.834 vs 0.818), but scores *below the baseline* on Precision@50 (0.480 vs 0.680),
while Logistic Regression scores above it (0.700). `class_weight="balanced"` improves Random
Forest's overall discrimination but doesn't concentrate its confidence correctly at the very
top of the ranking — exactly where Precision@K is measured. Since the deployed use case is a
short ranked queue an editor works top-down, Precision@K — not AUC — is the metric that should
decide the model. **Logistic Regression is therefore the model used for deployment.**

## 6. Interpretation

**Random Forest feature importances:**

| Feature | Importance |
|---|---|
| `impr_prior_avg` | 0.526 |
| `position_prior_avg` | 0.408 |
| `clicks_prior_avg` | 0.042 |
| `n_days_present_prior` | 0.017 |
| `sessions_prior_avg` | 0.008 |

**Logistic Regression coefficients:**

| Feature | Coefficient |
|---|---|
| `impr_prior_avg` | +1.039 |
| `position_prior_avg` | +0.712 |
| `n_days_present_prior` | +0.439 |
| `sessions_prior_avg` | +0.233 |
| `clicks_prior_avg` | −0.404 |

**In plain words:**
- `impr_prior_avg` and `position_prior_avg` dominate both models, together accounting for
  ~93% of the Random Forest's importance. That matches intuition: a page's recent impression
  volume and recent search position are the most directly observable signals of where it's
  heading next.
- **Surprise / negative result #1:** `clicks_prior_avg` has a *negative* coefficient in
  Logistic Regression — pages with more prior clicks were, all else equal, slightly *more*
  likely to be flagged as declining. A plausible read is that high-CTR pages have less room
  left to grow, so a flat or slightly-down impression trend registers more easily as
  "declining" for them than for a low-CTR page still gaining raw impressions. This is a
  directional read, not a proven mechanism, and is stated here as an honest unexpected result
  rather than smoothed over.
- **Surprise / negative result #2:** `sessions_prior_avg` — the one GA4-based feature — carries
  the least weight in both models. GSC-derived signals (impressions, position) did essentially
  all of the work for this label and window; engagement data added little on top. A
  well-understood "no effect" like this is treated as a valid, useful finding, not a
  shortcoming of the pipeline.

## 7. Recommendation

**The ranked action, tomorrow:** export the top-50 test-set pages by Logistic Regression score
into a weekly queue (`content_id`, `decline_risk_score`, `reason_code`) that a FlyRank editor
opens every Monday and works down top to bottom.

**Reason codes:**
- *"Position problem, not a demand problem"* — `position_prior_avg` is weak but
  `impr_prior_avg` is still reasonable → rewrite / on-page optimization.
- *"Visibility/demand problem"* — `impr_prior_avg` itself is low/falling with position stable →
  consider a merge or a deeper content review.
- *"Review manually — mixed signal"* — neither pattern is clear-cut → needs a human look before
  action.

**Confidence and limits, stated explicitly:** these are **decision-support, directional**
priorities based on one month of history — not a causal claim that refreshing a page will
improve it, and not a claim about Google's ranking algorithm. Confidence is lower for pages
with thin prior history (`n_days_present_prior` low, see Section 5 error analysis); those
pages are flagged separately (`low_confidence_thin_history`) rather than trusted at face value.

## 8. Reproducibility

**To re-run from a fresh clone:**
```bash
git clone https://github.com/youssefabozaidyou/flyrank-ml-internship
cd flyrank-ml-internship
pip install -r requirements.txt
```
Then open `work/notebooks/capstone.ipynb` in Colab (badge at the top of the notebook), set the
`HF_TOKEN` Colab Secret, and Runtime → Run all.

**Random seeds:** `random_state=42` everywhere a split or a model is fit
(`GroupShuffleSplit`, `LogisticRegression`, `RandomForestClassifier`).

**Environment (from the run that produced these numbers):** `duckdb==1.3.2`,
`datasets==4.0.0`, `pyarrow==18.1.0`, `scikit-learn==1.6.1`, `pandas==2.2.2`, `numpy==2.0.2`.

---

> **Claims checklist before submitting:** observed / measured / directional / decision-support
> language everywhere · no causal claims without an experiment or causal design · no
> "predicted Google's algorithm" · no client-identifying details · numbers in this report
> match a fresh re-run.
>
> - [x] Base rate (0.224) reported next to every Precision@K number (Section 5)
> - [x] AUC / lift over baseline reported alongside Precision@K, not in place of it (Section 5)
> - [x] All language is observed / measured / directional / decision-support (Sections 1, 6, 7)
> - [x] No causal claims anywhere in this report
> - [x] No "predicted Google's algorithm" language anywhere
> - [x] No client-identifying details anywhere in this report or in `work/`
> - [x] Numbers in this report match a fresh re-run of `work/notebooks/capstone.ipynb`
