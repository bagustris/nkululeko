# Cross-corpus / cross-lingual generalization for audio deepfake detection: a controlled ablation

Study period: 2026-09-20 to 2026-10-06. Backbone under test: `AasistModel`
(end-to-end fine-tuned wav2vec2-base SSL frontend + graph-attention
backend), compared against `DeepfakeADMModel` (ADM: frozen SSL + sptk,
multi-branch fusion) and `MLPModel` (frozen SSL + sptk, plain linear
head) as frozen-feature baselines. All numbers are EER on the held-out
test domain unless stated otherwise.

## 1. Starting point: LODO and cross-lingual numbers

5-fold leave-one-dataset-out (`itw`, `la`, `had`, `cvoicefake`, `habla`;
dev fixed on `for2sec`), model/features per `exp_lodo5_heavy.ini`:

| held-out fold | EER |
|---|---|
| itw | 6.52% |
| la | 3.41% |
| had | **30.96%** |
| cvoicefake | 9.46% |
| habla | 2.60% |
| **mean** | **10.5%** |

`had` is the outlier fold by a wide margin, which is why it anchors most
of the controlled ablation below. `itw` was added later as a second,
independent fold to check whether any pattern found on `had` generalizes.

Cross-lingual:

| setup | EER |
|---|---|
| DECRO corpus-swap, en→ch | 23.01% |
| DECRO corpus-swap, ch→en | 40.16% |
| DECRO corpus-swap, mean | **31.58%** |
| cvoicefake language-column LODO (same corpus, hold out one language) | **8.63%** |

True cross-corpus language transfer (DECRO, two different recording
pipelines) is far harder than within-corpus language holdout
(cvoicefake). Language-transfer difficulty is confounded with
corpus/recording-pipeline-transfer difficulty in this literature, and
the two should be reported separately rather than both being called
"cross-lingual generalization."

## 2. Controlled ablation design

To get an apples-to-apples comparison of generalization techniques
(RawBoost, domain-balanced batch sampling, SAM, DANN), we fixed:

- **Backbone**: AASIST, unfrozen SSL frontend (freezing was tried once —
  see `exp_aasist_had_frozen_last.ini` — and abandoned: 26.8% DEV EER
  after 8 epochs, still slowly improving, ~6-7x worse than unfrozen's
  ~4% DEV EER at the same point, and not worth the compute to run to
  convergence).
- **Data budget**: a stratified 2500/class subsample per pooled training
  domain, 1000/class for the fixed dev domain and the held-out test
  domain (seed=42 construction). Two independent folds were built this
  way: `had` held out (`_aasist_subsample_2500/`) and `itw` held out
  (`_aasist_subsample_2500_itwheld/`).
- **Everything else identical** between a technique's config and its
  bare baseline on the same fold (`patience=5`, `lr=1e-5`, `batch_size=16`,
  `epochs<=50`).

### 2.1 Architecture axis (frozen features vs. end-to-end), `had` fold, single run each

| model | frontend | EER | UAR | specificity(fake) |
|---|---|---|---|---|
| AASIST (end-to-end) | wav2vec2-base (unfrozen) | 7.95%* | 53.6% | 7.3% |
| ADM (frozen) | wav2vec2-base | 36.95% | 64.1% | 52.8% |
| ADM (frozen) | wav2vec2-xls-r-300m | 39.00% | 63.8% | 29.7% |
| MLP (frozen) | wav2vec2-base | 38.85% | 61.7% | 67.1% |

\* this single-run AASIST number is known to be a favorable outlier —
see §3. The architecture-axis conclusion does not depend on it: a
stronger SSL frontend (XLS-R-300M, 1024-dim) did **not** improve ADM's
EER (39.00% vs 36.95%), which rules out "frozen features just need a
better frontend" and points at the frozen-vs-end-to-end training axis
itself, not frontend quality, as the dominant factor. MLP (a plain
2-layer head) and ADM (multi-branch fusion) land in the same range
despite very different architectural complexity, so the bottleneck is
"frozen features," not "ADM's specific architecture."

### 2.2 A calibration finding that complicates every EER number above

At the model's own decision threshold (not the EER-optimal threshold),
every AASIST `had`-fold config — including the "good" 7.95% EER
baseline — predicts "real" for almost every held-out-domain sample:
specificity-for-fake as low as **1.5–12%**, UAR ~50–56%, i.e. a
deployed system using AASIST's native threshold would catch almost no
fakes in this domain. EER looks fine because it is a threshold-free
ROC statistic (scores still rank correctly); accuracy-at-threshold does
not. ADM and MLP do **not** show this collapse (specificity-for-fake
30–72%, UAR 60–66%) despite worse EER. So end-to-end fine-tuning buys
better score separability at the cost of a severe threshold-transfer
failure under domain shift; frozen-feature models transfer thresholds
better but separate classes less well. Any claim built only on EER in
this setting should be read with this trade-off in mind.

## 3. Why single-run and n=3 comparisons were not trustworthy

The original single-run AASIST `had`-fold baseline (7.95% EER) anchored
every technique comparison below it. Once reseeded (3 runs, unseeded
stochastic init — nkululeko does not fix a seed unless `MODEL.random_seed`
is set, in which case all runs become identical and are collapsed to
`runs=1`), that number turned out to be a favorable outlier:

| had-fold bare AASIST | run 1 | run 2 | run 3 | mean | std |
|---|---|---|---|---|---|
| EER | 14.90% | 27.15% | 18.80% | 20.28% | 5.11pp |

The original "baseline" (7.95%) sat *below the minimum* of 3 reseeded
runs. Run-to-run std (5.11pp) is roughly a quarter of the mean — bare
AASIST training on this data is intrinsically unstable. Several
single-run "technique X helps/hurts" conclusions flipped sign once
reseeded (domain-balanced sampling went from "hurts by 7.5pp" at n=1 to
"helps by 8.7pp" at n=3 to a much weaker, non-significant "helps by
4.3pp" at n=5; SAM on ADM/MLP went from "helps" at n=1 to "hurts" at
n=3). **Every number from here on is reported with its seed count and a
significance test, not as a point estimate.**

### 3.1 On the significance test itself

With n=3 per condition, Welch's t-test and an exact permutation test
(all `C(2n,n)` equal-size splits of the pooled samples, no distributional
assumption) can disagree substantially — the t-test's asymptotics do not
hold at n=3, and it can report a nominally significant p-value the exact
test flatly contradicts (seen directly at n=5 for `itw` domain-balanced:
Welch p=0.028, exact p=0.100). The exact test's best-possible p-value is
mechanically floored by sample size: 1/20=0.05 at n=3 vs n=3, 1/252≈0.004
at n=5 vs n=5, 1/12,870≈0.00008 at n=8 vs n=8. **n=3 can essentially
never produce a result a reviewer should believe, regardless of the true
effect size** — this is why several dramatic-looking n=3 deltas shrank
or vanished once more seeds were added. All p-values below are the exact
permutation test; Welch's is shown alongside only to flag disagreement.

## 4. Seeded technique comparison — `had` fold held out

| config | n | mean EER | std | vs. bare: Welch p | vs. bare: exact p |
|---|---|---|---|---|---|
| bare | 5 | 17.30% | 6.04pp | — | — |
| + domain-balanced sampling | 5 | 13.00% | 2.14pp | 0.194 | 0.167 |
| + SAM | 3 | 11.25% | 4.41pp | 0.205 | 0.232 |
| + RawBoost (algo=5) | 3 | 17.72% | 3.86pp | 0.918 | 0.929 |
| + DANN (`source_db` axis) | 8 | 16.75% | 6.61pp | not vs. bare — see §6 | not vs. bare — see §6 |
| + DANN (`language` axis) | 8 | 26.14% | 6.79pp | not vs. bare — see §6 | not vs. bare — see §6 |
| + DANN (`source_db`+`language`, 2-axis) | 3 | 25.75% | 12.57pp | 0.420† | 0.212† |

† vs. the 8-seed `source_db`-only arm, not vs. bare.

A natural-imbalance variant (no 2500/class equalization; single run
each, not reseeded) shows the same direction for domain-balanced
sampling: bare 8.95% → +domain-balanced 14.85%. Directionally
consistent with the balanced-pool result above, but not independently
confirmed since it was never reseeded.

## 5. Seeded technique comparison — `itw` fold held out

Built the same way as `had` (own stratified 2500/class subsample,
`_aasist_subsample_2500_itwheld/`), to check whether any `had`-fold
pattern generalizes to a second, independent held-out domain.

| config | n | mean EER | std | vs. bare: Welch p | vs. bare: exact p |
|---|---|---|---|---|---|
| bare | 5 | 20.38% | 5.97pp | — | — |
| + domain-balanced sampling | 5 | 24.79% | 4.94pp | 0.240 | 0.230 |
| + SAM | 3 | 16.47% | 1.96pp | 0.244 | 0.375 |
| + RawBoost (algo=5) | 3 | 14.65% | 3.62pp | 0.177 | 0.250 |

Domain-balanced sampling numerically helps on `had` (+4.3pp) and hurts
on `itw` (−4.4pp) — opposite directions, both **not significant**.
This is the corrected version of an earlier (n=3) claim that overstated
this as a confirmed fold-dependent sign-flip; at n=5 the effect in both
folds shrank to roughly half its apparent n=3 size. The honest
conclusion is **no significant effect found, in either direction, for
domain-balanced sampling, SAM, or RawBoost on AASIST, on either fold
tested.**

## 6. The one result that reached significance

The DANN axis ablation isolates whether 2-axis DANN's poor single-run
number (31.35%, `source_db`+`language`) came from the `language` axis
specifically or just from stacking two adversarial heads. Pushed to
n=8 per arm (the only cells extended this far, since this was the one
comparison with both a sizeable point estimate and a plausible
mechanism):

| DANN axis | n | mean EER | std |
|---|---|---|---|
| `source_db` only | 8 | 16.75% | 6.61pp |
| `language` only | 8 | 26.14% | 6.79pp |

**Diff −9.39pp. Welch p=0.014. Exact permutation p=0.016.** Welch and
the exact test agree closely here (unlike every other comparison in
this study), which is itself evidence this result is not a small-sample
artifact. Adversarially training AASIST's shared representation against
**language identity** specifically and significantly degrades
cross-corpus generalization relative to adversarially training against
**dataset identity** alone — the 2-axis config's poor performance is
attributable to the language axis, not to running two adversarial heads.

## 7. Other architectures, seeded (ADM, MLP + SAM)

| config | n | mean EER | std |
|---|---|---|---|
| ADM bare | 3 | 35.48% | 1.39pp |
| ADM + SAM | 3 | 37.73% | 0.90pp |
| MLP bare | 3 | 38.67% | 0.42pp |
| MLP + SAM | 3 | 39.45% | 0.91pp |

ADM: Welch p=0.138, exact p=0.200. MLP: Welch p=0.356, exact p=0.400.
Not significant, but both point the same direction as AASIST's SAM
result (no positive effect) once properly seeded — the single-run
numbers that originally suggested SAM helps ADM/MLP (36.95%→34.15%,
38.85%→38.10%) were themselves n=1 noise; seeded means move the other
way for both.

## 8. Headline conclusions

1. **[Significant, n=8 per arm]** Domain-adversarial training against
   language identity specifically — not dataset identity, and not
   "having two adversarial heads" generically — hurts cross-corpus
   deepfake-detection generalization (26.1% vs 16.8% EER, exact
   permutation p=0.016).
2. **[Descriptive]** Frozen-feature classifiers generalize far worse in
   EER than an end-to-end fine-tuned detector regardless of frontend
   strength or architectural complexity, but the end-to-end model's
   better EER comes with a severe decision-threshold miscalibration
   under domain shift that the frozen-feature models do not show
   (§2.2). Any single-metric (EER-only) comparison between these model
   classes is incomplete.
3. **[Null result, properly powered]** RawBoost, SAM, domain-balanced
   sampling, and 2-axis DANN showed no statistically significant effect
   — positive or negative — on AASIST's cross-corpus EER across two
   independent held-out folds, despite large, dramatic single-run deltas
   that originally motivated this whole investigation. The magnitude of
   n=1/n=3 noise in this setting (run-to-run std up to 6pp on a ~17–20pp
   mean) is itself a methodological finding: single-seed cross-corpus
   deepfake-detection comparisons in this regime should not be trusted.
4. **[Cross-lingual]** True cross-corpus language transfer (DECRO,
   31.6% EER) is far harder than within-corpus language holdout
   (cvoicefake, 8.6% EER) — these should not both be reported under the
   banner "cross-lingual generalization" without distinguishing corpus
   transfer from language transfer.

## 9. Known limitations / what would strengthen this further

- Only two of the five LODO folds (`had`, `itw`) were used for the
  controlled ablation; `la`, `cvoicefake`, `habla` were not checked
  against the same technique set.
- RawBoost, SAM, and 2-axis DANN were only reseeded to n=3 — they could
  still be underpowered false negatives, not confirmed true nulls. Any
  of them could be pushed to n=8 the same way the DANN-axis comparison
  was.
- The architecture-axis comparison (§2.1, §2.2) was never reseeded at
  all — it rests on single runs per (model, frontend) cell.
- A real bug in nkululeko (`[EXP] save` vs `[MODEL] save` confusion;
  `MODEL.save=False` crashing `traindevtest=True` runs) was found and
  filed as [issue #67](https://github.com/bagustris/nkululeko/issues/67);
  main has since merged a fix (PR #450) that was not yet pulled into
  this branch at the time of writing — see §10.

## 10. Process notes (for reproducing or extending this study)

- All ablation configs live in `data/multidb_ood/` and
  `data/multidb_ood/seeds3/` (gitignored — local, machine-specific
  absolute paths to `/home/bagus/data/...`; representative copies are
  included under `configs/` alongside this report).
- Checkpoint management: AASIST unconditionally writes a model
  checkpoint per epoch unless `[MODEL] save = False` — but setting that
  crashes `traindevtest=True` runs (see above), so the working pattern
  used throughout was: let checkpoints save normally, delete the run's
  `models/` directory immediately after the job exits successfully
  (report.pkl/logs/results are unaffected). Without this, 3-seed AASIST
  runs accumulate 30-50GB each and will fill a disk fast.
- `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` is worth setting for any
  unattended queue — one run hung for >24h on a stale HuggingFace Hub
  socket.
- `feat/aasist` is the only branch with any of this code (AASIST model,
  DANN, SAM, domain-balanced sampling); an accidental `git checkout
  main` / `git pull origin main` mid-queue silently breaks every job
  with "unknown model type: 'aasist'" rather than a clear error.
