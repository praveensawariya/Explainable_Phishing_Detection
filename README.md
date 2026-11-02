# Results archive — Leakage-Aware Evaluation of Explainable Phishing Website Detection Models with Cross-Source Transfer Testing

Authors: Dhrupa Mistry (corresponding author, dhrupamistry512@gmail.com) and Praveen Sawariya.
Repository: https://github.com/praveensawariya/Explainable_Phishing_Detection

This folder is the results archive promised in the paper's Data Availability statement.
Upload its contents to the repository root (alongside the notebook) so that every
result table in the paper can be checked.

## Contents

- `research_results/` — the full result archive produced by the paper's notebook
  (all CSV tables, figures and predictions behind Tables 1–14 and A1–A2), including
  the `revision/` subfolder with the multi-seed, decomposition and within-PhreshPhish
  results. The PhreshPhish samples are inside it:
  - `research_results/revision/phreshphish_test_sample_urls.csv` — the 10,000-URL
    PhreshPhish test sample (4,555 phishing / 5,445 legitimate, sampled with seed 42
    by reservoir sampling over the streamed test split).
  - `research_results/revision/phreshphish_train_sample_urls.csv` — the 20,000-URL PhreshPhish
    training sample (44.7% phishing, zero URL overlap with the test sample).
- `corrected_transfer/` — the corrected cross-source transfer results. IMPORTANT:
  the three transfer files inside `research_results/` (`phreshphish_external_predictions.csv`,
  `phreshphish_external_validation_metrics.csv`, `external_feature_threshold_sensitivity.csv`)
  were produced with an inverted feature coding and are SUPERSEDED by the files here.
  Corrected transfer result (10 seeds): ROC-AUC 0.5738 ± 0.0006, MCC 0.3592 ± 0.0064;
  fixed split (seed 42): accuracy 0.6551, precision 0.9145, recall 0.2678, F1 0.4143,
  ROC-AUC 0.5734, MCC 0.3616.
  - `transfer_rerun_A.csv` — per-seed corrected (docs-faithful) transfer metrics.
  - `transfer_rerun_B.csv` — per-seed conservative-variant sensitivity metrics.
  - `transfer_rerun_orig.csv` — per-seed original (inverted-coding) metrics, kept only
    as the coding-direction check reported in Section 5.8 of the paper.
  - `transfer_rerun_summary.json` — headline numbers for all three variants.
  - `sensitivity_corrected_A.csv` — corrected URL-length-bin sensitivity.
  - `phreshphish_predictions_fixed42_A.csv` — corrected fixed-split transfer predictions.
  - `phreshphish_predictions_fixed42_orig.csv` — original fixed-split transfer predictions.
  - `rerun_transfer.py` — the script that produced all of the above. To run it against
    this archive, set the `BASE` path at the top of the script to this repository's
    `research_results/` folder. Requires scikit-learn 1.6.1 (later versions do not
    reproduce the original inverted-coding numbers exactly) and the `ucimlrepo` package.
- `split_indices/` — group-aware split indices for the UCI Phishing Websites data
  (dataset 327), generated with `StratifiedGroupKFold(n_splits=5, shuffle=True)` on
  feature-vector groups (the group of a row is the hash of its 30 feature values), so
  no feature vector appears on both sides of any split.
  - `split_seed42_fold0.csv` — the paper's primary split (train 8,898 / test 2,157).
  - `split_seeds_fold0.csv` — fold-0 splits for all ten seeds used in the paper
    (42, 7, 11, 23, 101, 202, 303, 404, 505, 606).
  - `split_seed42_allfolds.csv` — all five outer folds for seed 42 (test_fold per row).

## Corrected feature coding (Section 4.5 of the paper)

The transfer extractor maps URL features to the UCI creators' coding
(Legitimate = +1, Suspicious = 0, Phishing = −1): having an IP address, a long URL,
an @ symbol, a prefix/suffix dash, multiple subdomains (leading "www." stripped
before counting dots), a shortening service, a double-slash redirect, or an https
token all map to −1; the absence of each maps to +1; an unparseable URL maps to +1.
The notebook's extractor implements exactly this coding.

## Notebook

The repository notebook (`Explainable_Phishing_Detection.ipynb`) contains the full
pipeline. Its stored transfer outputs were refreshed to the corrected values above;
a full re-run (Restart & Run All) regenerates every file in `research_results/`
except the corrected-transfer files, which are produced by `rerun_transfer.py`.
