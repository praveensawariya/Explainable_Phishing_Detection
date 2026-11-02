# Phishing detection: leakage-aware evaluation with cross-source transfer testing

Pipeline (run `Explainable_Phishing_Detection_Q1_Enhanced.ipynb` top to bottom, Colab CPU is sufficient):
1. Sections 1-23: UCI data (repository ID 327), duplicate/conflict audit, group-aware split (StratifiedGroupKFold, 5 folds, fold 0 = test), baseline models, SHAP, bootstrap, McNemar, calibration, timing.
2. Section 24: randomized search (8 candidates; ranges in research_results/revision/hyperparameter_search_spaces.csv).
3. Section 25: PhreshPhish URL-only transfer test (seeded reservoir sample, seed 42; the sample is in research_results/revision/phreshphish_test_sample_urls.csv).
4. Sections 26-33: SHAP stability, cluster bootstrap, conflicting-label sensitivity.
5. Sections 34-43 (revision): multi-seed runs, eight-feature UCI baseline, outer-fold sensitivity, conflict exclusion, group-aware paired tests, SHAP stability baselines, cost-sensitive thresholds, URL baselines, ablations.

Outputs: research_results/*.csv and research_results/revision/*.csv|png|json.
Environment: see research_results/experiment_metadata.json and revision_metadata.json.
