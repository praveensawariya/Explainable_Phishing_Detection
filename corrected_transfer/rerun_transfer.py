# BUNDLE NOTE (repository copy): to run this script against the bundled archive,
# set BASE below to this repository's research_results/ folder. Requires
# scikit-learn 1.6.1 (later versions do not reproduce the original
# inverted-coding numbers exactly) and the ucimlrepo package.
"""Corrected transfer re-run for the phishing paper (peer review M1).

Protocol (identical to the notebook, cells 12/50/52/70):
- UCI 327 via ucimlrepo; y = 1 iff raw label == -1; groups = hash_pandas_object.
- Per seed: StratifiedGroupKFold(5, shuffle, random_state=seed), fold 0 = test.
- 8-feature ET: Pipeline(SimpleImputer(median),
  ExtraTrees(n_estimators=250, max_features='sqrt', class_weight='balanced',
             random_state=seed)).
- PhreshPhish test sample: revision/phreshphish_test_sample_urls.csv (saved sample).

Extractors:
- ORIG: verbatim reconstruction of notebook cell 50 (1 = condition present).
- A (docs-faithful): all 8 flipped to UCI creators' coding (-1 = phishing
  condition present); subdomain strips a leading 'www.' before counting dots;
  parse-fallback = +1 (condition absent, mirroring ORIG's -1 = absent).
- B (conservative): flip only the 5 features where docs AND the UCI file's
  empirical pattern agree (IP, length, @, prefix, subdomain); keep ORIG
  direction for shortening, double-slash, https_token.

Validation first: ORIG must reproduce the published transfer numbers.
Then A (primary) and B (sensitivity) are reported.
"""
import json, re, ipaddress
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from pandas.util import hash_pandas_object
from ucimlrepo import fetch_ucirepo
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             matthews_corrcoef, confusion_matrix)

BASE = "/home/hatch/workspace/phishing_audit_full/revised_results"
OUT = "/home/hatch/workspace/phishing_rerun/results"

SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
              "buff.ly", "adf.ly", "bit.do", "cutt.ly", "rebrand.ly",
              "tiny.cc", "lnkd.in"}
URL8 = ["having_ip_address", "url_length", "shortining_service", "having_at_symbol",
        "double_slash_redirecting", "prefix_suffix", "having_sub_domain", "https_token"]
SEEDS = [42, 7, 11, 23, 101, 202, 303, 404, 505, 606]

def extract(url, mode):
    """mode: 'orig', 'A', 'B'. Returns dict of 8 features (+fallback flag)."""
    raw = str(url).strip()
    candidate = raw if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw) else "http://" + raw
    parsed = urlparse(candidate)
    host = (parsed.hostname or "").lower().strip(".")
    try:
        ipaddress.ip_address(host.strip("[]")); has_ip_o = 1
    except Exception:
        has_ip_o = -1
    length = len(raw)
    len_o = -1 if length < 54 else (0 if length <= 75 else 1)
    fallback = False
    try:
        domain_parts = host.split(".") if host else []
        shortening_o = 1 if host in SHORTENERS else -1
        at_o = 1 if "@" in raw else -1
        after_scheme = raw.split("://", 1)[-1]
        ds_o = 1 if "//" in after_scheme else -1
        prefix_o = 1 if "-" in host else -1
        dot_count = max(0, len(domain_parts) - 1)
        sub_o = -1 if dot_count <= 1 else (0 if dot_count == 2 else 1)
        https_o = 1 if "https" in host else -1
    except Exception:
        fallback = True
        shortening_o = at_o = ds_o = prefix_o = sub_o = https_o = -1
    if mode == "orig":
        return {"having_ip_address": has_ip_o, "url_length": len_o,
                "shortining_service": shortening_o, "having_at_symbol": at_o,
                "double_slash_redirecting": ds_o, "prefix_suffix": prefix_o,
                "having_sub_domain": sub_o, "https_token": https_o}, fallback
    # Corrected directions: -1 = phishing condition present.
    has_ip = -has_ip_o
    len_c = -len_o if len_o != 0 else 0
    # subdomain with www stripped (creators' convention)
    h = host[4:] if host.startswith("www.") else host
    parts = h.split(".") if h else []
    dc = max(0, len(parts) - 1)
    sub_c = 1 if dc <= 1 else (0 if dc == 2 else -1)
    if fallback:  # ORIG fallback -1 meant 'absent' for all six -> corrected +1
        short_c = at_c = ds_c = prefix_c = sub_c = https_c = 1
    else:
        short_c, at_c, ds_c, prefix_c, https_c = (-shortening_o, -at_o, -ds_o,
                                                  -prefix_o, -https_o)
    if mode == "A":
        return {"having_ip_address": has_ip, "url_length": len_c,
                "shortining_service": short_c, "having_at_symbol": at_c,
                "double_slash_redirecting": ds_c, "prefix_suffix": prefix_c,
                "having_sub_domain": sub_c, "https_token": https_c}, fallback
    if mode == "B":  # keep ORIG direction for the 3 disputed features
        return {"having_ip_address": has_ip, "url_length": len_c,
                "shortining_service": shortening_o, "having_at_symbol": at_c,
                "double_slash_redirecting": ds_o, "prefix_suffix": prefix_c,
                "having_sub_domain": sub_c, "https_token": https_o}, fallback
    raise ValueError(mode)

def metrics(yt, pred, prob):
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    return {"Accuracy": accuracy_score(yt, pred),
            "Precision": precision_score(yt, pred, zero_division=0),
            "Recall": recall_score(yt, pred, zero_division=0),
            "F1": f1_score(yt, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(yt, prob),
            "PR-AUC": average_precision_score(yt, prob),
            "MCC": matthews_corrcoef(yt, pred),
            "FPR": fp / (fp + tn), "FNR": fn / (fn + tp),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}

def group_split(Xd, yd, gd, seed, fold=0):
    sp = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(sp.split(Xd, yd, groups=gd)):
        if k == fold:
            return tr, te
    raise ValueError("fold")

def main():
    ds = fetch_ucirepo(id=327)
    X = ds.data.features.copy()
    y_raw = pd.to_numeric(ds.data.targets.iloc[:, 0], errors="coerce")
    X = X.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    y = (y_raw == -1).astype(int)
    valid = y_raw.notna()
    X, y = X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)
    groups = pd.Series(hash_pandas_object(X, index=False).astype("uint64").to_numpy())
    assert X.shape == (11055, 30) and groups.nunique() == 5785
    audit = pd.read_csv(f"{BASE}/duplicate_feature_audit.csv")
    assert (audit["group"].to_numpy() == groups.to_numpy()).all()

    pp = pd.read_csv(f"{BASE}/revision/phreshphish_test_sample_urls.csv")
    y_ext = pp["label"].astype(int).to_numpy()
    assert len(pp) == 10000

    ext = {}
    fb = {}
    for mode in ["orig", "A", "B"]:
        rows, fbs = [], 0
        for u in pp["url"]:
            d, f = extract(u, mode)
            rows.append(d); fbs += int(f)
        ext[mode] = pd.DataFrame(rows, columns=URL8)
        fb[mode] = fbs
    print("fallback (parse-failure) URL count per extractor:", fb)

    def run(mode):
        rows = []
        for seed in SEEDS:
            tr, te = group_split(X, y, groups, seed)
            est = Pipeline([("imputer", SimpleImputer(strategy="median")),
                            ("model", ExtraTreesClassifier(
                                n_estimators=250, max_features="sqrt",
                                class_weight="balanced", random_state=seed,
                                n_jobs=-1))])
            est.fit(X.iloc[tr][URL8], y.iloc[tr])
            pred = est.predict(X.iloc[te][URL8])
            prob = est.predict_proba(X.iloc[te][URL8])[:, 1]
            r = {"Seed": seed, "Scope": "within-UCI"}
            r.update({f"UCI-{k}": v for k, v in metrics(y.iloc[te], pred, prob).items()})
            pe = est.predict(ext[mode]); se = est.predict_proba(ext[mode])[:, 1]
            m = metrics(y_ext, pe, se)
            r.update({"Transfer ROC-AUC": m["ROC-AUC"], "Transfer MCC": m["MCC"],
                      "Transfer Accuracy": m["Accuracy"], "Transfer F1": m["F1"],
                      "Transfer FPR": m["FPR"], "Transfer TPR": m["Recall"],
                      "Transfer TP": m["TP"], "Transfer FP": m["FP"],
                      "Transfer FN": m["FN"], "Transfer TN": m["TN"]})
            rows.append(r)
            print(f"[{mode}] seed {seed}: within ROC {r['UCI-ROC-AUC']:.4f} | "
                  f"transfer ROC {m['ROC-AUC']:.4f} MCC {m['MCC']:.4f}")
        return pd.DataFrame(rows)

    res = {}
    for mode in ["orig", "A", "B"]:
        df = run(mode)
        df.to_csv(f"{OUT}/transfer_rerun_{mode}.csv", index=False)
        res[mode] = df

    # Fixed-split (seed 42) detail for ORIG validation vs saved predictions file
    tr, te = group_split(X, y, groups, 42)
    est = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("model", ExtraTreesClassifier(
                        n_estimators=250, max_features="sqrt",
                        class_weight="balanced", random_state=42, n_jobs=-1))])
    est.fit(X.iloc[tr][URL8], y.iloc[tr])
    saved = pd.read_csv(f"{BASE}/phreshphish_external_predictions.csv")
    for mode in ["orig", "A", "B"]:
        pe = est.predict(ext[mode]); se = est.predict_proba(ext[mode])[:, 1]
        m = metrics(y_ext, pe, se)
        print(f"[fixed42/{mode}] transfer:", {k: round(v, 4) for k, v in m.items()
                                              if k in ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "MCC", "FPR")})
        if mode == "orig":
            agree = int((pe == saved["predicted"].to_numpy()).sum())
            maxpd = float(np.max(np.abs(se - saved["phishing_probability"].to_numpy())))
            print(f"[fixed42/orig] agreement with saved predictions: {agree}/10000, "
                  f"max |prob diff| = {maxpd:.2e}")
            pd.DataFrame({"url": pp["url"], "actual": y_ext, "predicted": pe,
                          "phishing_probability": se}).to_csv(
                              f"{OUT}/phreshphish_predictions_fixed42_orig.csv", index=False)

    # Sensitivity for A (fixed-split model, corrected extractor, varied length bins)
    sens = []
    for sc, lc in [(50, 70), (54, 75), (60, 80)]:
        rows = []
        for u in pp["url"]:
            d, _ = extract(u, "A")
            L = len(str(u).strip())
            d["url_length"] = 1 if L < sc else (0 if L <= lc else -1)
            rows.append(d)
        Xe = pd.DataFrame(rows, columns=URL8)
        pe = est.predict(Xe); se = est.predict_proba(Xe)[:, 1]
        m = metrics(y_ext, pe, se)
        sens.append({"Short/medium cut": sc, "Medium/long cut": lc,
                     "Accuracy": m["Accuracy"], "F1": m["F1"],
                     "ROC-AUC": m["ROC-AUC"], "PR-AUC": m["PR-AUC"], "MCC": m["MCC"]})
    sens = pd.DataFrame(sens)
    sens.to_csv(f"{OUT}/sensitivity_corrected_A.csv", index=False)
    print("\nSensitivity (corrected A, fixed-split model):")
    print(sens.round(4).to_string(index=False))

    # Summary
    summ = {}
    for mode, df in res.items():
        summ[mode] = {
            "transfer ROC mean": float(df["Transfer ROC-AUC"].mean()),
            "transfer ROC sd": float(df["Transfer ROC-AUC"].std(ddof=1)),
            "transfer ROC min": float(df["Transfer ROC-AUC"].min()),
            "transfer ROC max": float(df["Transfer ROC-AUC"].max()),
            "transfer ROC below 0.5 (n seeds)": int((df["Transfer ROC-AUC"] < 0.5).sum()),
            "transfer MCC mean": float(df["Transfer MCC"].mean()),
            "transfer MCC sd": float(df["Transfer MCC"].std(ddof=1)),
            "transfer MCC negative (n seeds)": int((df["Transfer MCC"] < 0).sum()),
            "within ROC mean": float(df["UCI-ROC-AUC"].mean()),
            "within MCC mean": float(df["UCI-MCC"].mean()),
        }
    # Fixed-split A operating characteristics for Section 5.14
    pe = est.predict(ext["A"]); se = est.predict_proba(ext["A"])[:, 1]
    m = metrics(y_ext, pe, se)
    summ["fixed42_A"] = {"TPR": m["Recall"], "FPR": m["FPR"],
                         "TP": m["TP"], "FP": m["FP"], "FN": m["FN"], "TN": m["TN"]}
    for pi in (0.01, 0.0005):
        prec = m["Recall"] * pi / (m["Recall"] * pi + m["FPR"] * (1 - pi))
        summ["fixed42_A"][f"precision@{pi}"] = prec
        summ["fixed42_A"][f"FA-per-detection@{pi}"] = (1 - prec) / prec
    pd.DataFrame({"url": pp["url"], "actual": y_ext, "predicted": pe,
                  "phishing_probability": se}).to_csv(
                      f"{OUT}/phreshphish_predictions_fixed42_A.csv", index=False)
    with open(f"{OUT}/transfer_rerun_summary.json", "w") as f:
        json.dump(summ, f, indent=2)
    print("\nSUMMARY:")
    print(json.dumps(summ, indent=2))

if __name__ == "__main__":
    main()
