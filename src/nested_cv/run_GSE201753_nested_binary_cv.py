from __future__ import annotations

import json
import shutil
import subprocess
from collections import Counter
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from skrebate import ReliefF
except ImportError as exc:
    raise SystemExit("Missing required package skrebate. Install it before running.") from exc

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "GSE201753" / "GSE201753_raw_counts_GRCh38.p13_NCBI.tsv"
META = ROOT / "data" / "GSE201753" / "GSE201753_binary_metadata_109_samples.csv"
R_SCRIPT = ROOT / "src" / "nested_cv" / "nested_binary_deseq2_preprocess.R"
R_EXE = shutil.which("Rscript")
OUT = ROOT / "results" / "nested_cv"
SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
FEATURE_COUNTS = (10, 25, 50)
METHODS = ("Lasso", "SVM-RFE", "Random Forest", "SVM-RBF", "Double-RBF", "ReliefF", "SigFeature", "Ensemble")
CLASSIFIERS = ("Logistic Regression", "Random Forest", "SVM-Linear")


def stratified_folds(y: np.ndarray, n_splits: int, seed: int):
    from sklearn.model_selection import StratifiedKFold
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def select_features(X: pd.DataFrame, y: np.ndarray, n_features: int):
    scaler = StandardScaler()
    xs = scaler.fit_transform(X)
    names = np.asarray(X.columns)
    valid = xs.std(axis=0) > 0.01
    xs, names = xs[:, valid], names[valid]
    k = min(n_features, xs.shape[1])
    if k < 2:
        raise ValueError("Fewer than two usable genes")

    selected, scores = {}, {}
    lasso = Lasso(alpha=0.01, max_iter=5000).fit(xs, y)
    scores["Lasso"] = np.abs(lasso.coef_)
    selected["Lasso"] = names[np.argsort(scores["Lasso"])[::-1][:k]].tolist()

    linear = SVC(kernel="linear", class_weight="balanced")
    rfe = RFE(linear, n_features_to_select=k, step=0.1).fit(xs, y)
    scores["SVM-RFE"] = -rfe.ranking_.astype(float)
    selected["SVM-RFE"] = names[rfe.support_].tolist()

    rf = RandomForestClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=2,
                                class_weight="balanced", random_state=SEED, n_jobs=-1).fit(xs, y)
    scores["Random Forest"] = rf.feature_importances_
    selected["Random Forest"] = names[np.argsort(scores["Random Forest"])[::-1][:k]].tolist()

    rbf = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=SEED).fit(xs, y)
    confidence = rbf.predict_proba(xs).max(axis=1)
    rbf_scores = np.nan_to_num(np.array([abs(np.corrcoef(xs[:, i], confidence)[0, 1]) for i in range(xs.shape[1])]))
    scores["SVM-RBF"] = rbf_scores
    selected["SVM-RBF"] = names[np.argsort(rbf_scores)[::-1][:k]].tolist()

    stage1 = np.argsort(rbf_scores)[::-1][:max(k, xs.shape[1] // 2)]
    xs1, names1 = xs[:, stage1], names[stage1]
    rbf2 = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=SEED).fit(xs1, y)
    confidence2 = rbf2.predict_proba(xs1).max(axis=1)
    local = np.nan_to_num(np.array([abs(np.corrcoef(xs1[:, i], confidence2)[0, 1]) for i in range(xs1.shape[1])]))
    full = np.zeros(xs.shape[1]); full[stage1] = local
    scores["Double-RBF"] = full
    selected["Double-RBF"] = names1[np.argsort(local)[::-1][:k]].tolist()

    relief = ReliefF(n_features_to_select=k, n_neighbors=min(10, len(y) - 1)).fit(xs, y)
    scores["ReliefF"] = relief.feature_importances_
    selected["ReliefF"] = names[np.argsort(scores["ReliefF"])[::-1][:k]].tolist()

    ovr = OneVsRestClassifier(SVC(kernel="linear", class_weight="balanced", random_state=SEED)).fit(xs, y)
    weights = np.abs(np.vstack([est.coef_ for est in ovr.estimators_])).mean(axis=0)
    tstats = np.zeros(xs.shape[1])
    for i in range(xs.shape[1]):
        vals = [abs(ttest_ind(xs[y == a, i], xs[y == b, i], equal_var=False).statistic)
                for a, b in combinations(np.unique(y), 2)]
        tstats[i] = np.nanmean(vals) if vals else 0
    scores["SigFeature"] = np.nan_to_num(weights * tstats)
    selected["SigFeature"] = names[np.argsort(scores["SigFeature"])[::-1][:k]].tolist()

    borda = np.zeros(xs.shape[1])
    for values in scores.values():
        borda += np.argsort(np.argsort(values))
    selected["Ensemble"] = names[np.argsort(borda)[::-1][:k]].tolist()
    return selected


def classifier(name: str):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=5000, class_weight="balanced", C=0.1, random_state=SEED)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=1500, class_weight="balanced", max_features="sqrt",
                                      min_samples_leaf=2, random_state=SEED, n_jobs=-1)
    return SVC(kernel="linear", class_weight="balanced", C=0.1, probability=True, random_state=SEED)


def fit_score(train_df, valid_df, method, clf_name, k, seed):
    y_train = (train_df.pop("binary_class").to_numpy() == "GCA").astype(int)
    y_valid = (valid_df.pop("binary_class").to_numpy() == "GCA").astype(int)
    train_df.pop("sample_id"); valid_ids = valid_df.pop("sample_id").to_numpy();
    X_train, X_valid = train_df.astype(float), valid_df.astype(float)
    k = min(k, X_train.shape[1])
    smote = SMOTE(random_state=seed, k_neighbors=min(3, Counter(y_train)[0] - 1))
    xr, yr = smote.fit_resample(X_train, y_train)
    xr = pd.DataFrame(xr, columns=X_train.columns)
    genes = select_features(xr, yr, k)[method]
    scaler = StandardScaler().fit(xr[genes])
    model = classifier(clf_name).fit(scaler.transform(xr[genes]), yr)
    xv = scaler.transform(X_valid[genes])
    pred = model.predict(xv)
    prob = model.predict_proba(xv)[:, 1]
    ba = balanced_accuracy_score(y_valid, pred)
    f1 = f1_score(y_valid, pred, average="macro")
    auc = roc_auc_score(y_valid, prob) if len(np.unique(y_valid)) == 2 else np.nan
    return ba, f1, auc, y_valid, pred, prob, valid_ids, genes


def prepare_fold(train_df, valid_df, k, seed):
    y_train = (train_df.pop("binary_class").to_numpy() == "GCA").astype(int)
    y_valid = (valid_df.pop("binary_class").to_numpy() == "GCA").astype(int)
    train_df.pop("sample_id")
    valid_ids = valid_df.pop("sample_id").to_numpy()
    X_train, X_valid = train_df.astype(float), valid_df.astype(float)
    k = min(k, X_train.shape[1])
    smote = SMOTE(random_state=seed, k_neighbors=min(3, Counter(y_train)[0] - 1))
    xr, yr = smote.fit_resample(X_train, y_train)
    xr = pd.DataFrame(xr, columns=X_train.columns)
    selections = select_features(xr, yr, k)
    return xr, yr, X_valid, y_valid, valid_ids, selections


def score_prepared(xr, yr, X_valid, y_valid, valid_ids, genes, clf_name):
    scaler = StandardScaler().fit(xr[genes])
    model = classifier(clf_name).fit(scaler.transform(xr[genes]), yr)
    xv = scaler.transform(X_valid[genes])
    pred = model.predict(xv)
    prob = model.predict_proba(xv)[:, 1]
    ba = balanced_accuracy_score(y_valid, pred)
    f1 = f1_score(y_valid, pred, average="macro")
    auc = roc_auc_score(y_valid, prob) if len(np.unique(y_valid)) == 2 else np.nan
    return ba, f1, auc, y_valid, pred, prob, valid_ids


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(META)
    sample_ids = meta.sample_id.to_numpy()
    y = (meta.binary_class.to_numpy() == "GCA").astype(int)
    outer = stratified_folds(y, OUTER_FOLDS, SEED)
    manifest_rows = []
    outer_inner = {}
    for oi, (otr, ote) in enumerate(outer, 1):
        sid = f"outer_{oi}"
        manifest_rows += [(sid, sample_ids[i], "train") for i in otr]
        manifest_rows += [(sid, sample_ids[i], "validation") for i in ote]
        inner = stratified_folds(y[otr], INNER_FOLDS, SEED + oi)
        outer_inner[oi] = inner
        for ii, (itr_local, iva_local) in enumerate(inner, 1):
            itr, iva = otr[itr_local], otr[iva_local]
            iid = f"outer_{oi}_inner_{ii}"
            manifest_rows += [(iid, sample_ids[i], "train") for i in itr]
            manifest_rows += [(iid, sample_ids[i], "validation") for i in iva]
    manifest = pd.DataFrame(manifest_rows, columns=["split_id", "sample_id", "role"])
    manifest.to_csv(OUT / "split_manifest.csv", index=False)

    prep_summary = OUT / "preprocessed_splits" / "preprocessing_summary.csv"
    if not prep_summary.exists():
        if R_EXE is None:
            raise RuntimeError("Rscript was not found on PATH. Install R and Bioconductor DESeq2 before running.")
        subprocess.run([R_EXE, "--vanilla", str(R_SCRIPT), str(RAW), str(META),
                        str(OUT / "split_manifest.csv"), str(OUT / "preprocessed_splits")], check=True)
    prep = pd.read_csv(prep_summary)
    if not (prep.status == "ok").all():
        raise RuntimeError("One or more nested DESeq2 splits had insufficient significant genes; see preprocessing_summary.csv")

    inner_rows, outer_rows, prediction_rows, gene_rows = [], [], [], []
    for oi, (otr, ote) in enumerate(outer, 1):
        per_config = {}
        for ii in range(1, INNER_FOLDS + 1):
            d = OUT / "preprocessed_splits" / f"outer_{oi}_inner_{ii}"
            tr, va = pd.read_csv(d / "train_expression.csv"), pd.read_csv(d / "validation_expression.csv")
            max_k = tr.shape[1] - 2
            for k in FEATURE_COUNTS:
                if k > max_k:
                    continue
                try:
                    prepared = prepare_fold(tr.copy(), va.copy(), k, SEED + oi * 10 + ii)
                except Exception as exc:
                    for method, clf_name in product(METHODS, CLASSIFIERS):
                        inner_rows.append(dict(outer_fold=oi, inner_fold=ii, n_features=k, method=method,
                                               classifier=clf_name, status=f"error: {exc}"))
                    continue
                xr, yr, xv, yv, ids, selections = prepared
                for method, clf_name in product(METHODS, CLASSIFIERS):
                    try:
                        ba, f1, auc, *_ = score_prepared(xr, yr, xv, yv, ids, selections[method], clf_name)
                    except Exception as exc:
                        inner_rows.append(dict(outer_fold=oi, inner_fold=ii, n_features=k, method=method,
                                               classifier=clf_name, status=f"error: {exc}"))
                        continue
                    row = dict(outer_fold=oi, inner_fold=ii, n_features=k, method=method, classifier=clf_name,
                               balanced_accuracy=ba, macro_f1=f1, roc_auc=auc, status="ok")
                    inner_rows.append(row)
                    per_config.setdefault((k, method, clf_name), []).append((ba, f1))
        viable = [(cfg, vals) for cfg, vals in per_config.items() if len(vals) == INNER_FOLDS]
        if not viable:
            raise RuntimeError(f"No configuration completed all inner folds for outer fold {oi}")
        winner, vals = max(viable, key=lambda z: (np.mean([v[0] for v in z[1]]), np.mean([v[1] for v in z[1]])))
        k, method, clf_name = winner
        d = OUT / "preprocessed_splits" / f"outer_{oi}"
        tr, va = pd.read_csv(d / "train_expression.csv"), pd.read_csv(d / "validation_expression.csv")
        xr, yr, xv, yv, ids, selections = prepare_fold(tr.copy(), va.copy(), k, SEED + oi)
        genes = selections[method]
        ba, f1, auc, yt, yp, pp, ids = score_prepared(xr, yr, xv, yv, ids, genes, clf_name)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        outer_rows.append(dict(outer_fold=oi, selected_n_features=k, selected_method=method,
                               selected_classifier=clf_name, inner_mean_balanced_accuracy=np.mean([v[0] for v in vals]),
                               inner_mean_macro_f1=np.mean([v[1] for v in vals]), balanced_accuracy=ba,
                               macro_f1=f1, roc_auc=auc, sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
                               tn=tn, fp=fp, fn=fn, tp=tp))
        prediction_rows += [dict(outer_fold=oi, sample_id=s, actual=int(a), predicted=int(p), probability_gca=q)
                            for s, a, p, q in zip(ids, yt, yp, pp)]
        gene_rows += [dict(outer_fold=oi, method=method, n_features=k, gene_id=g) for g in genes]

    pd.DataFrame(inner_rows).to_csv(OUT / "inner_cv_all_configurations.csv", index=False)
    outer_df = pd.DataFrame(outer_rows)
    outer_df.to_csv(OUT / "outer_fold_results.csv", index=False)
    pred_df = pd.DataFrame(prediction_rows)
    pred_df.to_csv(OUT / "outer_fold_predictions.csv", index=False)
    gene_df = pd.DataFrame(gene_rows)
    gene_df.to_csv(OUT / "outer_fold_selected_genes.csv", index=False)

    yt, yp, pp = pred_df.actual.to_numpy(), pred_df.predicted.to_numpy(), pred_df.probability_gca.to_numpy()
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    summary = {
        "design": "5-fold outer CV with 3-fold inner selection; all preprocessing repeated inside training folds",
        "n_samples": len(pred_df), "balanced_accuracy": balanced_accuracy_score(yt, yp),
        "macro_f1": f1_score(yt, yp, average="macro"), "roc_auc": roc_auc_score(yt, pp),
        "sensitivity_gca": tp / (tp + fn), "specificity_control": tn / (tn + fp),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "selected_configurations": outer_df[["selected_n_features", "selected_method", "selected_classifier"]].to_dict("records"),
    }
    with open(OUT / "nested_cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
