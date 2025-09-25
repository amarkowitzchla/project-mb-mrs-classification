#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, joblib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RepeatedStratifiedKFold
from sklearn.base           import clone
from imblearn.pipeline      import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing     import StandardScaler, label_binarize
from sklearn.linear_model      import LogisticRegression
from sklearn.svm               import LinearSVC
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics           import (
    accuracy_score, precision_score, f1_score, recall_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
    make_scorer, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
)

# ───────────────────────── CONFIG
DATA_CSV     = "MBmerged-z-scores_MLready.csv"
RANDOM_STATE = 40
N_FOLDS      = 3

warnings.filterwarnings("ignore", category=UserWarning)
os.makedirs("derived", exist_ok=True)
os.makedirs("derived/binary_rocs_cv3", exist_ok=True)

# ───────────────────────── LOAD DATA
df_all = pd.read_csv(DATA_CSV).dropna(subset=["MOLECULAR"])
X_full = df_all.drop("MOLECULAR", axis=1)
y_full = df_all["MOLECULAR"]

# ───────────────────────── TASKS
CLASSES = ["G3", "G4", "SHH", "WNT"]

tasks = {
    # Composite OvR tasks
    "G3G4_vs_Rest"  : lambda s: s.isin(["G3", "G4"]).astype(int),
    "G3SHH_vs_Rest" : lambda s: s.isin(["G3", "SHH"]).astype(int),
    "G4WNT_vs_Rest" : lambda s: s.isin(["G4", "WNT"]).astype(int),

    # Composite pair (group vs group)
    "G3SHH_vs_G4WNT": lambda s: s.map(
        lambda x: 1 if x in ["G3", "SHH"] else (0 if x in ["G4", "WNT"] else np.nan)
    ),
}

# One-vs-Rest tasks for each single class
for c in CLASSES:
    tasks[f"{c}_vs_Rest"] = (lambda c: (lambda s: (s == c).astype(int)))(c)

# Pairwise one-vs-one tasks for all class pairs
def pair_fn(s, a, b):
    return s.map({a: 1, b: 0})   # keep NaNs if not a/b

for a, b in combinations(CLASSES, 2):
    tasks[f"{a}_vs_{b}"] = (lambda a, b: (lambda s: pair_fn(s, a, b)))(a, b)

# Explicit groupings for plotting (no duplicates)
OVR  = [f"{c}_vs_Rest" for c in CLASSES] + ["G3G4_vs_Rest", "G3SHH_vs_Rest", "G4WNT_vs_Rest"]
PAIR = [f"{a}_vs_{b}" for a, b in combinations(CLASSES, 2)] + ["G3SHH_vs_G4WNT"]

# ───────────────────────── MODELS / GRIDS
estimators = {
    "LogisticRegression": LogisticRegression(
        penalty="l1", solver="saga", class_weight="balanced",
        max_iter=10_000, random_state=RANDOM_STATE),
    "SVM": LinearSVC(
        penalty="l2", dual=False, class_weight="balanced",
        max_iter=10_000, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(
        n_jobs=-1, class_weight="balanced", random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}
param_grids = {
    "LogisticRegression": {"clf__C": [0.01, 0.1, 1, 10]},
    "SVM"              : {"clf__C": [0.01, 0.1, 1, 10]},
    "RandomForest"     : {"clf__n_estimators": [100, 200],
                          "clf__max_depth"   : [None, 10, 20]},
    "GradientBoosting" : {"clf__n_estimators": [100, 200],
                          "clf__learning_rate": [0.01, 0.1],
                          "clf__max_depth"   : [3, 5]},
}

cv_inner = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
scoring  = {"F1": make_scorer(f1_score, zero_division=0)}

# ───────────────────────── HELPERS
def safe_sampler(min_size):
    if min_size >= 6:
        k = max(1, min(3, min_size // 2))
        return SMOTE(k_neighbors=k, random_state=RANDOM_STATE)
    return RandomOverSampler(random_state=RANDOM_STATE)

def build_meta(X, y, pipes, chosen):
    meta = pd.DataFrame(index=X.index)
    for task, mname in chosen.items():
        pipe  = pipes[f"{task}__{mname}"]
        y_bin = tasks[task](y).dropna()         # ← drop here
        idx   = y_bin.index
        scores = (pipe.predict_proba(X.loc[idx])[:, 1]
                  if hasattr(pipe, "predict_proba")
                  else pipe.decision_function(X.loc[idx]))
        meta.loc[idx, f"{task}_score"] = scores
    return meta.fillna(0)

def _as_probs_or_scores(pipe, X):
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)

def _binary_metrics(y_true, scores):
    # threshold at 0.5 for proba; 0 for decision_function
    if scores.min() >= 0 and scores.max() <= 1:
        thr = 0.5
    else:
        thr = 0.0
    y_pred = (scores >= thr).astype(int)

    # robust auc: only if both classes present and scores have variation
    auc_val = np.nan
    if len(np.unique(y_true)) == 2 and np.std(scores) > 0:
        try:
            auc_val = roc_auc_score(y_true, scores)
        except Exception:
            auc_val = np.nan

    return {
        "n": len(y_true),
        "pos_rate": float(np.mean(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true))==2 else np.nan,
        "roc_auc": auc_val,
    }

def _flatten_params(p):
    # flatten param dict into simple {param: value}
    return {k.replace("clf__", ""): v for k, v in p.items()}

def _save_confusion(y_true, y_pred, labels, out_png, out_csv, normalize=None, title=None):
    """
    Save confusion matrix as PNG and CSV.
    normalize: None | 'true' | 'pred' | 'all' (sklearn semantics)
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    df_cm.to_csv(out_csv)
    plt.figure(figsize=(6,5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(include_values=True, xticks_rotation=45, colorbar=True)
    plt.title(title or "Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    return cm

# ───────────────────────── OUTER CV
outer_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
mean_fpr = np.linspace(0,1,401); fold_tprs, fold_aucs = [], []

# **NEW** collectors for binary ROC averaging
bin_mean_tprs = {t: [] for t in tasks}   # task → list[interpolated TPR]
bin_aucs      = {t: [] for t in tasks}

# collectors for spreadsheets
inner_cv_rows = []           # every grid candidate (inner-CV) per task & fold
outer_bin_rows = []          # selected binary models' holdout metrics per fold
outer_meta_rows = []         # meta-classifier holdout metrics per fold

# **NEW** aggregate validation confusion matrices
agg_val_cms = []
agg_labels  = None

for fold, (tr_idx, va_idx) in enumerate(outer_cv.split(X_full, y_full), 1):
    print(f"\n════════ Fold {fold}/{N_FOLDS} ════════")
    fold_dir = Path(f"derived/fold{fold}")
    fold_dir.mkdir(exist_ok=True, parents=True)

    X_tr, X_va = X_full.iloc[tr_idx], X_full.iloc[va_idx]
    y_tr, y_va = y_full.iloc[tr_idx], y_full.iloc[va_idx]

    best_pipes, best_models = {}, {}

    # ─── train binary
    for task, labelfn in tasks.items():
        y_raw = labelfn(y_tr)             # may contain NaNs (not in task)
        y_bin = y_raw.dropna()
        if y_bin.nunique() < 2:
            continue

        X_bin = X_tr.loc[y_bin.index]
        sampler = safe_sampler(y_bin.value_counts().min())

        best_f1, best_name, best_grid = -np.inf, None, None
        for mname, est in estimators.items():
            pipe = ImbPipeline([
                ("sampler", sampler),
                ("vth", VarianceThreshold()),
                ("scaler", StandardScaler()),
                ("clf", est)
            ])
            grid = GridSearchCV(
                pipe, param_grids[mname],
                cv=cv_inner, scoring="f1", n_jobs=-1, verbose=0, return_train_score=True
            )
            grid.fit(X_bin, y_bin)

            # ── LOG ALL CANDIDATES from this grid into inner_cv_rows
            cvres = grid.cv_results_
            for i in range(len(cvres["params"])):
                row = {
                    "fold": fold,
                    "task": task,
                    "model_name": mname,
                    "scoring": "f1",
                    "rank_test_score": int(cvres["rank_test_score"][i]),
                    "mean_test_score": float(cvres["mean_test_score"][i]),
                    "std_test_score": float(cvres["std_test_score"][i]),
                    "mean_train_score": float(cvres.get("mean_train_score", [np.nan]*len(cvres["params"]))[i]),
                    "std_train_score": float(cvres.get("std_train_score",  [np.nan]*len(cvres["params"]))[i]),
                }
                row.update(_flatten_params(cvres["params"][i]))
                inner_cv_rows.append(row)

            # track the single best model (by F1)
            if grid.best_score_ > best_f1:
                best_f1, best_name, best_grid = grid.best_score_, mname, grid

            best_pipes[f"{task}__{mname}"] = grid.best_estimator_

        if best_name:
            best_models[task] = best_name
            print(f"  {task:18s} → {best_name:17s} (F1={best_f1:.3f})")

            # ── EVALUATE SELECTED BINARY on TRAIN & VAL for this fold and log
            sel = best_pipes[f"{task}__{best_name}"]
            # TRAIN (on task-eligible rows)
            train_scores = _as_probs_or_scores(sel, X_bin)
            train_metrics = _binary_metrics(y_bin.values.astype(int), train_scores)
            # VAL (use task-eligible rows in val)
            y_va_bin = labelfn(y_va).dropna()
            if y_va_bin.nunique() >= 2:
                X_va_task = X_va.loc[y_va_bin.index]
                val_scores = _as_probs_or_scores(sel, X_va_task)
                val_metrics = _binary_metrics(y_va_bin.values.astype(int), val_scores)
            else:
                val_metrics = {k: np.nan for k in train_metrics.keys()}

            # param dump
            sel_params = {}
            try:
                sel_params = _flatten_params(best_grid.best_params_)
            except Exception:
                pass

            outer_bin_rows.append({
                "fold": fold,
                "task": task,
                "model_name": best_name,
                "phase": "train",
                **sel_params,
                **train_metrics
            })
            outer_bin_rows.append({
                "fold": fold,
                "task": task,
                "model_name": best_name,
                "phase": "val",
                **sel_params,
                **val_metrics
            })

    # persist binary pipes for this fold
    joblib.dump({"pipes": best_pipes, "best_models": best_models},
                fold_dir / "binary_pipes.joblib")

    # ─── per-task ROC on validation split
    mean_fpr = np.linspace(0,1,401)
    for task, mname in best_models.items():
        pipe   = best_pipes[f"{task}__{mname}"]
        y_bin  = tasks[task](y_va).dropna()
        if y_bin.nunique()<2:
            continue
        X_task = X_va.loc[y_bin.index]
        s      = _as_probs_or_scores(pipe, X_task)
        fpr, tpr, _ = roc_curve(y_bin, s)
        bin_mean_tprs[task].append(np.interp(mean_fpr, fpr, tpr))
        bin_aucs[task].append(auc(fpr, tpr))

    # ─── meta features / classifier
    X_meta_tr = build_meta(X_tr, y_tr, best_pipes, best_models)
    X_meta_va = build_meta(X_va, y_va, best_pipes, best_models)

    meta = LogisticRegression(multi_class="multinomial",
                              class_weight="balanced",
                              max_iter=10_000, random_state=RANDOM_STATE)
    grid = GridSearchCV(meta, {"C":[0.01,0.1,1,10]},
                        cv=cv_inner, scoring="f1_macro", n_jobs=-1, return_train_score=True)
    grid.fit(X_meta_tr, y_tr)
    best_meta = clone(meta).set_params(**grid.best_params_).fit(X_meta_tr, y_tr)

    # Save meta model + the exact meta feature column order used in this fold
    joblib.dump(best_meta, fold_dir / "meta_model.joblib")
    pd.Series(list(X_meta_tr.columns)).to_csv(fold_dir / "meta_features.csv", index=False, header=False)
    pd.Series(best_meta.classes_).to_csv(fold_dir / "meta_classes.csv", index=False, header=False)

    # ─── meta metrics on train/val (holdout = val)
    # TRAIN
    y_pred_tr = best_meta.predict(X_meta_tr)
    y_prob_tr = best_meta.predict_proba(X_meta_tr)
    meta_train = {
        "fold": fold, "phase": "train",
        "accuracy": accuracy_score(y_tr, y_pred_tr),
        "balanced_accuracy": balanced_accuracy_score(y_tr, y_pred_tr),
        "f1_macro": f1_score(y_tr, y_pred_tr, average="macro"),
        "f1_weighted": f1_score(y_tr, y_pred_tr, average="weighted"),
    }
    try:
        y_bin_tr = label_binarize(y_tr, classes=best_meta.classes_)
        aucs = []
        for i in range(y_bin_tr.shape[1]):
            if len(np.unique(y_bin_tr[:, i])) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_bin_tr[:, i], y_prob_tr[:, i])
            aucs.append(auc(fpr, tpr))
        meta_train["roc_auc_ovr_macro"] = float(np.mean(aucs)) if aucs else np.nan
    except Exception:
        meta_train["roc_auc_ovr_macro"] = np.nan

    # VAL
    y_pred_va = best_meta.predict(X_meta_va)
    y_prob_va = best_meta.predict_proba(X_meta_va)
    meta_val = {
        "fold": fold, "phase": "val",
        "accuracy": accuracy_score(y_va, y_pred_va),
        "balanced_accuracy": balanced_accuracy_score(y_va, y_pred_va),
        "f1_macro": f1_score(y_va, y_pred_va, average="macro"),
        "f1_weighted": f1_score(y_va, y_pred_va, average="weighted"),
    }
    try:
        y_bin_va = label_binarize(y_va, classes=best_meta.classes_)
        aucs = []
        for i in range(y_bin_va.shape[1]):
            if len(np.unique(y_bin_va[:, i])) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_bin_va[:, i], y_prob_va[:, i])
            aucs.append(auc(fpr, tpr))
        meta_val["roc_auc_ovr_macro"] = float(np.mean(aucs)) if aucs else np.nan
    except Exception:
        meta_val["roc_auc_ovr_macro"] = np.nan

    # store
    meta_params = {"meta_C": grid.best_params_.get("C")}
    outer_meta_rows.append({**{"model_name": "MultinomialLogReg"}, **meta_params, **meta_train})
    outer_meta_rows.append({**{"model_name": "MultinomialLogReg"}, **meta_params, **meta_val})

    # ─── SAVE META CONFUSION MATRICES (per fold)
    _save_confusion(
        y_tr, y_pred_tr, labels=list(best_meta.classes_),
        out_png=fold_dir / "meta_confusion_train.png",
        out_csv=fold_dir / "meta_confusion_train.csv",
        normalize=None, title=f"Meta Confusion – TRAIN (fold {fold})"
    )
    cm_val = _save_confusion(
        y_va, y_pred_va, labels=list(best_meta.classes_),
        out_png=fold_dir / "meta_confusion_val.png",
        out_csv=fold_dir / "meta_confusion_val.csv",
        normalize=None, title=f"Meta Confusion – VAL (fold {fold})"
    )
    agg_val_cms.append(cm_val)
    if agg_labels is None:
        agg_labels = list(best_meta.classes_)

    # ─── meta ROC fold
    mean_fpr = np.linspace(0,1,401)
    prob = y_prob_va
    y_bin = label_binarize(y_va, classes=best_meta.classes_)
    mean_tpr = np.zeros_like(mean_fpr); cnt = 0
    for i in range(y_bin.shape[1]):
        if len(np.unique(y_bin[:, i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], prob[:, i])
        mean_tpr += np.interp(mean_fpr, fpr, tpr); cnt += 1
    mean_tpr /= max(cnt, 1)
    auc_fold = auc(mean_fpr, mean_tpr)
    fold_tprs.append(mean_tpr); fold_aucs.append(auc_fold)

# ───────────────────────── AGGREGATE META ROC
fold_tprs = np.vstack(fold_tprs)
mean_fpr = np.linspace(0,1,401)
meta_mean = fold_tprs.mean(axis=0); meta_std = fold_tprs.std(axis=0)
meta_auc  = auc(mean_fpr, meta_mean)
plt.figure(figsize=(6,6))
plt.plot(mean_fpr, meta_mean, lw=2, label=f"Mean ROC (AUC={meta_auc:.2f})")
plt.fill_between(mean_fpr, np.maximum(meta_mean-meta_std,0),
                 np.minimum(meta_mean+meta_std,1), alpha=.25,label="±1 SD")
plt.plot([0,1],[0,1],'k--',lw=1); plt.xlim([0,1]); plt.ylim([0,1.05])
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("3-Fold Averaged ROC – Meta-Classifier"); plt.legend()
plt.tight_layout(); plt.savefig("derived/meta_classifier_roc_cv3.png",dpi=300)

# ───────────────────────── AGGREGATE META CONFUSION (validation only)
if agg_val_cms and agg_labels is not None:
    cm_sum = np.sum(np.stack(agg_val_cms, axis=0), axis=0)  # summed counts
    # Save counts
    df_sum = pd.DataFrame(cm_sum, index=agg_labels, columns=agg_labels)
    df_sum.to_csv("derived/meta_confusion_val_SUM.csv")
    # Plot counts
    plt.figure(figsize=(6,5))
    ConfusionMatrixDisplay(confusion_matrix=cm_sum, display_labels=agg_labels)\
        .plot(include_values=True, xticks_rotation=45, colorbar=True)
    plt.title("Meta Confusion – VAL (SUM over folds)")
    plt.tight_layout()
    plt.savefig("derived/meta_confusion_val_SUM.png", dpi=300)
    plt.close()

    # Row-normalized (true-rate) version
    cm_norm = cm_sum.astype(float) / np.maximum(cm_sum.sum(axis=1, keepdims=True), 1.0)
    pd.DataFrame(cm_norm, index=agg_labels, columns=agg_labels)\
        .to_csv("derived/meta_confusion_val_SUM_normalized.csv")
    plt.figure(figsize=(6,5))
    ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=agg_labels)\
        .plot(include_values=True, xticks_rotation=45, colorbar=True)
    plt.title("Meta Confusion – VAL (SUM over folds, row-normalized)")
    plt.tight_layout()
    plt.savefig("derived/meta_confusion_val_SUM_normalized.png", dpi=300)
    plt.close()

# ───────────────────────── COMPOSITE BINARY ROC PLOTS
def composite(task_list, title, fname):
    cmap = plt.cm.get_cmap("tab10", len(task_list))
    plt.figure(figsize=(6,6))
    for i, t in enumerate(task_list):
        if not bin_mean_tprs[t]: continue
        mean_tpr = np.mean(bin_mean_tprs[t], axis=0)
        mean_auc = np.mean(bin_aucs[t])
        plt.plot(mean_fpr, mean_tpr, color=cmap(i), lw=2,
                 label=f"{t} (AUC={mean_auc:.2f})")
    plt.plot([0,1],[0,1],'k--',lw=1); plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title); plt.legend(fontsize="small")
    plt.tight_layout()
    out = f"derived/binary_rocs_cv3/{fname}"
    plt.savefig(out,dpi=300); plt.close()
    print(f"Saved composite ROC → {out}")

composite(OVR,   "OvR Tasks – 5-Fold Avg",  "summary_ovr.png")
composite(PAIR,  "Pairwise Tasks – 5-Fold Avg", "summary_pairwise.png")

print(f"\nMeta AUC (mean of folds) = {meta_auc:.3f}")

# ─── write spreadsheets
pd.DataFrame(inner_cv_rows).to_csv("derived/metrics_inner_cv.csv", index=False)
pd.DataFrame(outer_bin_rows).to_csv("derived/metrics_outer_binary_eval.csv", index=False)
pd.DataFrame(outer_meta_rows).to_csv("derived/metrics_outer_meta_eval.csv", index=False)

print("Saved:")
print("  • derived/metrics_inner_cv.csv")
print("  • derived/metrics_outer_binary_eval.csv")
print("  • derived/metrics_outer_meta_eval.csv")
print("  • derived/*/meta_confusion_{train,val}.{csv,png}")
print("  • derived/meta_confusion_val_SUM.{csv,png}")
print("  • derived/meta_confusion_val_SUM_normalized.{csv,png}")
