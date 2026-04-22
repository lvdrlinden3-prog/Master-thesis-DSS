import os
import re
import json
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.base import clone
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score
from joblib import Parallel, delayed
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings('ignore')
os.makedirs('outputs', exist_ok=True)

# Accumulates metrics for all models; flushed to disk after each model completes
all_metrics = {}
METRICS_FILE = 'outputs/metrics_04_model_training.json'


def _json_safe(v):
    """Convert numpy/tuple values to JSON-serialisable types."""
    if isinstance(v, tuple):
        return list(v)
    if hasattr(v, 'item'):          # numpy scalar
        return v.item()
    return v


def flush_metrics():
    with open(METRICS_FILE, 'w') as _f:
        json.dump(all_metrics, _f, indent=2)
    print(f"  Metrics saved → {METRICS_FILE}")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
NO_SKILL_PRAUC = 0.0157
RANDOM_STATE   = 42
K_NEIGHBORS    = 5        # SMOTE/SMOTEENN — verified safe in scripts 03a/b/c
DATA_FILE      = 'data_final_modeling_ma_v7.xlsx'
TARGET         = 'target_next_year'
FEATURES       = [
    # --- Original 11 (v5) ---
    'profitability', 'leverage', 'cash_ratio', 'fcf_debt',
    'ppe_ratio', 'capex_intensity', 'asset_turnover',
    'interest_burden', 'net_margin', 'rev_growth', 'fcf_volatility',
    # --- New v6 (high coverage) ---
    'firm_size',          # log(at) — 0% null
    'rd_intensity',       # xrd/at — 0% null (zero-imputed)
    'rev_growth_lag1',    # lagged rev_growth — ~1 extra year lost per firm
    'altman_re_ta',       # re/at — ~7% null
]
COLORS = {
    'Logistic Regression': '#2c7bb6',
    'XGBoost':             '#d7191c',
    'MLP':                 '#1a9641',
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def find_output(filename):
    """Locate a file anywhere under outputs/ (handles user-organised subdirs)."""
    matches = list(Path('outputs').rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"'{filename}' not found anywhere under outputs/. "
            "Run the prerequisite script first."
        )
    return str(sorted(matches, key=lambda p: p.stat().st_mtime)[-1])


def load_strategy(filename):
    """Read a key=value strategy file written by scripts 03a/b/c."""
    path = find_output(filename)
    info = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                k, v = line.split('=', 1)
                info[k] = v
    return info


def parse_sampling_strategy(sampler_config):
    """Extract sampling_strategy from a sampler_config string written by scripts 03a/b/c."""
    m = re.search(r'sampling_strategy=([\d.]+)', sampler_config)
    return float(m.group(1)) if m else 0.20


def build_sampler(strategy_name, sampling_strategy=0.20):
    """Return a fresh sampler instance, or None for S1 (no resampling)."""
    if strategy_name.startswith('S1'):
        return None
    if 'SMOTEENN' in strategy_name:
        return SMOTEENN(
            smote=SMOTE(k_neighbors=K_NEIGHBORS, sampling_strategy=sampling_strategy,
                        random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
        )
    if 'SMOTE' in strategy_name:
        return SMOTE(k_neighbors=K_NEIGHBORS, sampling_strategy=sampling_strategy,
                     random_state=RANDOM_STATE)
    if 'RandomOverSampler' in strategy_name:
        return RandomOverSampler(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)
    if 'RandomUnderSampler' in strategy_name:
        return RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)
    raise ValueError(f"Unknown strategy: {strategy_name}")


def _score_fold(estimator, params, X, y, tr_idx, va_idx, sampler=None):
    """Fit a cloned estimator on one fold (with optional resampling) and return PR-AUC."""
    est = clone(estimator).set_params(**params)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    if sampler is not None:
        X_tr, y_tr = clone(sampler).fit_resample(X_tr, y_tr)
    est.fit(X_tr, y_tr)
    return average_precision_score(y[va_idx], est.predict_proba(X[va_idx])[:, 1])


def tqdm_random_search(estimator, param_dist, X, y, cv_folds, n_iter,
                       random_state, desc, sampler=None):
    """
    Randomized search with a tqdm progress bar (one tick per param combo).
    Folds within each combo run in parallel via joblib.
    sampler is cloned and applied inside each fold when provided.
    Returns (best_result_dict, all_results_list).
    """
    param_list = list(ParameterSampler(param_dist, n_iter=n_iter,
                                       random_state=random_state))
    all_results = []
    best_score  = -np.inf

    for params in tqdm(param_list, desc=desc, unit='combo'):
        fold_scores = Parallel(n_jobs=-1)(
            delayed(_score_fold)(estimator, params, X, y, tr_idx, va_idx, sampler)
            for tr_idx, va_idx in cv_folds
        )
        mean_score = float(np.mean(fold_scores))
        all_results.append({'params': params, 'mean_score': mean_score,
                            'fold_scores': fold_scores})
        if mean_score > best_score:
            best_score  = mean_score
            best_result = all_results[-1]

    return best_result, all_results


# ─────────────────────────────────────────────
# STEP 4.1 — Configuration and data loading
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 4.1 — Configuration and data loading")
print("=" * 60)

# Load winning strategies from Scripts 03a / 03b / 03c
lr_info  = load_strategy('selected_imbalance_strategy_lr.txt')
xgb_info = load_strategy('selected_imbalance_strategy_xgb.txt')
mlp_info = load_strategy('selected_imbalance_strategy_mlp.txt')

lr_strategy  = lr_info['selected_strategy']
xgb_strategy = xgb_info['selected_strategy']
mlp_strategy = mlp_info['selected_strategy']

lr_ss  = parse_sampling_strategy(lr_info.get('sampler_config', ''))
xgb_ss = parse_sampling_strategy(xgb_info.get('sampler_config', ''))
mlp_ss = parse_sampling_strategy(mlp_info.get('sampler_config', ''))

print(f"\nImbalance strategies loaded from Scripts 03a/b/c:")
print(f"  LR  : {lr_strategy}  (Val PR-AUC = {lr_info['val_pr_auc']}, ss={lr_ss})")
print(f"  XGB : {xgb_strategy}  (Val PR-AUC = {xgb_info['val_pr_auc']}, ss={xgb_ss})")
print(f"  MLP : {mlp_strategy}  (Val PR-AUC = {mlp_info['val_pr_auc']}, ss={mlp_ss})")

df = pd.read_excel(DATA_FILE)
df['fiscal_year'] = pd.to_datetime(df['datadate']).dt.year

df_train = df[df['fiscal_year'].between(2012, 2021)].copy()
df_val   = df[df['fiscal_year'] == 2022].copy()
df_test  = df[df['fiscal_year'].between(2023, 2024)].copy()

X_train = df_train[FEATURES].values;  y_train = df_train[TARGET].values
X_val   = df_val[FEATURES].values;    y_val   = df_val[TARGET].values
X_test  = df_test[FEATURES].values;   y_test  = df_test[TARGET].values

print(f"\nTrain (2012–2021): {len(y_train):,} obs | {y_train.sum()} pos | "
      f"{y_train.sum()/len(y_train)*100:.2f}%")
print(f"Val  (2022)      : {len(y_val):,} obs | {y_val.sum()} pos")
print(f"Test (2023–2024) : {len(y_test):,} obs | {y_test.sum()} pos")

n_pos_train = int(y_train.sum())
n_neg_train = int((y_train == 0).sum())

# XGBoost scale_pos_weight — used only for S1 (no resampling)
scale_pos_weight = n_neg_train / n_pos_train
print(f"\nXGBoost scale_pos_weight = {scale_pos_weight:.4f}  "
      f"({n_neg_train} neg / {n_pos_train} pos)  [used only when XGB strategy = S1]")

# Build per-model sampler objects
lr_sampler  = build_sampler(lr_strategy,  lr_ss)
xgb_sampler = build_sampler(xgb_strategy, xgb_ss)
mlp_sampler = build_sampler(mlp_strategy, mlp_ss)

# LR: class_weight='balanced' for S1, None when a sampler handles balance
lr_class_weight = 'balanced' if lr_strategy.startswith('S1') else None
# XGBoost: scale_pos_weight only meaningful without a sampler
xgb_scale_pos_w = scale_pos_weight if xgb_strategy.startswith('S1') else 1.0

print(f"\nPer-model config summary:")
print(f"  LR  sampler={type(lr_sampler).__name__ if lr_sampler else 'None'}, "
      f"class_weight={lr_class_weight!r}")
print(f"  XGB sampler={type(xgb_sampler).__name__ if xgb_sampler else 'None'}, "
      f"scale_pos_weight={xgb_scale_pos_w:.2f}")
print(f"  MLP sampler={type(mlp_sampler).__name__ if mlp_sampler else 'None'}")

# ─────────────────────────────────────────────
# STEP 4.2 — Expanding-window CV folds
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4.2 — Expanding-window CV folds (2012–2022)")
print("=" * 60)

df_cv    = df[df['fiscal_year'].between(2012, 2022)].copy()
X_cv     = df_cv[FEATURES].values
y_cv     = df_cv[TARGET].values
years_cv = df_cv['fiscal_year'].values

FOLD_SPECS = [
    (list(range(2012, 2017)), 2017),
    (list(range(2012, 2018)), 2018),
    (list(range(2012, 2019)), 2019),
    (list(range(2012, 2020)), 2020),
    (list(range(2012, 2021)), 2021),
    (list(range(2012, 2022)), 2022),
]

cv_folds = []
print("\n  Fold  Train years         Val year   N_train  N_pos_train  N_val  N_pos_val")
print("  " + "-" * 72)
for fold_i, (train_yrs, val_yr) in enumerate(FOLD_SPECS, 1):
    tr_idx = np.where(np.isin(years_cv, train_yrs))[0]
    va_idx = np.where(years_cv == val_yr)[0]
    n_pos_tr = int(y_cv[tr_idx].sum())
    n_pos_va = int(y_cv[va_idx].sum())
    cv_folds.append((tr_idx, va_idx))
    train_str = '–'.join([str(train_yrs[0]), str(train_yrs[-1])]) \
                if len(train_yrs) > 1 else str(train_yrs[0])
    warn = '  *** <70 pos' if n_pos_tr < 70 else ''
    print(f"  {fold_i}     {train_str:<20} {val_yr}          "
          f"{len(tr_idx):>6,}  {n_pos_tr:>11}  {len(va_idx):>5,}  {n_pos_va:>9}{warn}")

# ─────────────────────────────────────────────
# STEP 4.3 — Model 1: Logistic Regression
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4.3 — Model 1: Logistic Regression")
print("=" * 60)
print(f"  Strategy : {lr_strategy}")
print(f"  Config   : RobustScaler + LR(class_weight={lr_class_weight!r})"
      + (f" with {type(lr_sampler).__name__}(ss={lr_ss})" if lr_sampler else ""))

lr_cv_scores = []
for fold_i, (tr_idx, va_idx) in enumerate(tqdm(cv_folds, desc='LR CV', unit='fold'), 1):
    X_tr_f, y_tr_f = X_cv[tr_idx], y_cv[tr_idx]
    X_va_f, y_va_f = X_cv[va_idx], y_cv[va_idx]

    if lr_sampler is not None:
        X_tr_f, y_tr_f = clone(lr_sampler).fit_resample(X_tr_f, y_tr_f)

    sc_f = RobustScaler()
    X_tr_s = sc_f.fit_transform(X_tr_f)
    X_va_s = sc_f.transform(X_va_f)

    lr_f = LogisticRegression(
        max_iter=1000, solver='lbfgs', C=1.0,
        class_weight=lr_class_weight, random_state=RANDOM_STATE
    )
    lr_f.fit(X_tr_s, y_tr_f)
    ap = average_precision_score(y_va_f, lr_f.predict_proba(X_va_s)[:, 1])
    lr_cv_scores.append(ap)
    print(f"  Fold {fold_i}: Val PR-AUC = {ap:.4f}  "
          f"(lift = {ap/NO_SKILL_PRAUC:.2f}x)")

print(f"\n  LR CV PR-AUC: {np.mean(lr_cv_scores):.4f} +/- {np.std(lr_cv_scores):.4f}")

# Refit final LR on full 2012–2021 training data with best strategy
print(f"\n  Refitting final LR on full training set ...")
X_tr_lr, y_tr_lr = X_train.copy(), y_train.copy()
if lr_sampler is not None:
    X_tr_lr, y_tr_lr = lr_sampler.fit_resample(X_tr_lr, y_tr_lr)
    print(f"  After resampling: {int(y_tr_lr.sum())} pos / {int((y_tr_lr==0).sum())} neg")

lr_scaler = RobustScaler()
X_tr_lr_s = lr_scaler.fit_transform(X_tr_lr)
lr_final = LogisticRegression(
    max_iter=1000, solver='lbfgs', C=1.0,
    class_weight=lr_class_weight, random_state=RANDOM_STATE
)
lr_final.fit(X_tr_lr_s, y_tr_lr)

# Wrap in Pipeline so predict_proba(X_raw) works in downstream scripts
lr_pipeline = Pipeline([('scaler', lr_scaler), ('lr', lr_final)])
with open('outputs/model_baseline_logistic.pkl', 'wb') as f:
    pickle.dump(lr_pipeline, f)
print("  Saved: outputs/model_baseline_logistic.pkl")

lr_val_ap  = average_precision_score(y_val, lr_pipeline.predict_proba(X_val)[:, 1])
lr_val_roc = roc_auc_score(y_val, lr_pipeline.predict_proba(X_val)[:, 1])
print(f"  Val PR-AUC  (final model): {lr_val_ap:.4f}  "
      f"(lift = {lr_val_ap/NO_SKILL_PRAUC:.2f}x)")
print(f"  Val ROC-AUC (final model): {lr_val_roc:.4f}")

all_metrics['logistic_regression'] = {
    'imbalance_strategy':   lr_strategy,
    'cv_pr_auc_mean':       round(float(np.mean(lr_cv_scores)), 4),
    'cv_pr_auc_std':        round(float(np.std(lr_cv_scores)),  4),
    'cv_pr_auc_per_fold':   [round(s, 4) for s in lr_cv_scores],
    'val_pr_auc':           round(lr_val_ap,  4),
    'val_roc_auc':          round(lr_val_roc, 4),
    'lift_vs_noskill':      round(lr_val_ap / NO_SKILL_PRAUC, 3),
}
flush_metrics()

# ─────────────────────────────────────────────
# STEP 4.4 — Model 2: XGBoost
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4.4 — Model 2: XGBoost (RandomizedSearchCV n_iter=30)")
print("=" * 60)
print(f"  Strategy       : {xgb_strategy}")
print(f"  scale_pos_weight = {xgb_scale_pos_w:.2f}")
if xgb_sampler:
    print(f"  Resampler      : {type(xgb_sampler).__name__}(ss={xgb_ss}) applied inside each fold")

xgb_param_dist = {
    'n_estimators':     [300, 500, 800, 1200],
    'max_depth':        [3, 4, 5, 6],
    'learning_rate':    [0.005, 0.01, 0.05, 0.10],
    'subsample':        [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'min_child_weight': [5, 10, 20, 30],
    'gamma':            [0, 0.1, 0.5, 1.0],
}

xgb_base = XGBClassifier(
    scale_pos_weight=xgb_scale_pos_w,
    eval_metric='logloss',
    random_state=RANDOM_STATE,
    n_jobs=1,     # each fold fit uses 1 core; folds parallelised by tqdm_random_search
    verbosity=0,
)

print("\n  Running tqdm_random_search (n_iter=30, 6 folds) ...")
best_xgb_result, _ = tqdm_random_search(
    xgb_base, xgb_param_dist, X_cv, y_cv, cv_folds,
    n_iter=30, random_state=RANDOM_STATE, desc='XGBoost CV',
    sampler=xgb_sampler,
)

best_xgb_params = best_xgb_result['params']
best_xgb_mean   = best_xgb_result['mean_score']
xgb_cv_scores   = best_xgb_result['fold_scores']

print(f"\n  Best mean CV PR-AUC : {best_xgb_mean:.4f}")
print(f"  Best hyperparameters: {best_xgb_params}")
print(f"  Per-fold PR-AUC     : {[round(s, 4) for s in xgb_cv_scores]}")
print(f"  Mean +/- Std          : {np.mean(xgb_cv_scores):.4f} +/- {np.std(xgb_cv_scores):.4f}")

# Refit on full 2012–2021 training data
X_tr_xgb, y_tr_xgb = X_train.copy(), y_train.copy()
if xgb_sampler is not None:
    X_tr_xgb, y_tr_xgb = xgb_sampler.fit_resample(X_tr_xgb, y_tr_xgb)
    print(f"\n  After resampling: {int(y_tr_xgb.sum())} pos / {int((y_tr_xgb==0).sum())} neg")

best_xgb = XGBClassifier(
    **best_xgb_params,
    scale_pos_weight=xgb_scale_pos_w,
    eval_metric='logloss',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0,
)
best_xgb.fit(X_tr_xgb, y_tr_xgb)

with open('outputs/model_xgb.pkl', 'wb') as f:
    pickle.dump(best_xgb, f)
print("\nSaved: outputs/model_xgb.pkl")

xgb_val_proba = best_xgb.predict_proba(X_val)[:, 1]
xgb_val_ap    = average_precision_score(y_val, xgb_val_proba)
xgb_val_roc   = roc_auc_score(y_val, xgb_val_proba)
print(f"  Val PR-AUC  (final model): {xgb_val_ap:.4f}  "
      f"(lift = {xgb_val_ap/NO_SKILL_PRAUC:.2f}x)")
print(f"  Val ROC-AUC (final model): {xgb_val_roc:.4f}")

all_metrics['xgboost'] = {
    'imbalance_strategy':   xgb_strategy,
    'cv_pr_auc_mean':       round(float(np.mean(xgb_cv_scores)), 4),
    'cv_pr_auc_std':        round(float(np.std(xgb_cv_scores)),  4),
    'cv_pr_auc_per_fold':   [round(s, 4) for s in xgb_cv_scores],
    'val_pr_auc':           round(xgb_val_ap,  4),
    'val_roc_auc':          round(xgb_val_roc, 4),
    'lift_vs_noskill':      round(xgb_val_ap / NO_SKILL_PRAUC, 3),
    'best_hyperparams':     {k: _json_safe(v) for k, v in best_xgb_params.items()},
}
flush_metrics()

# ─────────────────────────────────────────────
# STEP 4.5 — Model 3: MLP Neural Network
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4.5 — Model 3: MLP Neural Network (RandomizedSearchCV n_iter=20)")
print("=" * 60)
print(f"  Strategy : {mlp_strategy}")

mlp_param_dist = {
    'mlp__hidden_layer_sizes': [
        (64,), (128,), (256,),
        (64, 32), (128, 64), (256, 128),
        (128, 64, 32), (256, 128, 64),
    ],
    'mlp__activation':          ['relu', 'tanh'],
    'mlp__alpha':               [1e-4, 1e-3, 1e-2, 0.1],
    'mlp__learning_rate_init':  [1e-4, 1e-3, 1e-2],
    'mlp__batch_size':          [64, 128, 256],
    'mlp__max_iter':            [500],
    'mlp__early_stopping':      [True],
    'mlp__validation_fraction': [0.1],
    'mlp__n_iter_no_change':    [10],
}

if mlp_sampler is None:
    print("  Pipeline: RobustScaler -> MLPClassifier (no resampling)")
    mlp_pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('mlp',    MLPClassifier(solver='adam', random_state=RANDOM_STATE)),
    ])
else:
    print(f"  Pipeline: {type(mlp_sampler).__name__}(ss={mlp_ss}) -> RobustScaler -> MLPClassifier")
    mlp_pipe = ImbPipeline([
        ('resampler', mlp_sampler),
        ('scaler',    RobustScaler()),
        ('mlp',       MLPClassifier(solver='adam', random_state=RANDOM_STATE)),
    ])

print("\n  Running tqdm_random_search (n_iter=20, 6 folds) ...")
best_mlp_result, _ = tqdm_random_search(
    mlp_pipe, mlp_param_dist, X_cv, y_cv, cv_folds,
    n_iter=20, random_state=RANDOM_STATE, desc='MLP CV',
    sampler=None,   # resampling is handled inside the pipeline
)

best_mlp_params = best_mlp_result['params']
best_mlp_mean   = best_mlp_result['mean_score']
mlp_cv_scores   = best_mlp_result['fold_scores']

print(f"\n  Best mean CV PR-AUC : {best_mlp_mean:.4f}")
print(f"  Best hyperparameters: {best_mlp_params}")
print(f"  Per-fold PR-AUC     : {[round(s, 4) for s in mlp_cv_scores]}")
print(f"  Mean +/- Std          : {np.mean(mlp_cv_scores):.4f} +/- {np.std(mlp_cv_scores):.4f}")

# Refit on full 2012–2021 training data
mlp_best_kwargs = {k.replace('mlp__', ''): v for k, v in best_mlp_params.items()}

if mlp_sampler is None:
    best_mlp_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('mlp',    MLPClassifier(solver='adam', random_state=RANDOM_STATE,
                                 **mlp_best_kwargs)),
    ])
else:
    best_mlp_pipeline = ImbPipeline([
        ('resampler', build_sampler(mlp_strategy, mlp_ss)),   # fresh sampler for final fit
        ('scaler',    RobustScaler()),
        ('mlp',       MLPClassifier(solver='adam', random_state=RANDOM_STATE,
                                    **mlp_best_kwargs)),
    ])

best_mlp_pipeline.fit(X_train, y_train)

with open('outputs/model_mlp_pipeline.pkl', 'wb') as f:
    pickle.dump(best_mlp_pipeline, f)
print("\nSaved: outputs/model_mlp_pipeline.pkl")

mlp_val_proba = best_mlp_pipeline.predict_proba(X_val)[:, 1]
mlp_val_ap    = average_precision_score(y_val, mlp_val_proba)
mlp_val_roc   = roc_auc_score(y_val, mlp_val_proba)
print(f"  Val PR-AUC  (final model): {mlp_val_ap:.4f}  "
      f"(lift = {mlp_val_ap/NO_SKILL_PRAUC:.2f}x)")
print(f"  Val ROC-AUC (final model): {mlp_val_roc:.4f}")

all_metrics['mlp'] = {
    'imbalance_strategy':   mlp_strategy,
    'cv_pr_auc_mean':       round(float(np.mean(mlp_cv_scores)), 4),
    'cv_pr_auc_std':        round(float(np.std(mlp_cv_scores)),  4),
    'cv_pr_auc_per_fold':   [round(s, 4) for s in mlp_cv_scores],
    'val_pr_auc':           round(mlp_val_ap,  4),
    'val_roc_auc':          round(mlp_val_roc, 4),
    'lift_vs_noskill':      round(mlp_val_ap / NO_SKILL_PRAUC, 3),
    'best_hyperparams':     {k: _json_safe(v) for k, v in best_mlp_params.items()},
}
flush_metrics()

# ─────────────────────────────────────────────
# STEP 4.6 — CV performance summary and Figure 10
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4.6 — CV performance summary and Figure 10")
print("=" * 60)

all_cv_scores = {
    'Logistic Regression': lr_cv_scores,
    'XGBoost':             xgb_cv_scores,
    'MLP':                 mlp_cv_scores,
}

# Table 09
rows = []
for model_name, scores in all_cv_scores.items():
    for fold_i, score in enumerate(scores, 1):
        train_yrs, val_yr = FOLD_SPECS[fold_i - 1]
        rows.append({
            'Model':           model_name,
            'Fold':            fold_i,
            'Train_Years':     '–'.join([str(train_yrs[0]), str(train_yrs[-1])])
                               if len(train_yrs) > 1 else str(train_yrs[0]),
            'Val_Year':        val_yr,
            'Val_PR_AUC':      round(score, 4),
            'Lift_vs_NoSkill': round(score / NO_SKILL_PRAUC, 3),
        })

cv_df = pd.DataFrame(rows)
cv_df.to_csv('outputs/table_09_cv_performance.csv', index=False)
print("\nTable 09 saved: outputs/table_09_cv_performance.csv")

print("\n  CV summary (mean +/- std across 4 folds):")
print(f"  {'Model':<25} Mean PR-AUC   Std    Lift vs no-skill")
print("  " + "-" * 55)
for model_name, scores in all_cv_scores.items():
    print(f"  {model_name:<25} {np.mean(scores):.4f}       "
          f"{np.std(scores):.4f}  "
          f"{np.mean(scores)/NO_SKILL_PRAUC:.2f}x")

# Figure 10 — CV stability
fig, ax = plt.subplots(figsize=(14, 5))

fold_labels = [
    '2012-16->2017\n(F1)', '2012-17->2018\n(F2)', '2012-18->2019\n(F3)',
    '2012-19->2020\n(F4)', '2012-20->2021\n(F5)', '2012-21->2022\n(F6)',
]
markers = {'Logistic Regression': 'o', 'XGBoost': 's', 'MLP': '^'}

for model_name, scores in all_cv_scores.items():
    mean_s = np.mean(scores)
    ax.plot(range(1, 7), scores,
            color=COLORS[model_name], marker=markers[model_name],
            lw=2, ms=8, zorder=3,
            label=f'{model_name} (CV mean = {mean_s:.4f}, '
                  f'lift = {mean_s/NO_SKILL_PRAUC:.2f}x)')

ax.axhline(y=NO_SKILL_PRAUC, color='gray', linestyle='--', lw=1.5,
           label=f'No-skill baseline (AP = {NO_SKILL_PRAUC})')
ax.set_xticks(range(1, 7))
ax.set_xticklabels(fold_labels, fontsize=8)
ax.set_ylabel('Val PR-AUC (Average Precision)')
ax.set_xlabel('Expanding-Window CV Fold')
ax.set_title('Figure 10: Expanding-Window CV Stability — PR-AUC per Fold\n'
             'Logistic Regression  ·  XGBoost  ·  MLP', fontsize=11)
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('outputs/fig_10_cv_stability.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 10 saved: outputs/fig_10_cv_stability.png")

# ─────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────
print("\n=== Script 04 complete. All outputs saved to outputs/ ===")
print("\nImbalance strategies applied:")
print(f"  LR  : {lr_strategy}")
print(f"  XGB : {xgb_strategy}")
print(f"  MLP : {mlp_strategy}")
print("\nFinal model val-set performance:")
print(f"  {'Model':<25} Val PR-AUC  Lift   Val ROC-AUC")
print("  " + "-" * 55)
for name, ap, roc in [
    ('Logistic Regression', lr_val_ap,  lr_val_roc),
    ('XGBoost',             xgb_val_ap, xgb_val_roc),
    ('MLP',                 mlp_val_ap, mlp_val_roc),
]:
    print(f"  {name:<25} {ap:.4f}     {ap/NO_SKILL_PRAUC:.2f}x  {roc:.4f}")
print(f"\n  No-skill PR-AUC baseline: {NO_SKILL_PRAUC}")
