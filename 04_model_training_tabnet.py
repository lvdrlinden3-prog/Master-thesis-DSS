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
from sklearn.metrics import average_precision_score, roc_auc_score
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

try:
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
except ImportError:
    raise ImportError(
        "pytorch-tabnet is required: pip install pytorch-tabnet"
    )

warnings.filterwarnings('ignore')
os.makedirs('outputs', exist_ok=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
NO_SKILL_PRAUC = 0.0157
RANDOM_STATE   = 42
K_NEIGHBORS    = 5
DATA_FILE      = 'data_final_modeling_ma_v7.xlsx'
TARGET         = 'target_next_year'
FEATURES       = [
    'profitability', 'leverage', 'cash_ratio', 'fcf_debt',
    'ppe_ratio', 'capex_intensity', 'asset_turnover',
    'interest_burden', 'net_margin', 'rev_growth', 'fcf_volatility',
    'firm_size', 'rd_intensity', 'rev_growth_lag1', 'altman_re_ta',
]

# TABnet training constants
TABNET_MAX_EPOCHS         = 100   
TABNET_PATIENCE           = 12
TABNET_BATCH_SIZE         = 512
TABNET_VIRTUAL_BATCH_SIZE = 64

COLORS = {
    'Logistic Regression': '#2c7bb6',
    'XGBoost':             '#d7191c',
    'MLP':                 '#1a9641',
    'TABnet':              '#ff7f00',
}

METRICS_FILE = 'outputs/metrics_04_tabnet_model_training.json'
all_metrics  = {}


def _json_safe(v):
    if isinstance(v, tuple):
        return list(v)
    if hasattr(v, 'item'):
        return v.item()
    return v


def flush_metrics():
    with open(METRICS_FILE, 'w') as _f:
        json.dump(all_metrics, _f, indent=2)
    print(f"  Metrics saved → {METRICS_FILE}")


# ─────────────────────────────────────────────
# TabNetWrapper — thin sklearn-compatible wrapper saved with the model
# ─────────────────────────────────────────────
class TabNetWrapper:
    """
    RobustScaler + TabNetClassifier in a single object.
    Provides predict_proba(X_raw) so downstream scripts require no changes.
    Must be re-defined identically in scripts 05/06/07 before unpickling.
    """
    def __init__(self, scaler, model):
        self.scaler = scaler
        self.model  = model

    def predict_proba(self, X):
        X_sc = self.scaler.transform(np.asarray(X)).astype(np.float32)
        return self.model.predict_proba(X_sc)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def find_output(filename):
    matches = list(Path('outputs').rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"'{filename}' not found anywhere under outputs/. "
            "Run the prerequisite script first."
        )
    return str(sorted(matches, key=lambda p: p.stat().st_mtime)[-1])


def load_strategy(filename):
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
    m = re.search(r'sampling_strategy=([\d.]+)', sampler_config)
    return float(m.group(1)) if m else 0.20


def build_sampler(strategy_name, sampling_strategy=0.20):
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


def _es_split(X, y, frac=0.10, seed=RANDOM_STATE):
    """Carve a held-out early-stopping set from training data."""
    n = len(X)
    n_es = max(2, int(frac * n))
    rng = np.random.default_rng(seed)
    es_idx = rng.choice(n, size=n_es, replace=False)
    fit_mask = np.ones(n, dtype=bool)
    fit_mask[es_idx] = False
    return X[fit_mask], y[fit_mask], X[es_idx], y[es_idx]


def tabnet_score_fold(arch_params, X, y, tr_idx, va_idx, sampler=None):
    """
    Fit a fresh TabNetClassifier on one expanding-window fold.
    Resampling applied inside the fold (no data leakage).
    Early stopping is on a 10% held-out subset of the (resampled) training fold.
    Returns PR-AUC on the fold's validation set.
    """
    X_tr, y_tr = X[tr_idx].copy(), y[tr_idx].copy()
    X_va, y_va = X[va_idx], y[va_idx]

    if sampler is not None:
        X_tr, y_tr = clone(sampler).fit_resample(X_tr, y_tr)

    scaler = RobustScaler()
    X_tr_sc = scaler.fit_transform(X_tr).astype(np.float32)
    X_va_sc = scaler.transform(X_va).astype(np.float32)
    y_tr_int = y_tr.astype(int)

    X_fit, y_fit, X_es, y_es = _es_split(X_tr_sc, y_tr_int)

    model = TabNetClassifier(
        n_d=arch_params.get('n_d', 16),
        n_a=arch_params.get('n_a', 16),
        n_steps=arch_params.get('n_steps', 3),
        gamma=arch_params.get('gamma', 1.3),
        n_independent=arch_params.get('n_independent', 2),
        n_shared=arch_params.get('n_shared', 2),
        lambda_sparse=arch_params.get('lambda_sparse', 1e-3),
        mask_type=arch_params.get('mask_type', 'sparsemax'),
        optimizer_fn=torch.optim.Adam,
        optimizer_params={'lr': arch_params.get('lr', 0.02)},
        seed=RANDOM_STATE,
        verbose=0,
    )
    model.fit(
        X_fit, y_fit,
        eval_set=[(X_es, y_es)],
        eval_metric=['auc'],
        max_epochs=TABNET_MAX_EPOCHS,
        patience=TABNET_PATIENCE,
        batch_size=TABNET_BATCH_SIZE,
        virtual_batch_size=TABNET_VIRTUAL_BATCH_SIZE,
    )
    return average_precision_score(y_va, model.predict_proba(X_va_sc)[:, 1])


def tabnet_random_search(X, y, cv_folds, param_dist, n_iter, sampler=None):
    """
    Random search over TABnet architecture parameters.
    Folds run sequentially (PyTorch threading conflicts with joblib multiprocessing).
    Returns (best_result_dict, all_results_list).
    """
    param_list = list(ParameterSampler(param_dist, n_iter=n_iter,
                                       random_state=RANDOM_STATE))
    all_results = []
    best_score  = -np.inf
    best_result = None

    for params in tqdm(param_list, desc='TABnet CV', unit='combo'):
        fold_scores = [
            tabnet_score_fold(params, X, y, tr_idx, va_idx, sampler)
            for tr_idx, va_idx in cv_folds
        ]
        mean_score = float(np.mean(fold_scores))
        result = {'params': params, 'mean_score': mean_score,
                  'fold_scores': fold_scores}
        all_results.append(result)
        if mean_score > best_score:
            best_score  = mean_score
            best_result = result
        tqdm.write(f"  mean CV PR-AUC={mean_score:.4f} | params={params}")

    return best_result, all_results


# ─────────────────────────────────────────────
# STEP 4t.1 — Configuration and data loading
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 4t.1 — Configuration and data loading")
print("=" * 60)

tabnet_info     = load_strategy('selected_imbalance_strategy_tabnet.txt')
tabnet_strategy = tabnet_info['selected_strategy']
tabnet_ss       = parse_sampling_strategy(tabnet_info.get('sampler_config', ''))

print(f"\nImbalance strategy loaded from Script 03d:")
print(f"  TABnet : {tabnet_strategy}  "
      f"(Val PR-AUC = {tabnet_info['val_pr_auc']}, ss={tabnet_ss})")

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

tabnet_sampler = build_sampler(tabnet_strategy, tabnet_ss)
print(f"\nSampler: {type(tabnet_sampler).__name__ if tabnet_sampler else 'None'}")

# ─────────────────────────────────────────────
# STEP 4t.2 — Expanding-window CV folds
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4t.2 — Expanding-window CV folds (2012–2022)")
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
# STEP 4t.3 — TABnet hyperparameter search
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4t.3 — TABnet: RandomizedSearchCV (n_iter=10, 6 folds)")
print("=" * 60)
print(f"  Strategy : {tabnet_strategy}")
if tabnet_sampler:
    print(f"  Resampler: {type(tabnet_sampler).__name__}(ss={tabnet_ss}) applied inside each fold")
print("  Note: folds run sequentially (PyTorch thread safety)")

tabnet_param_dist = {
    'n_d':           [8, 16, 32, 64],
    'n_a':           [8, 16, 32, 64],
    'n_steps':       [3, 4, 5, 6],
    'gamma':         [1.0, 1.3, 1.5, 2.0],
    'n_independent': [1, 2],
    'n_shared':      [1, 2],
    'lambda_sparse': [1e-4, 1e-3, 1e-2],
    'mask_type':     ['sparsemax', 'entmax'],
    'lr':            [5e-3, 1e-2, 2e-2],
}

print(f"\n  Search space: {sum(len(v) for v in tabnet_param_dist.values())} "
      f"total options across {len(tabnet_param_dist)} hyperparameters")
print("  Running search (this may take 20–60 minutes) ...")

best_tabnet_result, _ = tabnet_random_search(
    X_cv, y_cv, cv_folds, tabnet_param_dist,
    n_iter=10, sampler=tabnet_sampler
)

best_tabnet_params = best_tabnet_result['params']
best_tabnet_mean   = best_tabnet_result['mean_score']
tabnet_cv_scores   = best_tabnet_result['fold_scores']

print(f"\n  Best mean CV PR-AUC : {best_tabnet_mean:.4f}")
print(f"  Best hyperparameters: {best_tabnet_params}")
print(f"  Per-fold PR-AUC     : {[round(s, 4) for s in tabnet_cv_scores]}")
print(f"  Mean +/- Std          : {np.mean(tabnet_cv_scores):.4f} +/- "
      f"{np.std(tabnet_cv_scores):.4f}")

# ─────────────────────────────────────────────
# STEP 4t.4 — Refit on full 2012–2021 training set
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4t.4 — Refit final TABnet on full training set (2012–2021)")
print("=" * 60)

X_tr_final, y_tr_final = X_train.copy(), y_train.copy()
if tabnet_sampler is not None:
    X_tr_final, y_tr_final = tabnet_sampler.fit_resample(X_tr_final, y_tr_final)
    print(f"  After resampling: {int(y_tr_final.sum())} pos / "
          f"{int((y_tr_final == 0).sum())} neg")

final_scaler = RobustScaler()
X_tr_sc = final_scaler.fit_transform(X_tr_final).astype(np.float32)
X_val_sc = final_scaler.transform(X_val).astype(np.float32)
y_tr_int = y_tr_final.astype(int)

# Use val set (2022) as early stopping monitor for the final model
# This is acceptable for the final refit (not CV fold evaluation)
final_tabnet = TabNetClassifier(
    n_d=best_tabnet_params.get('n_d', 16),
    n_a=best_tabnet_params.get('n_a', 16),
    n_steps=best_tabnet_params.get('n_steps', 3),
    gamma=best_tabnet_params.get('gamma', 1.3),
    n_independent=best_tabnet_params.get('n_independent', 2),
    n_shared=best_tabnet_params.get('n_shared', 2),
    lambda_sparse=best_tabnet_params.get('lambda_sparse', 1e-3),
    mask_type=best_tabnet_params.get('mask_type', 'sparsemax'),
    optimizer_fn=torch.optim.Adam,
    optimizer_params={'lr': best_tabnet_params.get('lr', 0.02)},
    seed=RANDOM_STATE,
    verbose=1,
)
final_tabnet.fit(
    X_tr_sc, y_tr_int,
    eval_set=[(X_val_sc, y_val.astype(int))],
    eval_metric=['auc'],
    max_epochs=TABNET_MAX_EPOCHS,
    patience=TABNET_PATIENCE,
    batch_size=TABNET_BATCH_SIZE,
    virtual_batch_size=TABNET_VIRTUAL_BATCH_SIZE,
)

# Wrap scaler + model for seamless downstream use
tabnet_pipeline = TabNetWrapper(final_scaler, final_tabnet)

with open('outputs/model_tabnet.pkl', 'wb') as f:
    pickle.dump(tabnet_pipeline, f)
print("\nSaved: outputs/model_tabnet.pkl")

tabnet_val_proba = tabnet_pipeline.predict_proba(X_val)[:, 1]
tabnet_val_ap    = average_precision_score(y_val, tabnet_val_proba)
tabnet_val_roc   = roc_auc_score(y_val, tabnet_val_proba)
print(f"  Val PR-AUC  (final model): {tabnet_val_ap:.4f}  "
      f"(lift = {tabnet_val_ap/NO_SKILL_PRAUC:.2f}x)")
print(f"  Val ROC-AUC (final model): {tabnet_val_roc:.4f}")

all_metrics['tabnet'] = {
    'imbalance_strategy':   tabnet_strategy,
    'cv_pr_auc_mean':       round(float(np.mean(tabnet_cv_scores)), 4),
    'cv_pr_auc_std':        round(float(np.std(tabnet_cv_scores)),  4),
    'cv_pr_auc_per_fold':   [round(s, 4) for s in tabnet_cv_scores],
    'val_pr_auc':           round(tabnet_val_ap,  4),
    'val_roc_auc':          round(tabnet_val_roc, 4),
    'lift_vs_noskill':      round(tabnet_val_ap / NO_SKILL_PRAUC, 3),
    'best_hyperparams':     {k: _json_safe(v) for k, v in best_tabnet_params.items()},
}
flush_metrics()

# ─────────────────────────────────────────────
# STEP 4t.5 — CV performance table and Figure 10_tabnet
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4t.5 — CV performance table and Figure 10_tabnet")
print("=" * 60)

rows = []
for fold_i, score in enumerate(tabnet_cv_scores, 1):
    train_yrs, val_yr = FOLD_SPECS[fold_i - 1]
    rows.append({
        'Model':           'TABnet',
        'Fold':            fold_i,
        'Train_Years':     '–'.join([str(train_yrs[0]), str(train_yrs[-1])])
                           if len(train_yrs) > 1 else str(train_yrs[0]),
        'Val_Year':        val_yr,
        'Val_PR_AUC':      round(score, 4),
        'Lift_vs_NoSkill': round(score / NO_SKILL_PRAUC, 3),
    })

# Attempt to merge with existing 3-model CV table if available
try:
    existing_cv = pd.read_csv(find_output('table_09_cv_performance.csv'))
    cv_df = pd.concat([existing_cv, pd.DataFrame(rows)], ignore_index=True)
    print("  Merged with existing table_09_cv_performance.csv (3 models)")
except Exception:
    cv_df = pd.DataFrame(rows)
    print("  table_09_cv_performance.csv not found — saving TABnet-only table")

cv_df.to_csv('outputs/table_09_tabnet_cv_performance.csv', index=False)
print("Table saved: outputs/table_09_tabnet_cv_performance.csv")

# Figure 10_tabnet — TABnet CV stability (standalone)
fold_labels = [
    '2012-16->2017\n(F1)', '2012-17->2018\n(F2)', '2012-18->2019\n(F3)',
    '2012-19->2020\n(F4)', '2012-20->2021\n(F5)', '2012-21->2022\n(F6)',
]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(range(1, 7), tabnet_cv_scores,
        color=COLORS['TABnet'], marker='D', lw=2, ms=8, zorder=3,
        label=f"TABnet (CV mean = {np.mean(tabnet_cv_scores):.4f}, "
              f"lift = {np.mean(tabnet_cv_scores)/NO_SKILL_PRAUC:.2f}x)")
ax.axhline(y=NO_SKILL_PRAUC, color='gray', linestyle='--', lw=1.5,
           label=f'No-skill baseline (AP = {NO_SKILL_PRAUC})')
ax.set_xticks(range(1, 7))
ax.set_xticklabels(fold_labels, fontsize=8)
ax.set_ylabel('Val PR-AUC (Average Precision)')
ax.set_xlabel('Expanding-Window CV Fold')
ax.set_title('Figure 10_tabnet: Expanding-Window CV Stability — TABnet PR-AUC per Fold',
             fontsize=11)
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig('outputs/fig_10_tabnet_cv_stability.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure saved: outputs/fig_10_tabnet_cv_stability.png")

# ─────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────
print("\n=== Script 04_tabnet complete. All outputs saved to outputs/ ===")
print(f"\nImbalance strategy applied: {tabnet_strategy}")
print(f"\nFinal TABnet val-set performance:")
print(f"  Val PR-AUC  : {tabnet_val_ap:.4f}  (lift = {tabnet_val_ap/NO_SKILL_PRAUC:.2f}x)")
print(f"  Val ROC-AUC : {tabnet_val_roc:.4f}")
print(f"\nCV PR-AUC: {np.mean(tabnet_cv_scores):.4f} +/- {np.std(tabnet_cv_scores):.4f}")
print(f"\nBest hyperparameters:")
for k, v in best_tabnet_params.items():
    print(f"  {k}: {v}")
print(f"\n  No-skill PR-AUC baseline: {NO_SKILL_PRAUC}")
