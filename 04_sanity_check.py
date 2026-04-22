import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.base import clone
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score
from joblib import Parallel, delayed
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings('ignore')

def _score_fold(estimator, params, X, y, tr_idx, va_idx):
    est = clone(estimator).set_params(**params)
    est.fit(X[tr_idx], y[tr_idx])
    return average_precision_score(y[va_idx], est.predict_proba(X[va_idx])[:, 1])

def tqdm_random_search(estimator, param_dist, X, y, cv_folds, n_iter, random_state, desc):
    param_list = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=random_state))
    all_results = []
    best_score = -np.inf
    for params in tqdm(param_list, desc=desc, unit='combo'):
        fold_scores = Parallel(n_jobs=-1)(
            delayed(_score_fold)(estimator, params, X, y, tr_idx, va_idx)
            for tr_idx, va_idx in cv_folds
        )
        mean_score = float(np.mean(fold_scores))
        all_results.append({'params': params, 'mean_score': mean_score, 'fold_scores': fold_scores})
        if mean_score > best_score:
            best_score = mean_score
            best_result = all_results[-1]
    return best_result, all_results

def find_output(filename):
    matches = list(Path('outputs').rglob(filename))
    if not matches:
        raise FileNotFoundError(f"'{filename}' not found anywhere under outputs/")
    return str(sorted(matches, key=lambda p: p.stat().st_mtime)[-1])

RANDOM_STATE = 42
DATA_FILE    = 'data_final_modeling_ma_v7.xlsx'
TARGET       = 'target_next_year'
FEATURES     = [
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
NO_SKILL_PRAUC = 0.0157

print("=" * 60)
print("SANITY CHECK — minimal run (2 combos x 2 folds)")
print("=" * 60)

# --- Data loading ---
strategy_path = find_output('selected_imbalance_strategy.txt')
print(f"[OK] Strategy file found: {strategy_path}")

df = pd.read_excel(DATA_FILE)
df['fiscal_year'] = pd.to_datetime(df['datadate']).dt.year

df_cv    = df[df['fiscal_year'].between(2012, 2022)].copy()
X_cv     = df_cv[FEATURES].values
y_cv     = df_cv[TARGET].values
years_cv = df_cv['fiscal_year'].values

# Only 2 folds (last 2 of the full expanding-window set)
FOLD_SPECS_MINI = [
    (list(range(2012, 2021)), 2021),
    (list(range(2012, 2022)), 2022),
]
cv_folds_mini = []
for train_yrs, val_yr in FOLD_SPECS_MINI:
    tr_idx = np.where(np.isin(years_cv, train_yrs))[0]
    va_idx = np.where(years_cv == val_yr)[0]
    cv_folds_mini.append((tr_idx, va_idx))
print(f"[OK] Data loaded: {len(y_cv):,} obs, {y_cv.sum()} pos | {len(cv_folds_mini)} mini folds")

scale_pos_weight = int((y_cv == 0).sum()) / int((y_cv == 1).sum())

# --- LR: just 1 fold ---
print("\n[LR] Running 1 fold ...")
tr_idx, va_idx = cv_folds_mini[0]
sc = RobustScaler()
Xtr_s = sc.fit_transform(X_cv[tr_idx])
Xva_s = sc.transform(X_cv[va_idx])
lr = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0,
                        class_weight='balanced', random_state=RANDOM_STATE)
lr.fit(Xtr_s, y_cv[tr_idx])
ap = average_precision_score(y_cv[va_idx], lr.predict_proba(Xva_s)[:, 1])
print(f"[OK] LR fold PR-AUC = {ap:.4f}  (lift = {ap/NO_SKILL_PRAUC:.2f}x)")

# --- XGBoost: 2 combos x 2 folds ---
print("\n[XGB] tqdm_random_search (n_iter=2, 2 folds) ...")
xgb_param_dist = {
    'n_estimators':     [300, 500],
    'max_depth':        [3, 4],
    'learning_rate':    [0.05],
    'subsample':        [0.8],
    'colsample_bytree': [0.8],
    'min_child_weight': [10],
    'gamma':            [0],
}
xgb_base = XGBClassifier(scale_pos_weight=scale_pos_weight,
                          eval_metric='logloss', random_state=RANDOM_STATE,
                          n_jobs=1, verbosity=0)
best_xgb, _ = tqdm_random_search(xgb_base, xgb_param_dist, X_cv, y_cv, cv_folds_mini,
                                  n_iter=2, random_state=RANDOM_STATE, desc='XGBoost CV')
print(f"[OK] XGBoost best mean CV PR-AUC = {best_xgb['mean_score']:.4f}  "
      f"params = {best_xgb['params']}")

# --- MLP: 2 combos x 2 folds ---
print("\n[MLP] tqdm_random_search (n_iter=2, 2 folds) ...")
mlp_pipe = ImbPipeline([
    ('resampler', RandomOverSampler(sampling_strategy=0.20, random_state=RANDOM_STATE)),
    ('scaler',    RobustScaler()),
    ('mlp',       MLPClassifier(solver='adam', random_state=RANDOM_STATE)),
])
mlp_param_dist = {
    'mlp__hidden_layer_sizes': [(64,), (128,)],
    'mlp__activation':         ['relu'],
    'mlp__alpha':              [1e-3],
    'mlp__learning_rate_init': [1e-3],
    'mlp__batch_size':         [128],
    'mlp__max_iter':           [200],
    'mlp__early_stopping':     [True],
    'mlp__validation_fraction':[0.1],
    'mlp__n_iter_no_change':   [15],
}
best_mlp, _ = tqdm_random_search(mlp_pipe, mlp_param_dist, X_cv, y_cv, cv_folds_mini,
                                  n_iter=2, random_state=RANDOM_STATE, desc='MLP CV')
print(f"[OK] MLP best mean CV PR-AUC = {best_mlp['mean_score']:.4f}  "
      f"params = {best_mlp['params']}")

print("\n" + "=" * 60)
print("SANITY CHECK PASSED — all steps executed without errors.")
print("You can now run the full 04_model_training.py")
print("=" * 60)
