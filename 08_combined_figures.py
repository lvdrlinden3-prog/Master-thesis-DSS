import os
import json
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score,
    confusion_matrix, brier_score_loss,
    balanced_accuracy_score,
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')

OUTPUT_DIR = 'output_combined'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# TabNetWrapper — must match 04_model_training_tabnet.py exactly for pickle
# ─────────────────────────────────────────────
class TabNetWrapper:
    _estimator_type = "classifier"
    classes_        = np.array([0, 1])

    def __init__(self, scaler, model):
        self.scaler = scaler
        self.model  = model

    def fit(self, *args, **kwargs):
        return self

    def predict_proba(self, X):
        X_sc = self.scaler.transform(np.asarray(X)).astype(np.float32)
        return self.model.predict_proba(X_sc)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
NO_SKILL_PRAUC = 0.0157
RANDOM_STATE   = 42
DATA_FILE      = 'data_final_modeling_ma_v7.xlsx'
TARGET         = 'target_next_year'
FEATURES = [
    'profitability', 'leverage', 'cash_ratio', 'fcf_debt',
    'ppe_ratio', 'capex_intensity', 'asset_turnover',
    'interest_burden', 'net_margin', 'rev_growth', 'fcf_volatility',
    'firm_size', 'rd_intensity', 'rev_growth_lag1', 'altman_re_ta',
]
COLORS = {
    'Logistic Regression': '#2c7bb6',
    'XGBoost':             '#d7191c',
    'MLP':                 '#1a9641',
    'TABnet':              '#ff7f00',
}
MARKERS = {
    'Logistic Regression': 'o',
    'XGBoost':             's',
    'MLP':                 '^',
    'TABnet':              'D',
}
MODEL_ORDER    = ['Logistic Regression', 'XGBoost', 'MLP', 'TABnet']
THRESHOLD_GRID = np.arange(0.001, 0.151, 0.001)
N_BOOT         = 2000

THEORY_DIRECTIONS = {
    'profitability':    '(-)',
    'leverage':         '(+/-)',
    'cash_ratio':       '(-)',
    'fcf_debt':         '(+/-)',
    'ppe_ratio':        '(+)',
    'capex_intensity':  '(-)',
    'asset_turnover':   '(-)',
    'interest_burden':  '(+)',
    'net_margin':       '(-)',
    'rev_growth':       '(-)',
    'fcf_volatility':   '(+)',
    'firm_size':        '(-)',
    'rd_intensity':     '(+)',
    'rev_growth_lag1':  '(-)',
    'altman_re_ta':     '(-)',
}

FOLD_SPECS = [
    (list(range(2012, 2017)), 2017),
    (list(range(2012, 2018)), 2018),
    (list(range(2012, 2019)), 2019),
    (list(range(2012, 2020)), 2020),
    (list(range(2012, 2021)), 2021),
    (list(range(2012, 2022)), 2022),
]


def find_output(filename):
    """Locate a file anywhere under outputs/."""
    matches = list(Path('outputs').rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"'{filename}' not found under outputs/. Run the prerequisite scripts first."
        )
    return str(sorted(matches, key=lambda p: p.stat().st_mtime)[-1])


def out(filename):
    """Return full path in output_combined/."""
    return os.path.join(OUTPUT_DIR, filename)


def sic_to_division(sic):
    try:
        sic = int(sic)
    except (ValueError, TypeError):
        return 'Unknown'
    if   100  <= sic <=  999: return 'Agriculture'
    elif 1000 <= sic <= 1499: return 'Mining'
    elif 1500 <= sic <= 1999: return 'Construction'
    elif 2000 <= sic <= 3999: return 'Manufacturing'
    elif 4000 <= sic <= 4999: return 'Transportation & Utilities'
    elif 5000 <= sic <= 5199: return 'Wholesale Trade'
    elif 5200 <= sic <= 5999: return 'Retail Trade'
    elif 6000 <= sic <= 6999: return 'Finance & Insurance'
    elif 7000 <= sic <= 8999: return 'Services'
    elif 9000 <= sic <= 9999: return 'Public Administration'
    else:                      return 'Unknown'


def safe_round(v, decimals):
    """Round v to decimals; return np.nan if v is NaN or None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    try:
        return round(float(v), decimals)
    except (TypeError, ValueError):
        return np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — SCRIPT 04 EQUIVALENT: CV Stability (Table 09, Figure 10)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 1 — CV Stability  (Table 09 · Figure 10)")
print("=" * 60)

# Load per-fold CV PR-AUC scores from the JSON metrics files written by Scripts 04 / 04_tabnet
cv_scores = {}

try:
    with open(find_output('metrics_04_model_training.json')) as _f:
        m04 = json.load(_f)
    cv_scores['Logistic Regression'] = m04['logistic_regression']['cv_pr_auc_per_fold']
    cv_scores['XGBoost']             = m04['xgboost']['cv_pr_auc_per_fold']
    cv_scores['MLP']                 = m04['mlp']['cv_pr_auc_per_fold']
    print("  LR / XGBoost / MLP CV scores loaded from metrics_04_model_training.json")
except Exception as e:
    print(f"  WARNING — could not load metrics_04_model_training.json: {e}")

try:
    with open(find_output('metrics_04_tabnet_model_training.json')) as _f:
        m04t = json.load(_f)
    cv_scores['TABnet'] = m04t['tabnet']['cv_pr_auc_per_fold']
    print("  TABnet CV scores loaded from metrics_04_tabnet_model_training.json")
except Exception as e:
    print(f"  WARNING — could not load metrics_04_tabnet_model_training.json: {e}")

# Table 09 — combined CV performance (all 4 models)
rows = []
for model_name in MODEL_ORDER:
    if model_name not in cv_scores:
        continue
    for fold_i, score in enumerate(cv_scores[model_name], 1):
        train_yrs, val_yr = FOLD_SPECS[fold_i - 1]
        rows.append({
            'Model':           model_name,
            'Fold':            fold_i,
            'Train_Years':     f"{train_yrs[0]}–{train_yrs[-1]}",
            'Val_Year':        val_yr,
            'Val_PR_AUC':      round(score, 4),
            'Lift_vs_NoSkill': round(score / NO_SKILL_PRAUC, 3),
        })

cv_df = pd.DataFrame(rows)
cv_df.to_csv(out('table_09_cv_performance.csv'), index=False)
print(f"  Table 09 saved → {out('table_09_cv_performance.csv')}")

print("\n  CV summary (mean ± std):")
print(f"  {'Model':<25} Mean PR-AUC   Std    Lift vs no-skill")
print("  " + "-" * 55)
for model_name in MODEL_ORDER:
    if model_name not in cv_scores:
        continue
    s = cv_scores[model_name]
    print(f"  {model_name:<25} {np.mean(s):.4f}       "
          f"{np.std(s):.4f}  {np.mean(s)/NO_SKILL_PRAUC:.2f}x")

# Figure 10 — CV stability, all 4 models, equal treatment
fold_labels = [
    '2012-16→2017\n(F1)', '2012-17→2018\n(F2)', '2012-18→2019\n(F3)',
    '2012-19→2020\n(F4)', '2012-20→2021\n(F5)', '2012-21→2022\n(F6)',
]

fig, ax = plt.subplots(figsize=(14, 5))
for model_name in MODEL_ORDER:
    if model_name not in cv_scores:
        continue
    scores = cv_scores[model_name]
    mean_s = np.mean(scores)
    ax.plot(range(1, 7), scores,
            color=COLORS[model_name], marker=MARKERS[model_name],
            lw=2, ms=8, zorder=3,
            label=f'{model_name} (CV mean = {mean_s:.4f}, '
                  f'lift = {mean_s/NO_SKILL_PRAUC:.2f}x)')

ax.axhline(y=NO_SKILL_PRAUC, color='gray', linestyle='--', lw=1.5,
           label=f'No-skill baseline (AP = {NO_SKILL_PRAUC})')
ax.set_xticks(range(1, 7))
ax.set_xticklabels(fold_labels, fontsize=8)
ax.set_ylabel('Val PR-AUC (Average Precision)')
ax.set_xlabel('Expanding-Window CV Fold')
ax.set_title(
    'Figure 10: Expanding-Window CV Stability — PR-AUC per Fold\n'
    'Logistic Regression  ·  XGBoost  ·  MLP  ·  TABnet',
    fontsize=11
)
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig(out('fig_10_cv_stability.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Figure 10 saved → {out('fig_10_cv_stability.png')}")


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA AND MODELS (shared by Parts 2–4)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Loading data and model artifacts")
print("=" * 60)

df = pd.read_excel(DATA_FILE)
df['fiscal_year']  = pd.to_datetime(df['datadate']).dt.year
df['sic_division'] = df['sic'].apply(sic_to_division)

df_train = df[df['fiscal_year'].between(2012, 2021)].copy()
df_val   = df[df['fiscal_year'] == 2022].copy().reset_index(drop=True)
df_test  = df[df['fiscal_year'].between(2023, 2024)].copy().reset_index(drop=True)

X_train = df_train[FEATURES].values;  y_train = df_train[TARGET].values
X_val   = df_val[FEATURES].values;    y_val   = df_val[TARGET].values
X_test  = df_test[FEATURES].values;   y_test  = df_test[TARGET].values

print(f"  Train (2012–2021): {len(y_train):,} obs | {y_train.sum()} pos")
print(f"  Val  (2022)      : {len(y_val):,} obs  | {y_val.sum()} pos")
print(f"  Test (2023–2024) : {len(y_test):,} obs  | {y_test.sum()} pos")

lr_path     = find_output('model_baseline_logistic.pkl')
xgb_path    = find_output('model_xgb.pkl')
mlp_path    = find_output('model_mlp_pipeline.pkl')
tabnet_path = find_output('model_tabnet.pkl')

with open(lr_path,     'rb') as f: lr_pipeline     = pickle.load(f)
with open(xgb_path,    'rb') as f: model_xgb       = pickle.load(f)
with open(mlp_path,    'rb') as f: mlp_pipeline    = pickle.load(f)
with open(tabnet_path, 'rb') as f: tabnet_pipeline = pickle.load(f)

print(f"  Loaded: {lr_path}")
print(f"  Loaded: {xgb_path}")
print(f"  Loaded: {mlp_path}")
print(f"  Loaded: {tabnet_path}")

models = {
    'Logistic Regression': lr_pipeline,
    'XGBoost':             model_xgb,
    'MLP':                 mlp_pipeline,
    'TABnet':              tabnet_pipeline,
}

val_probas  = {name: m.predict_proba(X_val)[:,  1] for name, m in models.items()}
test_probas = {name: m.predict_proba(X_test)[:, 1] for name, m in models.items()}

# Threshold per model: argmax F1 on val set
optimal_thresholds = {}
for name in MODEL_ORDER:
    f1_v = [f1_score(y_val, (val_probas[name] >= t).astype(int), zero_division=0)
            for t in THRESHOLD_GRID]
    optimal_thresholds[name] = float(THRESHOLD_GRID[np.argmax(f1_v)])
print("  Optimal thresholds (val argmax F1):",
      {k: round(v, 3) for k, v in optimal_thresholds.items()})


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — SCRIPT 05 EQUIVALENT: Evaluation (Tables 10–11, Figures 11–13)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 2 — Model Evaluation  (Tables 10–11 · Figures 11–13)")
print("=" * 60)

# ── Table 10 — test metrics, all 4 models ────────────────────────────────────
N_test     = len(y_test)
n_pos_test = int(y_test.sum())
top5_n     = max(1, int(np.round(0.05 * N_test)))
top10_n    = max(1, int(np.round(0.10 * N_test)))

print(f"\n  Test set: {N_test:,} obs | {n_pos_test} pos | "
      f"base rate = {n_pos_test/N_test*100:.2f}%")
print(f"  Top 5%  = {top5_n} firms | Top 10% = {top10_n} firms")

metric_rows = []
for name in MODEL_ORDER:
    proba  = test_probas[name]
    thresh = optimal_thresholds[name]
    preds  = (proba >= thresh).astype(int)

    auc_roc = roc_auc_score(y_test, proba)
    pr_auc  = average_precision_score(y_test, proba)
    f1      = f1_score(y_test, preds, zero_division=0)
    prec    = precision_score(y_test, preds, zero_division=0)
    rec     = recall_score(y_test, preds, zero_division=0)
    brier   = brier_score_loss(y_test, proba)
    bal_acc = balanced_accuracy_score(y_test, preds)
    tn, fp_v, fn_v, tp_v = confusion_matrix(y_test, preds).ravel()
    specificity = tn / (tn + fp_v) if (tn + fp_v) > 0 else 0.0

    sorted_idx = np.argsort(proba)[::-1]
    pos_top5   = int(y_test[sorted_idx[:top5_n]].sum())
    pos_top10  = int(y_test[sorted_idx[:top10_n]].sum())
    lift5  = round((pos_top5  / top5_n)  / NO_SKILL_PRAUC, 3)
    lift10 = round((pos_top10 / top10_n) / NO_SKILL_PRAUC, 3)

    metric_rows.append({
        'Model':             name,
        'Threshold':         round(thresh, 3),
        'AUC_ROC':           round(auc_roc, 4),
        'PR_AUC':            round(pr_auc,  4),
        'Lift_vs_NoSkill':   round(pr_auc / NO_SKILL_PRAUC, 3),
        'F1':                round(f1,      4),
        'Precision':         round(prec,    4),
        'Recall':            round(rec,     4),
        'Specificity':       round(specificity, 4),
        'Balanced_Accuracy': round(bal_acc, 4),
        'Brier_Score':       round(brier,   4),
        'Lift_at_5pct':      lift5,
        'Lift_at_10pct':     lift10,
        'TP': tp_v, 'FP': fp_v, 'FN': fn_v, 'TN': tn,
        'Pos_in_Top5pct':    pos_top5,
        'Pos_in_Top10pct':   pos_top10,
    })

metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv(out('table_10_test_metrics.csv'), index=False)
print(f"  Table 10 saved → {out('table_10_test_metrics.csv')}")

best_model = metrics_df.loc[metrics_df['PR_AUC'].idxmax(), 'Model']
best_prauc = metrics_df['PR_AUC'].max()
print(f"\n  Best model (highest Test PR-AUC): {best_model} ({best_prauc:.4f})")

print(f"\n  {'Model':<25} ROC-AUC  PR-AUC  Lift   F1      Prec    Rec")
print("  " + "-" * 70)
for row in metric_rows:
    print(f"  {row['Model']:<25} {row['AUC_ROC']:.4f}   {row['PR_AUC']:.4f}  "
          f"{row['Lift_vs_NoSkill']:.2f}x  {row['F1']:.4f}  "
          f"{row['Precision']:.4f}  {row['Recall']:.4f}")

# ── Table 11 — bootstrap statistical comparison, all 6 pairwise pairs ────────
rng = np.random.default_rng(RANDOM_STATE)

def bootstrap_compare(y, proba_a, proba_b, n_boot, rng, metric='roc'):
    score_fn = roc_auc_score if metric == 'roc' else average_precision_score
    obs_a    = score_fn(y, proba_a)
    obs_b    = score_fn(y, proba_b)
    obs_diff = obs_a - obs_b
    boot_diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        if y[idx].sum() < 2 or (y[idx] == 0).sum() < 2:
            continue
        d = score_fn(y[idx], proba_a[idx]) - score_fn(y[idx], proba_b[idx])
        boot_diffs.append(d)
    boot_diffs = np.array(boot_diffs)
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    p = float(2 * np.mean(boot_diffs < 0) if obs_diff >= 0
              else 2 * np.mean(boot_diffs > 0))
    return obs_a, obs_b, obs_diff, ci_lo, ci_hi, min(p, 1.0)

PAIRS = [
    ('XGBoost',  'Logistic Regression'),
    ('MLP',      'Logistic Regression'),
    ('TABnet',   'Logistic Regression'),
    ('XGBoost',  'MLP'),
    ('TABnet',   'XGBoost'),
    ('TABnet',   'MLP'),
]

stat_rows = []
print(f"\n  Running bootstrap comparisons (n={N_BOOT}) ...")
for name_a, name_b in PAIRS:
    for metric_label, metric_key in [('ROC-AUC', 'roc'), ('PR-AUC', 'pr')]:
        obs_a, obs_b, delta, ci_lo, ci_hi, p = bootstrap_compare(
            y_test, test_probas[name_a], test_probas[name_b],
            N_BOOT, rng, metric=metric_key
        )
        if p < 0.05:
            interp = "statistically significant"
        elif abs(delta) >= 0.03:
            interp = "practically sig., statistically underpowered"
        else:
            interp = "not significant"
        stat_rows.append({
            'Model_A':         name_a,
            'Model_B':         name_b,
            'Metric':          metric_label,
            'Score_A':         round(obs_a,  4),
            'Score_B':         round(obs_b,  4),
            'Delta_A_minus_B': round(delta,  4),
            'CI_lower_95':     round(ci_lo,  4),
            'CI_upper_95':     round(ci_hi,  4),
            'p_value':         round(p,      4),
            'n_bootstrap':     N_BOOT,
            'Interpretation':  interp,
        })

stat_df = pd.DataFrame(stat_rows)
stat_df.to_csv(out('table_11_statistical_tests.csv'), index=False)
print(f"  Table 11 saved → {out('table_11_statistical_tests.csv')}")

# ── Figure 11 — ROC + PR curves, all 4 models ────────────────────────────────
fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))

for name in MODEL_ORDER:
    proba = test_probas[name]
    color = COLORS[name]
    auc   = roc_auc_score(y_test, proba)
    ap    = average_precision_score(y_test, proba)

    fpr, tpr, _ = roc_curve(y_test, proba)
    ax_roc.plot(fpr, tpr, color=color, lw=2,
                label=f'{name} (AUC = {auc:.4f})')

    prec_c, rec_c, _ = precision_recall_curve(y_test, proba)
    ax_pr.plot(rec_c, prec_c, color=color, lw=2,
               label=f'{name} (AP = {ap:.4f}, lift = {ap/NO_SKILL_PRAUC:.2f}x)')

ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, label='No-skill (AUC = 0.50)')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('Figure 11a: ROC Curves — Test Set (2023–2024)', fontsize=11)
ax_roc.legend(fontsize=9, loc='lower right')
ax_roc.grid(alpha=0.3)

ax_pr.axhline(y=NO_SKILL_PRAUC, color='gray', linestyle='--', lw=1.5,
              label=f'No-skill baseline (prevalence = 1.57%)')
ax_pr.set_xlabel('Recall')
ax_pr.set_ylabel('Precision (log scale)')
ax_pr.set_yscale('log')
ax_pr.set_title('Figure 11b: Precision-Recall Curves — Test Set (2023–2024)\n'
                '(log y-scale; base rate = 1.57%)', fontsize=11)
ax_pr.legend(fontsize=9, loc='upper right')
ax_pr.grid(alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(out('fig_11_roc_pr_curves.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"\n  Figure 11 saved → {out('fig_11_roc_pr_curves.png')}")

# ── Figure 12 — Confusion matrices (1 × 4) ───────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, name in zip(axes, MODEL_ORDER):
    proba  = test_probas[name]
    thresh = optimal_thresholds[name]
    preds  = (proba >= thresh).astype(int)
    cm     = confusion_matrix(y_test, preds)

    ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(f'{name}\n(threshold = {thresh:.3f})', fontsize=10)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Neg (0)', 'Pos (1)'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Neg (0)', 'Pos (1)'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=14, fontweight='bold')

plt.suptitle('Figure 12: Confusion Matrices — Test Set (2023–2024)\n'
             'Threshold = argmax(F1) on validation set', fontsize=12)
plt.tight_layout()
plt.savefig(out('fig_12_confusion_matrices.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Figure 12 saved → {out('fig_12_confusion_matrices.png')}")

# ── Figure 13 — Cumulative lift chart, all 4 models ──────────────────────────
pct_grid    = np.arange(1, 101)
lift_curves = {}

for name in MODEL_ORDER:
    proba      = test_probas[name]
    sorted_idx = np.argsort(proba)[::-1]
    cumpos     = np.cumsum(y_test[sorted_idx])
    lifts = []
    for pct in pct_grid:
        n_top    = max(1, int(np.round(pct / 100 * N_test)))
        expected = n_top * NO_SKILL_PRAUC
        lifts.append(cumpos[n_top - 1] / expected if expected > 0 else 0)
    lift_curves[name] = np.array(lifts)

fig, ax = plt.subplots(figsize=(10, 6))
for name in MODEL_ORDER:
    ax.plot(pct_grid, lift_curves[name], color=COLORS[name], lw=2, label=name)

ax.axhline(y=1.0, color='gray', linestyle='--', lw=1.5,
           label='Random screening (lift = 1.0x)')
ax.axvline(x=10, color='black', linestyle=':', lw=1, alpha=0.6)

for name in MODEL_ORDER:
    lift_at10 = lift_curves[name][9]
    ax.annotate(f'{name.split()[0]}: {lift_at10:.1f}x',
                xy=(10, lift_at10), fontsize=8.5, ha='left',
                xytext=(12, lift_at10))

ax.set_xlabel('% of population screened (ranked by predicted probability)')
ax.set_ylabel('Cumulative Lift (= precision@k% / no-skill baseline)')
ax.set_title(
    f'Figure 13: Cumulative Lift Chart — Test Set (2023–2024)\n'
    f'No-skill baseline = {NO_SKILL_PRAUC} (prevalence = 1.57%)',
    fontsize=11
)
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)
ax.set_xlim(0, 100)
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig(out('fig_13_lift_chart.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Figure 13 saved → {out('fig_13_lift_chart.png')}")

print(f"\n  Lift@10% summary:")
for name in MODEL_ORDER:
    print(f"    {name:<25} {lift_curves[name][9]:.2f}x")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — SCRIPT 06 EQUIVALENT: Feature Importance (Table 12, Figures 14–16)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 3 — Feature Importance  (Table 12 · Figures 14–16)")
print("=" * 60)

# Scaled training/test data for LR LinearExplainer
lr_scaler      = lr_pipeline.named_steps['scaler']
lr_model_obj   = lr_pipeline.named_steps['lr']
X_train_scaled = lr_scaler.transform(X_train)
X_test_scaled  = lr_scaler.transform(X_test)

# ── XGBoost SHAP (TreeExplainer) ─────────────────────────────────────────────
print("\n  Computing XGBoost SHAP values ...")
explainer_xgb   = shap.TreeExplainer(model_xgb)
shap_values_xgb = explainer_xgb.shap_values(X_test)
mean_abs_shap_xgb = np.abs(shap_values_xgb).mean(axis=0)
xgb_rank          = np.argsort(mean_abs_shap_xgb)[::-1]

# ── Figure 14 — XGBoost SHAP beeswarm (XGBoost-specific, unchanged) ──────────
print("  Generating Figure 14: XGBoost SHAP beeswarm ...")
shap_expl = shap.Explanation(
    values=shap_values_xgb,
    base_values=np.full(len(X_test), explainer_xgb.expected_value),
    data=X_test,
    feature_names=FEATURES,
)
plt.figure(figsize=(9, 7))
shap.plots.beeswarm(shap_expl, max_display=15, show=False, plot_size=None)
fig14 = plt.gcf()
fig14.suptitle(
    f'Figure 14: XGBoost SHAP Beeswarm — Test Set (2023–2024)\n'
    f'n={len(X_test):,} obs | colour = feature value (red=high, blue=low)',
    fontsize=10, y=1.01
)
plt.tight_layout()
fig14.savefig(out('fig_14_shap_beeswarm_xgb.png'), dpi=300, bbox_inches='tight')
plt.close('all')
print(f"  Figure 14 saved → {out('fig_14_shap_beeswarm_xgb.png')}")

# ── LR SHAP (LinearExplainer) ─────────────────────────────────────────────────
print("  Computing LR SHAP values ...")
explainer_lr     = shap.LinearExplainer(lr_model_obj, X_train_scaled, nsamples=500)
shap_values_lr   = explainer_lr.shap_values(X_test_scaled)
mean_abs_shap_lr = np.abs(shap_values_lr).mean(axis=0)
lr_rank          = np.argsort(mean_abs_shap_lr)[::-1]

# ── MLP permutation importance ────────────────────────────────────────────────
print("  Computing MLP permutation importance (n_repeats=50) ...")
perm_mlp      = permutation_importance(
    mlp_pipeline, X_test, y_test,
    scoring='average_precision', n_repeats=50,
    random_state=RANDOM_STATE, n_jobs=-1,
)
mean_perm_mlp = perm_mlp.importances_mean
std_perm_mlp  = perm_mlp.importances_std
mlp_rank      = np.argsort(mean_perm_mlp)[::-1]
mlp_imp_norm  = np.clip(mean_perm_mlp, 0, None)
mlp_imp_norm  = (mlp_imp_norm / mlp_imp_norm.max()
                 if mlp_imp_norm.max() > 0 else mlp_imp_norm)

# ── TABnet attention importance ───────────────────────────────────────────────
print("  Computing TABnet attention importance ...")
tabnet_model_inner  = tabnet_pipeline.model
tabnet_scaler_inner = tabnet_pipeline.scaler
X_test_sc = tabnet_scaler_inner.transform(X_test).astype(np.float32)
M_explain, _ = tabnet_model_inner.explain(X_test_sc)
tabnet_attention = M_explain.mean(axis=0)

if hasattr(tabnet_model_inner, 'feature_importances_'):
    tabnet_main_imp = tabnet_model_inner.feature_importances_
else:
    tabnet_main_imp = tabnet_attention / tabnet_attention.sum()

tabnet_main_rank = np.argsort(tabnet_main_imp)[::-1]

# Normalise all to [0, 1]
xgb_norm         = mean_abs_shap_xgb / mean_abs_shap_xgb.max()
lr_norm          = mean_abs_shap_lr  / mean_abs_shap_lr.max()
tabnet_main_norm = (tabnet_main_imp / tabnet_main_imp.max()
                    if tabnet_main_imp.max() > 0 else tabnet_main_imp)

# ── Table 12 — unified 4-model feature importance ────────────────────────────
try:
    coef_csv = pd.read_csv(find_output('table_07_logistic_coefficients.csv'))
    coef_map = dict(zip(coef_csv['Feature'], coef_csv['Coefficient']))
    tc_map   = dict(zip(coef_csv['Feature'], coef_csv['Theory_Consistent']))
except Exception:
    coef_map = {f: np.nan for f in FEATURES}
    tc_map   = {f: '?' for f in FEATURES}

pos_idx = np.where(y_test == 1)[0]
imp_rows = []
for i, feat in enumerate(FEATURES):
    xgb_sign = '+' if np.mean(shap_values_xgb[pos_idx, i]) >= 0 else '-'
    lr_coef  = coef_map.get(feat, np.nan)
    imp_rows.append({
        'Feature':              feat,
        'Theory_Direction':     THEORY_DIRECTIONS[feat],
        'LR_Coefficient':       safe_round(lr_coef, 4),
        'Theory_Consistent_LR': tc_map.get(feat, '?'),
        'XGBoost_MeanAbsSHAP':  round(mean_abs_shap_xgb[i], 6),
        'XGBoost_Rank':         int(np.where(xgb_rank == i)[0][0]) + 1,
        'XGBoost_SHAP_Sign':    xgb_sign,
        'XGBoost_Norm':         round(xgb_norm[i], 4),
        'LR_MeanAbsSHAP':       round(mean_abs_shap_lr[i], 6),
        'LR_Rank':              int(np.where(lr_rank == i)[0][0]) + 1,
        'LR_Norm':              round(lr_norm[i], 4),
        'MLP_PermImp_Mean':     round(mean_perm_mlp[i], 6),
        'MLP_PermImp_Std':      round(std_perm_mlp[i],  6),
        'MLP_Rank':             int(np.where(mlp_rank == i)[0][0]) + 1,
        'MLP_Norm':             round(mlp_imp_norm[i], 4),
        'TABnet_AttentionImp':  round(float(tabnet_main_imp[i]), 6),
        'TABnet_Rank':          int(np.where(tabnet_main_rank == i)[0][0]) + 1,
        'TABnet_Norm':          round(float(tabnet_main_norm[i]), 4),
    })

imp_df = pd.DataFrame(imp_rows).sort_values('XGBoost_Rank').reset_index(drop=True)
imp_df.to_csv(out('table_12_feature_importance.csv'), index=False)
print(f"  Table 12 saved → {out('table_12_feature_importance.csv')}")

# ── Figure 15 — 4-model feature importance comparison ────────────────────────
feat_order = imp_df['Feature'].tolist()
n_feats    = len(feat_order)
y_pos      = np.arange(n_feats)
bar_h      = 0.20

def get_norm(feat, col):
    v = imp_df.loc[imp_df['Feature'] == feat, col].values
    return float(v[0]) if len(v) > 0 and pd.notna(v[0]) else 0.0

xgb_vals = [get_norm(f, 'XGBoost_Norm') for f in feat_order]
lr_vals  = [get_norm(f, 'LR_Norm')      for f in feat_order]
mlp_vals = [get_norm(f, 'MLP_Norm')     for f in feat_order]
tab_vals = [get_norm(f, 'TABnet_Norm')  for f in feat_order]

fig15, ax15 = plt.subplots(figsize=(11, 9))
ax15.barh(y_pos + 1.5*bar_h, xgb_vals, height=bar_h, color=COLORS['XGBoost'],
          alpha=0.85, label='XGBoost (mean |SHAP|, normalised)')
ax15.barh(y_pos + 0.5*bar_h, lr_vals,  height=bar_h, color=COLORS['Logistic Regression'],
          alpha=0.85, label='Logistic Regression (mean |SHAP|, normalised)')
ax15.barh(y_pos - 0.5*bar_h, mlp_vals, height=bar_h, color=COLORS['MLP'],
          alpha=0.85, label='MLP (permutation importance, normalised)')
ax15.barh(y_pos - 1.5*bar_h, tab_vals, height=bar_h, color=COLORS['TABnet'],
          alpha=0.85, label='TABnet (attention importance, normalised)')

ax15.set_yticks(y_pos)
ax15.set_yticklabels(feat_order, fontsize=9)
ax15.invert_yaxis()
ax15.set_xlabel('Normalised Feature Importance (1.0 = most important for each model)')
ax15.set_title(
    'Figure 15: Feature Importance Comparison — All Four Models\n'
    'Sorted by XGBoost SHAP rank | Test Set (2023–2024)',
    fontsize=11
)
ax15.legend(fontsize=8, loc='lower right')
ax15.axvline(x=0, color='black', lw=0.6)
ax15.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(out('fig_15_feature_importance_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Figure 15 saved → {out('fig_15_feature_importance_comparison.png')}")

# ── Figure 16 — SHAP dependence plots, top 4 XGBoost features (unchanged) ────
top4_features = [FEATURES[i] for i in xgb_rank[:4]]
print(f"  Generating Figure 16: SHAP dependence (top 4 XGB features: {top4_features}) ...")

fig16, axes16 = plt.subplots(2, 2, figsize=(12, 9))
axes16 = axes16.flatten()
for ax, feat in zip(axes16, top4_features):
    feat_idx = FEATURES.index(feat)
    shap.dependence_plot(
        feat_idx, shap_values_xgb, X_test,
        feature_names=FEATURES, ax=ax, show=False,
        dot_size=8, alpha=0.5,
    )
    ax.set_title(f'{feat}\n(Theory: {THEORY_DIRECTIONS[feat]})', fontsize=10)
    ax.grid(alpha=0.3)

fig16.suptitle(
    'Figure 16: XGBoost SHAP Dependence Plots — Top 4 Features\n'
    'Test Set (2023–2024) | Colour = interaction feature (auto-selected by SHAP)',
    fontsize=11
)
plt.tight_layout()
plt.savefig(out('fig_16_shap_dependence.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Figure 16 saved → {out('fig_16_shap_dependence.png')}")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4 — SCRIPT 07 EQUIVALENT: Error Analysis (Tables 13–15, Figures 17–20)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 4 — Error Analysis  (Tables 13–15 · Figures 17–20)")
print("=" * 60)

# Best-model threshold (argmax F1 on val set — same protocol as Part 2)
best_thresh    = optimal_thresholds[best_model]
proba_best     = test_probas[best_model]
preds_best     = (proba_best >= best_thresh).astype(int)
tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, preds_best).ravel()
print(f"\n  Best model: {best_model} | threshold = {best_thresh:.3f}")
print(f"  TP={tp_b}  FP={fp_b}  FN={fn_b}  TN={tn_b}")

# Annotate test frame with predictions from best model
df_test['proba_best'] = proba_best
df_test['pred_best']  = preds_best
df_test['error_type'] = 'TN'
df_test.loc[(df_test['pred_best'] == 1) & (df_test[TARGET] == 1), 'error_type'] = 'TP'
df_test.loc[(df_test['pred_best'] == 1) & (df_test[TARGET] == 0), 'error_type'] = 'FP'
df_test.loc[(df_test['pred_best'] == 0) & (df_test[TARGET] == 1), 'error_type'] = 'FN'

# ── Table 13 — sector-level analysis ─────────────────────────────────────────
sector_rows = []
for div, grp in df_test.groupby('sic_division', sort=False):
    n_firms   = len(grp)
    n_pos     = int(grp[TARGET].sum())
    base_rate = n_pos / n_firms if n_firms > 0 else 0
    mean_prob = grp['proba_best'].mean()

    n_top_sec     = max(1, int(np.round(0.10 * n_firms)))
    sorted_sec    = grp.sort_values('proba_best', ascending=False)
    pos_top10_sec = int(sorted_sec.iloc[:n_top_sec][TARGET].sum())
    lift_sec      = (pos_top10_sec / n_top_sec) / base_rate if base_rate > 0 else np.nan

    pr_auc_sec = np.nan
    if 2 <= n_pos < n_firms:
        try:
            pr_auc_sec = average_precision_score(grp[TARGET], grp['proba_best'])
        except Exception:
            pass

    sector_rows.append({
        'SIC_Division':    div,
        'N_Firms':         n_firms,
        'N_Targets':       n_pos,
        'Base_Rate_pct':   round(base_rate * 100, 2),
        'Mean_PredProb':   round(mean_prob, 4),
        'Pos_in_Top10pct': pos_top10_sec,
        'Top10pct_N':      n_top_sec,
        'Lift_at_10pct':   round(lift_sec, 3) if not np.isnan(lift_sec) else np.nan,
        'PR_AUC':          round(pr_auc_sec, 4) if not np.isnan(pr_auc_sec) else np.nan,
        'N_TP':            int((grp['error_type'] == 'TP').sum()),
        'N_FP':            int((grp['error_type'] == 'FP').sum()),
        'N_FN':            int((grp['error_type'] == 'FN').sum()),
    })

sector_df = (pd.DataFrame(sector_rows)
             .sort_values('N_Targets', ascending=False)
             .reset_index(drop=True))
sector_df.to_csv(out('table_13_sector_analysis.csv'), index=False)
print(f"  Table 13 saved → {out('table_13_sector_analysis.csv')}")

# ── Figure 17 — sector analysis ──────────────────────────────────────────────
fig17, axes17 = plt.subplots(1, 2, figsize=(14, 6))

ax_a = axes17[0]
plot_sec = sector_df[sector_df['N_Targets'] >= 1].copy()
ax_a.barh(plot_sec['SIC_Division'], plot_sec['Base_Rate_pct'],
          color='#2c7bb6', alpha=0.75)
ax_a.axvline(x=NO_SKILL_PRAUC * 100, color='gray', linestyle='--', lw=1.5,
             label=f'Overall base rate ({NO_SKILL_PRAUC*100:.2f}%)')
ax_a.set_xlabel('M&A Target Base Rate (%)')
ax_a.set_title('Panel A: M&A Target Rate by SIC Division\n(Test Set 2023–2024)', fontsize=10)
ax_a.legend(fontsize=8)
ax_a.invert_yaxis()
for i, (_, row) in enumerate(plot_sec.iterrows()):
    ax_a.text(row['Base_Rate_pct'] + 0.05, i,
              f"n={row['N_Targets']}/{row['N_Firms']}",
              va='center', fontsize=7.5, color='black')

ax_b = axes17[1]
valid = sector_df.dropna(subset=['Lift_at_10pct'])
sizes = (valid['N_Firms'] / valid['N_Firms'].max() * 500).clip(lower=30)
sc = ax_b.scatter(valid['Base_Rate_pct'], valid['Lift_at_10pct'],
                  s=sizes, c=valid['N_Targets'],
                  cmap='YlOrRd', alpha=0.8, edgecolors='black', linewidths=0.5)
plt.colorbar(sc, ax=ax_b, label='# M&A Targets')
ax_b.axhline(y=1.0, color='gray', linestyle='--', lw=1, label='Random (lift=1.0x)')
for _, row in valid.iterrows():
    ax_b.annotate(row['SIC_Division'].split()[0],
                  (row['Base_Rate_pct'], row['Lift_at_10pct']),
                  fontsize=7, xytext=(3, 3), textcoords='offset points')
ax_b.set_xlabel('Sector Base Rate (%)')
ax_b.set_ylabel(f'Lift@10% — {best_model}')
ax_b.set_title('Panel B: Model Lift@10% vs Base Rate by Sector\n'
               '(bubble size ∝ number of firms)', fontsize=10)
ax_b.legend(fontsize=8)

fig17.suptitle('Figure 17: Sector-Level Analysis — Test Set (2023–2024)', fontsize=12)
plt.tight_layout()
plt.savefig(out('fig_17_sector_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Figure 17 saved → {out('fig_17_sector_analysis.png')}")

# ── Table 14 — error profile (feature means per TP / FP / FN) ────────────────
pos_mask   = df_test[TARGET] == 1
error_rows = []
for feat in FEATURES:
    row = {'Feature': feat}
    for grp_code in ('TP', 'FP', 'FN'):
        vals = df_test.loc[df_test['error_type'] == grp_code, feat].dropna()
        row[f'{grp_code}_mean']   = round(vals.mean(),   4) if len(vals) > 0 else np.nan
        row[f'{grp_code}_median'] = round(vals.median(), 4) if len(vals) > 0 else np.nan
        row[f'{grp_code}_n']      = len(vals)
    pos_vals = df_test.loc[pos_mask, feat].dropna()
    row['AllPos_mean']   = round(pos_vals.mean(),   4)
    row['AllPos_median'] = round(pos_vals.median(), 4)
    row['AllPos_n']      = len(pos_vals)
    all_vals = df_test[feat].dropna()
    row['Overall_mean']   = round(all_vals.mean(),   4)
    row['Overall_median'] = round(all_vals.median(), 4)
    error_rows.append(row)

error_df = pd.DataFrame(error_rows)
error_df.to_csv(out('table_14_error_profile.csv'), index=False)
print(f"  Table 14 saved → {out('table_14_error_profile.csv')}")

# ── Figure 18 — error characterisation heatmap ───────────────────────────────
overall_means = error_df.set_index('Feature')['Overall_mean']
overall_stds  = {f: df_test[f].std() for f in FEATURES}
heat_data = {}
for grp_code in ('TP', 'FP', 'FN'):
    col = error_df.set_index('Feature')[f'{grp_code}_mean']
    heat_data[grp_code] = (col - overall_means) / pd.Series(overall_stds)

heat_df = pd.DataFrame(heat_data)

fig18, axes18 = plt.subplots(1, 2, figsize=(14, 7),
                              gridspec_kw={'width_ratios': [1, 2]})
sns.heatmap(
    heat_df, ax=axes18[0], cmap='RdBu_r', center=0, vmin=-3, vmax=3,
    annot=True, fmt='.2f', linewidths=0.5,
    cbar_kws={'label': 'Z-score vs overall mean'},
)
axes18[0].set_title(
    f'Panel A: Feature Z-scores by Error Group\n'
    f'(best model: {best_model}, threshold={best_thresh:.3f})',
    fontsize=10
)
axes18[0].set_xlabel('Error Group')
axes18[0].set_ylabel('Feature')

tp_means = error_df.set_index('Feature')['TP_mean']
if fn_b > 0:
    compare_means = error_df.set_index('Feature')['FN_mean']
    compare_label = f'False Negative (n={fn_b})'
    panel_b_title = ('Panel B: TP vs FN Feature Means\n'
                     '(sorted by |TP mean − FN mean|, largest diff at top)')
else:
    compare_means = error_df.set_index('Feature')['FP_mean']
    compare_label = f'False Positive (n={fp_b})'
    panel_b_title = ('Panel B: TP vs FP Feature Means\n'
                     '(FN=0 at this threshold; showing over-prediction profile)')

diff_abs       = (tp_means - compare_means).abs().sort_values(ascending=False)
feat_order_err = diff_abs.index.tolist()
y_pos_err      = np.arange(len(feat_order_err))
bar_h_err      = 0.35
ax_bar         = axes18[1]
ax_bar.barh(y_pos_err + bar_h_err/2,
            [tp_means[f]      for f in feat_order_err], height=bar_h_err,
            color='#d7191c', alpha=0.8, label=f'True Positive (n={tp_b})')
ax_bar.barh(y_pos_err - bar_h_err/2,
            [compare_means[f] for f in feat_order_err], height=bar_h_err,
            color='#2c7bb6', alpha=0.8, label=compare_label)
ax_bar.set_yticks(y_pos_err)
ax_bar.set_yticklabels(feat_order_err, fontsize=9)
ax_bar.invert_yaxis()
ax_bar.axvline(0, color='black', lw=0.6)
ax_bar.set_xlabel('Feature Mean Value')
ax_bar.set_title(panel_b_title, fontsize=10)
ax_bar.legend(fontsize=9)
ax_bar.grid(axis='x', alpha=0.3)

fig18.suptitle('Figure 18: Error Characterisation — Test Set (2023–2024)', fontsize=12)
plt.tight_layout()
plt.savefig(out('fig_18_error_characterization.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Figure 18 saved → {out('fig_18_error_characterization.png')}")

# ── Table 15 — top-50 predicted targets ──────────────────────────────────────
id_cols_present = [c for c in ['gvkey', 'conm', 'tic', 'sic', 'sic_division']
                   if c in df_test.columns]
top50_df = df_test[id_cols_present + [TARGET]].copy()
top50_df['Pred_Prob']    = proba_best
top50_df['Actual_Label'] = y_test
top50_df['Rank']         = (top50_df['Pred_Prob']
                            .rank(ascending=False, method='first').astype(int))
top50_df = top50_df.sort_values('Rank').head(50).reset_index(drop=True)
top50_df['Correct'] = ((top50_df['Pred_Prob'] >= best_thresh).astype(int)
                       == top50_df['Actual_Label'])
top50_df.to_csv(out('table_15_top50_predicted_targets.csv'), index=False)
n_tp_top50 = int(top50_df['Actual_Label'].sum())
print(f"  Table 15 saved → {out('table_15_top50_predicted_targets.csv')}")
print(f"  Top-50 captures {n_tp_top50}/{int(y_test.sum())} targets "
      f"({n_tp_top50/y_test.sum()*100:.1f}% recall at rank ≤ 50)")

# ── Figure 19 — probability calibration, all 4 models ────────────────────────
N_BINS = 10
fig19, (ax_cal, ax_hist) = plt.subplots(1, 2, figsize=(13, 5))

print(f"\n  Calibration ECE (quantile binning, n_bins={N_BINS}):")
for name in MODEL_ORDER:
    proba = test_probas[name]
    frac_pos, mean_pred = calibration_curve(
        y_test, proba, n_bins=N_BINS, strategy='quantile'
    )
    ax_cal.plot(mean_pred, frac_pos, marker='o', lw=2,
                color=COLORS[name], label=name, ms=5)
    ax_hist.hist(proba, bins=40, alpha=0.35, color=COLORS[name],
                 label=name, density=True)

    # ECE
    counts, edges = np.histogram(proba, bins=N_BINS)
    ece = 0.0
    for b in range(N_BINS):
        mask_b = (proba >= edges[b]) & (proba < edges[b + 1])
        if mask_b.sum() > 0:
            ece += (mask_b.sum() / len(proba)
                    * abs(y_test[mask_b].mean() - proba[mask_b].mean()))
    print(f"    {name:<25} ECE = {ece:.4f}")

ax_cal.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
ax_cal.set_xlabel('Mean Predicted Probability')
ax_cal.set_ylabel('Fraction of Positives')
ax_cal.set_title('Panel A: Reliability Diagram\n(quantile binning, n_bins=10)', fontsize=10)
ax_cal.legend(fontsize=9)
ax_cal.grid(alpha=0.3)

ax_hist.axvline(x=best_thresh, color='black', linestyle='--', lw=1.5,
                label=f'Best-model threshold ({best_thresh:.3f})')
ax_hist.set_xlabel('Predicted Probability')
ax_hist.set_ylabel('Density (log scale)')
ax_hist.set_yscale('log')
ax_hist.set_title('Panel B: Distribution of Predicted Probabilities\n'
                  '(log y-scale; dashed = optimal threshold)', fontsize=10)
ax_hist.legend(fontsize=9)
ax_hist.grid(alpha=0.3, which='both')

fig19.suptitle('Figure 19: Probability Calibration — Test Set (2023–2024)', fontsize=12)
plt.tight_layout()
plt.savefig(out('fig_19_calibration.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Figure 19 saved → {out('fig_19_calibration.png')}")

# ── Figure 20 — threshold sensitivity, all 4 models (equal treatment) ────────
fig20, (ax_f1, ax_lift) = plt.subplots(1, 2, figsize=(14, 5))

for name in MODEL_ORDER:
    proba = test_probas[name]
    f1s, precs, recs, lifts = [], [], [], []
    for t in THRESHOLD_GRID:
        preds_t = (proba >= t).astype(int)
        f1s.append(f1_score(y_test, preds_t, zero_division=0))
        precs.append(precision_score(y_test, preds_t, zero_division=0))
        recs.append(recall_score(y_test, preds_t, zero_division=0))
        n_pred = preds_t.sum()
        lifts.append((y_test[preds_t == 1].sum() / n_pred) / NO_SKILL_PRAUC
                     if n_pred > 0 else 0)

    # All models use identical line widths
    ax_f1.plot(THRESHOLD_GRID, f1s,   color=COLORS[name], lw=2,
               linestyle='-',  label=f'{name} F1')
    ax_f1.plot(THRESHOLD_GRID, precs, color=COLORS[name], lw=1.2,
               linestyle='--', alpha=0.7)
    ax_f1.plot(THRESHOLD_GRID, recs,  color=COLORS[name], lw=1.2,
               linestyle=':',  alpha=0.7)
    ax_lift.plot(THRESHOLD_GRID, lifts, color=COLORS[name], lw=2, label=name)

ax_f1.axvline(x=best_thresh, color='black', linestyle='--', lw=1.2,
              label=f'{best_model} thresh ({best_thresh:.3f})')
ax_f1.set_xlabel('Threshold')
ax_f1.set_ylabel('Score')
ax_f1.set_title('Panel A: F1 (solid), Precision (dashed),\nRecall (dotted) vs Threshold',
                fontsize=10)
ax_f1.legend(fontsize=8, loc='center right')
ax_f1.grid(alpha=0.3)
ax_f1.set_xlim(THRESHOLD_GRID[0], THRESHOLD_GRID[-1])
ax_f1.set_ylim(0, 1.02)

ax_lift.axhline(y=1.0, color='gray', linestyle='--', lw=1,
                label='No-skill (lift=1.0x)')
ax_lift.axvline(x=best_thresh, color='black', linestyle='--', lw=1.2,
                label=f'{best_model} thresh ({best_thresh:.3f})')
ax_lift.set_xlabel('Threshold')
ax_lift.set_ylabel('Precision Lift = Precision / Base Rate')
ax_lift.set_title('Panel B: Precision Lift vs Threshold\n'
                  '(lift=1.0x is random screening)', fontsize=10)
ax_lift.legend(fontsize=8, loc='upper left')
ax_lift.grid(alpha=0.3)
ax_lift.set_xlim(THRESHOLD_GRID[0], THRESHOLD_GRID[-1])
ax_lift.set_ylim(bottom=0)

fig20.suptitle('Figure 20: Threshold Sensitivity — Test Set (2023–2024)', fontsize=12)
plt.tight_layout()
plt.savefig(out('fig_20_threshold_sensitivity.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Figure 20 saved → {out('fig_20_threshold_sensitivity.png')}")


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Script 08 complete — all outputs saved to output_combined/")
print("=" * 60)

outputs_list = [
    ("Table 09", "table_09_cv_performance.csv"),
    ("Figure 10", "fig_10_cv_stability.png"),
    ("Table 10", "table_10_test_metrics.csv"),
    ("Table 11", "table_11_statistical_tests.csv"),
    ("Figure 11", "fig_11_roc_pr_curves.png"),
    ("Figure 12", "fig_12_confusion_matrices.png"),
    ("Figure 13", "fig_13_lift_chart.png"),
    ("Table 12",  "table_12_feature_importance.csv"),
    ("Figure 14", "fig_14_shap_beeswarm_xgb.png"),
    ("Figure 15", "fig_15_feature_importance_comparison.png"),
    ("Figure 16", "fig_16_shap_dependence.png"),
    ("Table 13",  "table_13_sector_analysis.csv"),
    ("Table 14",  "table_14_error_profile.csv"),
    ("Table 15",  "table_15_top50_predicted_targets.csv"),
    ("Figure 17", "fig_17_sector_analysis.png"),
    ("Figure 18", "fig_18_error_characterization.png"),
    ("Figure 19", "fig_19_calibration.png"),
    ("Figure 20", "fig_20_threshold_sensitivity.png"),
]

print()
for label, fname in outputs_list:
    print(f"  {label:<12} → output_combined/{fname}")

print(f"\n  Best model (Test PR-AUC):  {best_model} ({best_prauc:.4f}, "
      f"lift = {best_prauc/NO_SKILL_PRAUC:.2f}x)")
print(f"  No-skill PR-AUC baseline:  {NO_SKILL_PRAUC}")
print(f"  Models compared:           {', '.join(MODEL_ORDER)}")
