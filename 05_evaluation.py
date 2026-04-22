import os
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score,
    confusion_matrix, brier_score_loss,
    balanced_accuracy_score,
)

warnings.filterwarnings('ignore')
os.makedirs('outputs', exist_ok=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
NO_SKILL_PRAUC = 0.0157
RANDOM_STATE   = 42
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
THRESHOLD_GRID = np.arange(0.001, 0.151, 0.001)
N_BOOT         = 2000


def find_output(filename):
    """Locate a file anywhere under outputs/ (handles user-organised subdirs)."""
    matches = list(Path('outputs').rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"'{filename}' not found anywhere under outputs/. "
            "Run the prerequisite script first."
        )
    return str(sorted(matches, key=lambda p: p.stat().st_mtime)[-1])


# ─────────────────────────────────────────────
# STEP 5.1 -- Load data and models
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 5.1 -- Load data and models")
print("=" * 60)

df = pd.read_excel(DATA_FILE)
df['fiscal_year'] = pd.to_datetime(df['datadate']).dt.year

df_val  = df[df['fiscal_year'] == 2022].copy()
df_test = df[df['fiscal_year'].between(2023, 2024)].copy()

X_val  = df_val[FEATURES].values;  y_val  = df_val[TARGET].values
X_test = df_test[FEATURES].values; y_test = df_test[TARGET].values

print(f"\n  Val  (2022)     : {len(y_val):,} obs | {y_val.sum()} pos | "
      f"{y_val.sum()/len(y_val)*100:.2f}%")
print(f"  Test (2023–2024): {len(y_test):,} obs | {y_test.sum()} pos | "
      f"{y_test.sum()/len(y_test)*100:.2f}%")

# Load model artifacts
lr_path  = find_output('model_baseline_logistic.pkl')
xgb_path = find_output('model_xgb.pkl')
mlp_path = find_output('model_mlp_pipeline.pkl')

with open(lr_path,  'rb') as f: lr_pipeline  = pickle.load(f)
with open(xgb_path, 'rb') as f: model_xgb    = pickle.load(f)
with open(mlp_path, 'rb') as f: mlp_pipeline = pickle.load(f)

print(f"\n  Loaded: {lr_path}")
print(f"  Loaded: {xgb_path}")
print(f"  Loaded: {mlp_path}")

# Generate predicted probabilities
val_probas = {
    'Logistic Regression': lr_pipeline.predict_proba(X_val)[:, 1],
    'XGBoost':             model_xgb.predict_proba(X_val)[:, 1],
    'MLP':                 mlp_pipeline.predict_proba(X_val)[:, 1],
}
test_probas = {
    'Logistic Regression': lr_pipeline.predict_proba(X_test)[:, 1],
    'XGBoost':             model_xgb.predict_proba(X_test)[:, 1],
    'MLP':                 mlp_pipeline.predict_proba(X_test)[:, 1],
}


# ─────────────────────────────────────────────
# STEP 5.2 -- Threshold selection on validation set
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5.2 -- Threshold selection (argmax F1 on val set)")
print("=" * 60)

optimal_thresholds = {}
print(f"\n  {'Model':<25} Opt. Threshold  Val F1   Val Precision  Val Recall")
print("  " + "-" * 65)
for name, proba in val_probas.items():
    f1_scores = [
        f1_score(y_val, (proba >= t).astype(int), zero_division=0)
        for t in THRESHOLD_GRID
    ]
    best_t = float(THRESHOLD_GRID[np.argmax(f1_scores)])
    best_f1 = float(np.max(f1_scores))
    preds = (proba >= best_t).astype(int)
    optimal_thresholds[name] = best_t
    print(f"  {name:<25} {best_t:.3f}           {best_f1:.4f}  "
          f"{precision_score(y_val, preds, zero_division=0):.4f}         "
          f"{recall_score(y_val, preds, zero_division=0):.4f}")


# ─────────────────────────────────────────────
# STEP 5.3 -- Full metric table on TEST set
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5.3 -- Full metric table on TEST set")
print("=" * 60)

N_test      = len(y_test)
n_pos_test  = int(y_test.sum())
top5_n      = max(1, int(np.round(0.05 * N_test)))
top10_n     = max(1, int(np.round(0.10 * N_test)))
expected_5  = round(0.05 * N_test * NO_SKILL_PRAUC, 2)  # prevalence * n_top
expected_10 = round(0.10 * N_test * NO_SKILL_PRAUC, 2)

print(f"\n  Test set: {N_test:,} obs | {n_pos_test} pos | "
      f"base rate = {n_pos_test/N_test*100:.2f}%")
print(f"  Top 5%  = {top5_n} firms | expected positives (random) = {expected_5:.1f}")
print(f"  Top 10% = {top10_n} firms | expected positives (random) = {expected_10:.1f}")

metric_rows = []
for name in ['Logistic Regression', 'XGBoost', 'MLP']:
    proba = test_probas[name]
    thresh = optimal_thresholds[name]
    preds  = (proba >= thresh).astype(int)

    auc_roc = roc_auc_score(y_test, proba)
    pr_auc  = average_precision_score(y_test, proba)
    f1      = f1_score(y_test, preds, zero_division=0)
    prec    = precision_score(y_test, preds, zero_division=0)
    rec     = recall_score(y_test, preds, zero_division=0)
    brier   = brier_score_loss(y_test, proba)
    bal_acc = balanced_accuracy_score(y_test, preds)

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Lift@5% and Lift@10%
    sorted_idx = np.argsort(proba)[::-1]
    pos_top5   = int(y_test[sorted_idx[:top5_n]].sum())
    pos_top10  = int(y_test[sorted_idx[:top10_n]].sum())
    lift5  = round((pos_top5 / top5_n) / NO_SKILL_PRAUC, 3)
    lift10 = round((pos_top10 / top10_n) / NO_SKILL_PRAUC, 3)

    metric_rows.append({
        'Model':             name,
        'Threshold':         round(thresh, 3),
        'AUC_ROC':           round(auc_roc, 4),
        'PR_AUC':            round(pr_auc, 4),
        'Lift_vs_NoSkill':   round(pr_auc / NO_SKILL_PRAUC, 3),
        'F1':                round(f1, 4),
        'Precision':         round(prec, 4),
        'Recall':            round(rec, 4),
        'Specificity':       round(specificity, 4),
        'Balanced_Accuracy': round(bal_acc, 4),
        'Brier_Score':       round(brier, 4),
        'Lift_at_5pct':      lift5,
        'Lift_at_10pct':     lift10,
        'TP':                tp,
        'FP':                fp,
        'FN':                fn,
        'TN':                tn,
        'Pos_in_Top5pct':    pos_top5,
        'Pos_in_Top10pct':   pos_top10,
    })

metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv('outputs/table_10_test_metrics.csv', index=False)
print("\nTable 10 saved: outputs/table_10_test_metrics.csv")

print(f"\n  {'Model':<25} ROC-AUC  PR-AUC  Lift  F1      Prec    Rec     Spec    BalAcc  Brier   L@5%   L@10%")
print("  " + "-" * 105)
for row in metric_rows:
    print(f"  {row['Model']:<25} "
          f"{row['AUC_ROC']:.4f}   {row['PR_AUC']:.4f}  "
          f"{row['Lift_vs_NoSkill']:.2f}x  "
          f"{row['F1']:.4f}  {row['Precision']:.4f}  {row['Recall']:.4f}  "
          f"{row['Specificity']:.4f}  {row['Balanced_Accuracy']:.4f}  "
          f"{row['Brier_Score']:.4f}  "
          f"{row['Lift_at_5pct']:.2f}x  {row['Lift_at_10pct']:.2f}x")

print(f"\n  Raw counts (TP/FP/FN/TN) at optimal threshold:")
for row in metric_rows:
    print(f"  {row['Model']:<25} TP={row['TP']:3d}  FP={row['FP']:4d}  "
          f"FN={row['FN']:3d}  TN={row['TN']:5d}")

print(f"\n  Lift details (top-k firms screened):")
print(f"  {'Model':<25} Top 5% ({top5_n} firms)          Top 10% ({top10_n} firms)")
print("  " + "-" * 65)
for row in metric_rows:
    print(f"  {row['Model']:<25} "
          f"{row['Pos_in_Top5pct']:2d}/{top5_n} pos (lift {row['Lift_at_5pct']:.2f}x)   "
          f"{row['Pos_in_Top10pct']:2d}/{top10_n} pos (lift {row['Lift_at_10pct']:.2f}x)")
print(f"  Random screening:          "
      f"~{expected_5:.1f}/{top5_n} expected (1.00x)      "
      f"~{expected_10:.1f}/{top10_n} expected (1.00x)")
print(f"  No-skill baseline: PR-AUC = {NO_SKILL_PRAUC}")


# ─────────────────────────────────────────────
# STEP 5.4-5.5 -- ROC + PR curves + Confusion matrices
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5.4-5.5 -- ROC/PR curves and confusion matrices")
print("=" * 60)

# Figure 11 -- ROC + PR curves side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax_roc, ax_pr = axes

for name in ['Logistic Regression', 'XGBoost', 'MLP']:
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

# ROC panel
ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, label='No-skill (AUC = 0.50)')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('Figure 11a: ROC Curves -- Test Set (2023–2024)', fontsize=11)
ax_roc.legend(fontsize=9, loc='lower right')
ax_roc.grid(alpha=0.3)

# PR panel (log y-scale; base rate = 1.25%)
ax_pr.axhline(y=NO_SKILL_PRAUC, color='gray', linestyle='--', lw=1.5,
              label=f'No-skill baseline (prevalence = 1.57%)')
ax_pr.set_xlabel('Recall')
ax_pr.set_ylabel('Precision (log scale)')
ax_pr.set_yscale('log')
ax_pr.set_title('Figure 11b: Precision-Recall Curves -- Test Set (2023–2024)\n'
                '(log y-scale; base rate = 1.57%)', fontsize=11)
ax_pr.legend(fontsize=9, loc='upper right')
ax_pr.grid(alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('outputs/fig_11_roc_pr_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nFigure 11 saved: outputs/fig_11_roc_pr_curves.png")

# Figure 12 -- Confusion matrices (one per model)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, name in zip(axes, ['Logistic Regression', 'XGBoost', 'MLP']):
    proba  = test_probas[name]
    thresh = optimal_thresholds[name]
    preds  = (proba >= thresh).astype(int)
    cm     = confusion_matrix(y_test, preds)

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
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

plt.suptitle('Figure 12: Confusion Matrices -- Test Set (2023–2024)\n'
             'Threshold = argmax(F1) on validation set', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/fig_12_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 12 saved: outputs/fig_12_confusion_matrices.png")


# ─────────────────────────────────────────────
# STEP 5.6 -- Bootstrap statistical comparison (n=2000)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5.6 -- Bootstrap statistical comparison (n=2000)")
print("=" * 60)
print("  Pairs: XGBoost vs LR | MLP vs LR | XGBoost vs MLP")

rng = np.random.default_rng(RANDOM_STATE)
n   = len(y_test)

def bootstrap_compare(y, proba_a, proba_b, n_boot, rng, metric='roc'):
    """Bootstrap CI and p-value for Delta(metric) = metric_A - metric_B."""
    if metric == 'roc':
        score_fn = roc_auc_score
    else:
        score_fn = average_precision_score

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
    # p-value: proportion of bootstrap samples in opposite direction
    if obs_diff >= 0:
        p = float(2 * np.mean(boot_diffs < 0))
    else:
        p = float(2 * np.mean(boot_diffs > 0))
    p = min(p, 1.0)

    return obs_a, obs_b, obs_diff, ci_lo, ci_hi, p

PAIRS = [
    ('XGBoost', 'Logistic Regression'),
    ('MLP',     'Logistic Regression'),
    ('XGBoost', 'MLP'),
]

stat_rows = []
print(f"\n  {'Comparison':<30} Metric    Delta    95% CI              p-value  Interpretation")
print("  " + "-" * 90)
for name_a, name_b in PAIRS:
    for metric_label, metric_key in [('ROC-AUC', 'roc'), ('PR-AUC', 'pr')]:
        pa = test_probas[name_a]
        pb = test_probas[name_b]
        obs_a, obs_b, delta, ci_lo, ci_hi, p = bootstrap_compare(
            y_test, pa, pb, N_BOOT, rng, metric=metric_key
        )

        # Interpretation: plan note — DAUC < 0.03 unlikely to reach p<0.05 with ~89 pos
        if p < 0.05:
            interp = "statistically significant"
        elif abs(delta) >= 0.03:
            interp = "practically sig., statistically underpowered"
        else:
            interp = "not significant"

        pair_label = f"{name_a} vs {name_b}"
        print(f"  {pair_label:<30} {metric_label:<9} "
              f"{delta:+.4f}  [{ci_lo:+.4f}, {ci_hi:+.4f}]  "
              f"p={p:.3f}   {interp}")

        stat_rows.append({
            'Model_A':        name_a,
            'Model_B':        name_b,
            'Metric':         metric_label,
            f'{name_a.split()[0]}_score': round(obs_a, 4),
            f'{name_b.split()[0]}_score': round(obs_b, 4),
            'Delta_A_minus_B': round(delta, 4),
            'CI_lower_95':    round(ci_lo, 4),
            'CI_upper_95':    round(ci_hi, 4),
            'p_value':        round(p, 4),
            'n_bootstrap':    N_BOOT,
            'Interpretation': interp,
        })

stat_df = pd.DataFrame(stat_rows)
stat_df.to_csv('outputs/table_11_statistical_tests.csv', index=False)
print("\nTable 11 saved: outputs/table_11_statistical_tests.csv")
print(f"\n  Note: test set has {n_pos_test} positive cases. Power note from plan:")
print("  Delta AUC < 0.03 is unlikely to reach p<0.05 with ~89 pos. Report effect sizes")
print("  alongside p-values; if p>0.05 but Delta>=0.03, characterise as")
print("  'practically significant, statistically underpowered'.")


# ─────────────────────────────────────────────
# STEP 5.7 -- Lift chart (Figure 13)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5.7 -- Lift chart (Figure 13)")
print("=" * 60)

# Compute cumulative lift across 1%-100% of population
pct_grid = np.arange(1, 101)   # 1% to 100%
lift_curves = {}

for name in ['Logistic Regression', 'XGBoost', 'MLP']:
    proba = test_probas[name]
    sorted_idx = np.argsort(proba)[::-1]
    y_sorted   = y_test[sorted_idx]
    cumpos     = np.cumsum(y_sorted)

    lifts = []
    for pct in pct_grid:
        n_top = max(1, int(np.round(pct / 100 * N_test)))
        pos_in_top = cumpos[n_top - 1]
        expected   = n_top * NO_SKILL_PRAUC    # base_rate * n_top
        lifts.append(pos_in_top / expected if expected > 0 else 0)
    lift_curves[name] = np.array(lifts)

fig, ax = plt.subplots(figsize=(10, 6))

for name in ['Logistic Regression', 'XGBoost', 'MLP']:
    ax.plot(pct_grid, lift_curves[name], color=COLORS[name], lw=2, label=name)

ax.axhline(y=1.0, color='gray', linestyle='--', lw=1.5, label='Random screening (lift = 1.0x)')

# Annotate at 10%
ax.axvline(x=10, color='black', linestyle=':', lw=1, alpha=0.6)
for name in ['XGBoost', 'MLP', 'Logistic Regression']:
    lift_at10 = lift_curves[name][9]   # index 9 = 10%
    pos_at10  = int(np.round(0.10 * N_test * NO_SKILL_PRAUC * lift_at10))
    ax.annotate(f'{name.split()[0]}: {lift_at10:.1f}x ({pos_at10}/{n_pos_test} targets)',
                xy=(10, lift_at10), fontsize=8.5, ha='left',
                xytext=(12, lift_at10))

# Expected at 10%
expected_10_abs = 0.10 * N_test * NO_SKILL_PRAUC
ax.text(10.5, 1.05, f'At top 10% ({top10_n} firms): ~{expected_10_abs:.1f} expected\nfrom random screening',
        fontsize=8, color='gray', va='bottom')

ax.set_xlabel('% of population screened (ranked by predicted probability)')
ax.set_ylabel('Cumulative Lift (= precision@k% / no-skill baseline)')
ax.set_title(
    f'Figure 13: Cumulative Lift Chart -- Test Set (2023–2024)\n'
    f'No-skill baseline = {NO_SKILL_PRAUC} (prevalence = 1.57%)', fontsize=11
)
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)
ax.set_xlim(0, 100)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('outputs/fig_13_lift_chart.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\nFigure 13 saved: outputs/fig_13_lift_chart.png")

# Print lift@5% and lift@10% from the chart
print(f"\n  Lift chart summary:")
print(f"  {'Model':<25} Lift@5%   Lift@10%")
print("  " + "-" * 45)
for name in ['Logistic Regression', 'XGBoost', 'MLP']:
    print(f"  {name:<25} {lift_curves[name][4]:.2f}x     {lift_curves[name][9]:.2f}x")


# ─────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────
best_model = metrics_df.loc[metrics_df['PR_AUC'].idxmax(), 'Model']
best_prauc = metrics_df['PR_AUC'].max()

print("\n" + "=" * 60)
print("Script 05 complete. All outputs saved to outputs/")
print("=" * 60)
print(f"\n  Best model by Test PR-AUC: {best_model} ({best_prauc:.4f}, "
      f"lift = {best_prauc/NO_SKILL_PRAUC:.2f}x)")
print(f"  (This model is used as 'BEST MODEL' in Script 07 error analysis)")
print(f"\n  Outputs:")
print(f"    table_10_test_metrics.csv")
print(f"    table_11_statistical_tests.csv")
print(f"    fig_11_roc_pr_curves.png")
print(f"    fig_12_confusion_matrices.png")
print(f"    fig_13_lift_chart.png")
print(f"\n  No-skill PR-AUC baseline: {NO_SKILL_PRAUC}")
