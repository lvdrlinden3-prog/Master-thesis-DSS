import os
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

warnings.filterwarnings('ignore')
os.makedirs('outputs', exist_ok=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
NO_SKILL_PRAUC = 0.0157
RANDOM_STATE   = 42
DATA_FILE      = 'data_final_modeling_ma_v7.xlsx'
TARGET         = 'target_next_year'
FEATURES = [
    # --- Original 11 (v5) ---
    'profitability', 'leverage', 'cash_ratio', 'fcf_debt',
    'ppe_ratio', 'capex_intensity', 'asset_turnover',
    'interest_burden', 'net_margin', 'rev_growth', 'fcf_volatility',
    # --- New v6 (high coverage) ---
    'firm_size',
    'rd_intensity',
    'rev_growth_lag1',
    'altman_re_ta',
]
COLORS = {
    'Logistic Regression': '#2c7bb6',
    'XGBoost':             '#d7191c',
    'MLP':                 '#1a9641',
}

# Theory directions from Script 02 — used for alignment column in Table 12
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
# STEP 6.1 — Load data and models
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 6.1 — Load data and models")
print("=" * 60)

df = pd.read_excel(DATA_FILE)
df['fiscal_year'] = pd.to_datetime(df['datadate']).dt.year

df_train = df[df['fiscal_year'].between(2012, 2021)].copy()
df_test  = df[df['fiscal_year'].between(2023, 2024)].copy()

X_train = df_train[FEATURES].values;  y_train = df_train[TARGET].values
X_test  = df_test[FEATURES].values;   y_test  = df_test[TARGET].values

print(f"\n  Train (2012–2021)  : {len(y_train):,} obs | {y_train.sum()} pos")
print(f"  Test  (2023–2024)  : {len(X_test):,} obs | {y_test.sum()} pos")

lr_path  = find_output('model_baseline_logistic.pkl')
xgb_path = find_output('model_xgb.pkl')
mlp_path = find_output('model_mlp_pipeline.pkl')

with open(lr_path,  'rb') as f: lr_pipeline  = pickle.load(f)
with open(xgb_path, 'rb') as f: model_xgb    = pickle.load(f)
with open(mlp_path, 'rb') as f: mlp_pipeline = pickle.load(f)

print(f"\n  Loaded: {lr_path}")
print(f"  Loaded: {xgb_path}")
print(f"  Loaded: {mlp_path}")

# Identify best model from Script 05 output (fallback: XGBoost)
try:
    metrics_df  = pd.read_csv(find_output('table_10_test_metrics.csv'))
    best_model  = metrics_df.loc[metrics_df['PR_AUC'].idxmax(), 'Model']
    best_prauc  = metrics_df['PR_AUC'].max()
    print(f"\n  Best model from Script 05: {best_model} (Test PR-AUC = {best_prauc:.4f}, "
          f"lift = {best_prauc / NO_SKILL_PRAUC:.2f}x)")
except Exception:
    best_model = 'XGBoost'
    print("  (table_10 not found — defaulting best model to XGBoost)")

# Scaled test data for LR LinearExplainer
lr_scaler      = lr_pipeline.named_steps['scaler']
lr_model       = lr_pipeline.named_steps['lr']
X_train_scaled = lr_scaler.transform(X_train)
X_test_scaled  = lr_scaler.transform(X_test)

# ─────────────────────────────────────────────
# STEP 6.2 — XGBoost SHAP (TreeExplainer)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6.2 — XGBoost SHAP (TreeExplainer, full test set)")
print("=" * 60)

explainer_xgb  = shap.TreeExplainer(model_xgb)
shap_values_xgb = explainer_xgb.shap_values(X_test)   # shape: (n_test, n_features)
mean_abs_shap_xgb = np.abs(shap_values_xgb).mean(axis=0)

print(f"\n  SHAP values computed for {len(X_test):,} test observations")
print(f"\n  Mean |SHAP| per feature (XGBoost, descending):")
xgb_rank = np.argsort(mean_abs_shap_xgb)[::-1]
for rank, i in enumerate(xgb_rank, 1):
    print(f"    {rank:2d}. {FEATURES[i]:<20}  {mean_abs_shap_xgb[i]:.5f}")

# ── Figure 14: SHAP beeswarm ──────────────────────────────────────────────
print("\n  Generating Figure 14: SHAP beeswarm plot ...")

shap_expl = shap.Explanation(
    values=shap_values_xgb,
    base_values=np.full(len(X_test), explainer_xgb.expected_value),
    data=X_test,
    feature_names=FEATURES,
)

# beeswarm creates its own figure; pass plot_size=None so we can pre-size it
plt.figure(figsize=(9, 7))
shap.plots.beeswarm(shap_expl, max_display=15, show=False, plot_size=None)
fig14 = plt.gcf()
fig14.suptitle(
    f'Figure 14: XGBoost SHAP Beeswarm — Test Set (2023–2024)\n'
    f'n={len(X_test):,} obs | colour = feature value (red=high, blue=low)',
    fontsize=10, y=1.01
)
plt.tight_layout()
fig14.savefig('outputs/fig_14_shap_beeswarm_xgb.png', dpi=300, bbox_inches='tight')
plt.close('all')
print("  Figure 14 saved: outputs/fig_14_shap_beeswarm_xgb.png")

# ─────────────────────────────────────────────
# STEP 6.3 — LR SHAP (LinearExplainer)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6.3 — Logistic Regression SHAP (LinearExplainer)")
print("=" * 60)

explainer_lr    = shap.LinearExplainer(lr_model, X_train_scaled, nsamples=500)
shap_values_lr  = explainer_lr.shap_values(X_test_scaled)   # shape: (n_test, n_features)
mean_abs_shap_lr = np.abs(shap_values_lr).mean(axis=0)

print(f"\n  Mean |SHAP| per feature (LR, descending):")
lr_rank = np.argsort(mean_abs_shap_lr)[::-1]
for rank, i in enumerate(lr_rank, 1):
    print(f"    {rank:2d}. {FEATURES[i]:<20}  {mean_abs_shap_lr[i]:.5f}")

# ─────────────────────────────────────────────
# STEP 6.4 — MLP Permutation Importance
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6.4 — MLP Permutation Importance (n_repeats=50, scoring=AP)")
print("=" * 60)
print("  This may take ~60–120 seconds ...")

perm_result = permutation_importance(
    mlp_pipeline, X_test, y_test,
    scoring='average_precision',
    n_repeats=50,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

mean_perm_mlp = perm_result.importances_mean   # higher = more important
std_perm_mlp  = perm_result.importances_std

# Normalise to [0, 1] for cross-model comparison
# (permutation drops in AP are on a different scale than SHAP values)
mlp_imp_norm = np.clip(mean_perm_mlp, 0, None)   # clip negatives (noise) to 0
mlp_imp_norm = mlp_imp_norm / mlp_imp_norm.max() if mlp_imp_norm.max() > 0 else mlp_imp_norm

print(f"\n  Permutation importance per feature (MLP, descending):")
mlp_rank = np.argsort(mean_perm_mlp)[::-1]
for rank, i in enumerate(mlp_rank, 1):
    print(f"    {rank:2d}. {FEATURES[i]:<20}  {mean_perm_mlp[i]:.5f} ± {std_perm_mlp[i]:.5f}")

# ─────────────────────────────────────────────
# STEP 6.5 — Unified feature importance table (Table 12)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6.5 — Unified feature importance table (Table 12)")
print("=" * 60)

# Normalise SHAP means to [0, 1] for cross-model rank comparison
xgb_norm = mean_abs_shap_xgb / mean_abs_shap_xgb.max()
lr_norm  = mean_abs_shap_lr  / mean_abs_shap_lr.max()
mlp_norm = mlp_imp_norm   # already normalised above

# Load LR coefficients from Script 02 for theory-alignment column
try:
    coef_csv = pd.read_csv(find_output('table_07_logistic_coefficients.csv'))
    coef_map  = dict(zip(coef_csv['Feature'], coef_csv['Coefficient']))
    tc_map    = dict(zip(coef_csv['Feature'], coef_csv['Theory_Consistent']))
except Exception:
    coef_map = {f: np.nan for f in FEATURES}
    tc_map   = {f: '?' for f in FEATURES}

rows = []
pos_idx = np.where(y_test == 1)[0]   # restrict to actual M&A targets
for i, feat in enumerate(FEATURES):
    # XGBoost sign: mean SHAP among TRUE POSITIVES (Class 1 instances).
    # Using full-test mean is misleading under 81:1 imbalance — log-odds SHAP
    # is dominated by Class 0 observations and is negative for all features.
    xgb_sign = '+' if np.mean(shap_values_xgb[pos_idx, i]) >= 0 else '-'
    rows.append({
        'Feature':               feat,
        'Theory_Direction':      THEORY_DIRECTIONS[feat],
        'LR_Coefficient':        round(coef_map.get(feat, np.nan), 4),
        'Theory_Consistent_LR':  tc_map.get(feat, '?'),
        'XGBoost_MeanAbsSHAP':   round(mean_abs_shap_xgb[i], 6),
        'XGBoost_Rank':          int(np.where(xgb_rank == i)[0][0]) + 1,
        'XGBoost_SHAP_Sign':     xgb_sign,
        'LR_MeanAbsSHAP':        round(mean_abs_shap_lr[i], 6),
        'LR_Rank':               int(np.where(lr_rank == i)[0][0]) + 1,
        'MLP_PermImp_Mean':      round(mean_perm_mlp[i], 6),
        'MLP_PermImp_Std':       round(std_perm_mlp[i], 6),
        'MLP_Rank':              int(np.where(mlp_rank == i)[0][0]) + 1,
        'XGBoost_Norm':          round(xgb_norm[i], 4),
        'LR_Norm':               round(lr_norm[i], 4),
        'MLP_Norm':              round(mlp_norm[i], 4),
    })

imp_df = pd.DataFrame(rows).sort_values('XGBoost_Rank')
imp_df.to_csv('outputs/table_12_feature_importance.csv', index=False)
print("Table 12 saved: outputs/table_12_feature_importance.csv")

print(f"\n  Feature importance summary (sorted by XGBoost SHAP rank):")
print(f"  {'Rank':<5} {'Feature':<20} {'Theory':<7} "
      f"{'XGB mean|SHAP|':<16} {'LR mean|SHAP|':<15} {'MLP PermImp':<12} "
      f"{'XGB sign':<10} {'LR TC'}")
print("  " + "-" * 90)
for _, row in imp_df.iterrows():
    print(f"  {row['XGBoost_Rank']:<5} {row['Feature']:<20} "
          f"{row['Theory_Direction']:<7} "
          f"{row['XGBoost_MeanAbsSHAP']:<16.5f} "
          f"{row['LR_MeanAbsSHAP']:<15.5f} "
          f"{row['MLP_PermImp_Mean']:<12.5f} "
          f"{row['XGBoost_SHAP_Sign']:<10} "
          f"{row['Theory_Consistent_LR']}")

# ─────────────────────────────────────────────
# STEP 6.6 — Figure 15: Feature importance comparison (all 3 models)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6.6 — Figure 15: Feature importance comparison bar chart")
print("=" * 60)

# Sort by XGBoost normalised importance for a consistent visual ranking
feat_order = imp_df['Feature'].tolist()   # already sorted by XGB rank
xgb_vals   = [imp_df.loc[imp_df['Feature'] == f, 'XGBoost_Norm'].values[0] for f in feat_order]
lr_vals    = [imp_df.loc[imp_df['Feature'] == f, 'LR_Norm'].values[0]     for f in feat_order]
mlp_vals   = [imp_df.loc[imp_df['Feature'] == f, 'MLP_Norm'].values[0]    for f in feat_order]

n_feats = len(feat_order)
y_pos   = np.arange(n_feats)
bar_h   = 0.25

fig15, ax15 = plt.subplots(figsize=(10, 8))
ax15.barh(y_pos + bar_h,  xgb_vals, height=bar_h, color=COLORS['XGBoost'],
          alpha=0.85, label='XGBoost (mean |SHAP|, normalised)')
ax15.barh(y_pos,          lr_vals,  height=bar_h, color=COLORS['Logistic Regression'],
          alpha=0.85, label='Logistic Regression (mean |SHAP|, normalised)')
ax15.barh(y_pos - bar_h,  mlp_vals, height=bar_h, color=COLORS['MLP'],
          alpha=0.85, label='MLP (permutation importance, normalised)')

ax15.set_yticks(y_pos)
ax15.set_yticklabels(feat_order, fontsize=9)
ax15.invert_yaxis()   # most important at top
ax15.set_xlabel('Normalised Feature Importance (1.0 = most important for each model)')
ax15.set_title(
    'Figure 15: Feature Importance Comparison — All Three Models\n'
    'Sorted by XGBoost SHAP rank | Test Set (2023–2024)',
    fontsize=11
)
ax15.legend(fontsize=9, loc='lower right')
ax15.axvline(x=0, color='black', lw=0.6)
ax15.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/fig_15_feature_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 15 saved: outputs/fig_15_feature_importance_comparison.png")

# ─────────────────────────────────────────────
# STEP 6.7 — Figure 16: SHAP dependence plots (top 4 XGBoost features)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6.7 — Figure 16: SHAP dependence plots (top 4 XGBoost features)")
print("=" * 60)

top4_features = [FEATURES[i] for i in xgb_rank[:4]]
print(f"  Top 4 features: {top4_features}")

fig16, axes16 = plt.subplots(2, 2, figsize=(12, 9))
axes16 = axes16.flatten()

for ax, feat in zip(axes16, top4_features):
    feat_idx = FEATURES.index(feat)
    shap.dependence_plot(
        feat_idx,
        shap_values_xgb,
        X_test,
        feature_names=FEATURES,
        ax=ax,
        show=False,
        dot_size=8,
        alpha=0.5,
    )
    ax.set_title(f'{feat}\n(Theory: {THEORY_DIRECTIONS[feat]})', fontsize=10)
    ax.grid(alpha=0.3)

fig16.suptitle(
    'Figure 16: XGBoost SHAP Dependence Plots — Top 4 Features\n'
    'Test Set (2023–2024) | Colour = interaction feature (auto-selected by SHAP)',
    fontsize=11
)
plt.tight_layout()
plt.savefig('outputs/fig_16_shap_dependence.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 16 saved: outputs/fig_16_shap_dependence.png")

# ─────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Script 06 complete. All outputs saved to outputs/")
print("=" * 60)

print(f"\n  Best model (Script 05): {best_model}")
print(f"\n  Top 5 features by XGBoost SHAP rank:")
for rank in range(5):
    feat = feat_order[rank]
    row  = imp_df[imp_df['Feature'] == feat].iloc[0]
    print(f"    {rank+1}. {feat:<20}  XGB|SHAP|={row['XGBoost_MeanAbsSHAP']:.5f}  "
          f"sign={row['XGBoost_SHAP_Sign']}  theory={row['Theory_Direction']}  "
          f"LR_TC={row['Theory_Consistent_LR']}")

print(f"\n  Rank agreement (XGBoost SHAP rank vs LR SHAP rank):")
from scipy.stats import spearmanr
xgb_ranks_arr = imp_df['XGBoost_Rank'].values
lr_ranks_arr  = imp_df['LR_Rank'].values
rho, pval = spearmanr(xgb_ranks_arr, lr_ranks_arr)
print(f"    Spearman rho = {rho:.3f}  (p = {pval:.3f})")
print(f"    Interpretation: {'strong' if abs(rho) >= 0.6 else 'moderate' if abs(rho) >= 0.4 else 'weak'} "
      f"rank agreement between XGBoost and LR feature importance orderings")

print(f"\n  Features with largest rank disagreement (|XGB rank - LR rank| > 4):")
imp_df['Rank_Diff'] = (imp_df['XGBoost_Rank'] - imp_df['LR_Rank']).abs()
large_diff = imp_df[imp_df['Rank_Diff'] > 4].sort_values('Rank_Diff', ascending=False)
if len(large_diff):
    for _, row in large_diff.iterrows():
        print(f"    {row['Feature']:<20}  XGB rank={row['XGBoost_Rank']:2d}  "
              f"LR rank={row['LR_Rank']:2d}  diff={row['Rank_Diff']:2d}")
else:
    print("    None — XGBoost and LR agree on feature ranking throughout.")

print(f"\n  Outputs:")
print(f"    table_12_feature_importance.csv")
print(f"    fig_14_shap_beeswarm_xgb.png")
print(f"    fig_15_feature_importance_comparison.png")
print(f"    fig_16_shap_dependence.png")
print(f"\n  No-skill PR-AUC baseline: {NO_SKILL_PRAUC}")
