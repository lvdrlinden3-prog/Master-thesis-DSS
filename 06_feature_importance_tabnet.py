import os
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')
os.makedirs('outputs', exist_ok=True)

# ─────────────────────────────────────────────
# TabNetWrapper — must match 04_tabnet exactly
# ─────────────────────────────────────────────
class TabNetWrapper:
    # Class-level attributes so they are present even on objects loaded from
    # an older pkl that pre-dates these additions to __init__.
    _estimator_type = "classifier"
    classes_        = np.array([0, 1])

    def __init__(self, scaler, model):
        self.scaler = scaler
        self.model  = model

    def fit(self, *args, **kwargs):
        # No-op: model is already trained. Required by sklearn's permutation_importance.
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
    matches = list(Path('outputs').rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"'{filename}' not found anywhere under outputs/. "
            "Run the prerequisite script first."
        )
    return str(sorted(matches, key=lambda p: p.stat().st_mtime)[-1])


# ─────────────────────────────────────────────
# STEP 6t.1 — Load data and models
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 6t.1 — Load data and models")
print("=" * 60)

df = pd.read_excel(DATA_FILE)
df['fiscal_year'] = pd.to_datetime(df['datadate']).dt.year

df_test = df[df['fiscal_year'].between(2023, 2024)].copy()
X_test  = df_test[FEATURES].values
y_test  = df_test[TARGET].values

print(f"\n  Test (2023–2024): {len(X_test):,} obs | {y_test.sum()} pos")

tabnet_path = find_output('model_tabnet.pkl')
with open(tabnet_path, 'rb') as f:
    tabnet_pipeline = pickle.load(f)
print(f"\n  Loaded: {tabnet_path}")


# ─────────────────────────────────────────────
# STEP 6t.2 — TABnet attention-based feature importance
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6t.2 — TABnet attention-based feature importance")
print("=" * 60)

# feature_importances_ is derived from the aggregated attention masks
# averaged across all decision steps — reflects selection frequency per feature
tabnet_model  = tabnet_pipeline.model
tabnet_scaler = tabnet_pipeline.scaler

X_test_sc = tabnet_scaler.transform(X_test).astype(np.float32)

# Trigger explain() to compute attention masks (also sets feature_importances_)
M_explain, masks = tabnet_model.explain(X_test_sc)
# M_explain: (n_samples, n_features) — average attention weight per sample per feature
tabnet_attention = M_explain.mean(axis=0)   # (n_features,) — mean over test set

print(f"\n  Attention-based importance (mean attention weight, descending):")
att_rank = np.argsort(tabnet_attention)[::-1]
for rank, i in enumerate(att_rank, 1):
    print(f"    {rank:2d}. {FEATURES[i]:<20}  {tabnet_attention[i]:.5f}")

# Also check model.feature_importances_ (consistent with attention)
if hasattr(tabnet_model, 'feature_importances_'):
    fi_attr = tabnet_model.feature_importances_
    print(f"\n  model.feature_importances_ (normalised, sums to 1):")
    fi_rank = np.argsort(fi_attr)[::-1]
    for rank, i in enumerate(fi_rank, 1):
        print(f"    {rank:2d}. {FEATURES[i]:<20}  {fi_attr[i]:.5f}")
    # Use model.feature_importances_ as primary (normalised, consistent)
    tabnet_main_imp = fi_attr
else:
    # Fallback: normalise attention weights
    tabnet_main_imp = tabnet_attention / tabnet_attention.sum()
    print("  (feature_importances_ not available — using normalised attention)")

tabnet_main_rank = np.argsort(tabnet_main_imp)[::-1]


# ─────────────────────────────────────────────
# STEP 6t.3 — TABnet permutation importance
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6t.3 — TABnet Permutation Importance (n_repeats=50, scoring=AP)")
print("=" * 60)
print("  This may take a few minutes ...")

perm_result = permutation_importance(
    tabnet_pipeline, X_test, y_test,
    scoring='average_precision',
    n_repeats=50,
    random_state=RANDOM_STATE,
    n_jobs=1,   # TabNet uses PyTorch internally — avoid multi-process conflicts
)

mean_perm = perm_result.importances_mean
std_perm  = perm_result.importances_std

print(f"\n  Permutation importance per feature (descending):")
perm_rank = np.argsort(mean_perm)[::-1]
for rank, i in enumerate(perm_rank, 1):
    print(f"    {rank:2d}. {FEATURES[i]:<20}  {mean_perm[i]:.5f} ± {std_perm[i]:.5f}")


# ─────────────────────────────────────────────
# STEP 6t.4 — Unified feature importance table (Table 12_tabnet)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6t.4 — Unified feature importance table (Table 12_tabnet)")
print("=" * 60)

# Normalise to [0, 1]
tabnet_att_norm  = tabnet_attention / tabnet_attention.max() if tabnet_attention.max() > 0 else tabnet_attention
perm_norm        = np.clip(mean_perm, 0, None)
perm_norm        = perm_norm / perm_norm.max() if perm_norm.max() > 0 else perm_norm
main_imp_norm    = tabnet_main_imp / tabnet_main_imp.max() if tabnet_main_imp.max() > 0 else tabnet_main_imp

# Load existing 3-model importances from Script 06 if available
existing_imp = None
try:
    existing_imp = pd.read_csv(find_output('table_12_feature_importance.csv'))
    print("  Loaded existing table_12_feature_importance.csv (3-model importances)")
except Exception:
    print("  table_12_feature_importance.csv not found — TABnet-only table will be saved")

rows = []
for i, feat in enumerate(FEATURES):
    row = {
        'Feature':                    feat,
        'Theory_Direction':           THEORY_DIRECTIONS[feat],
        'TABnet_AttentionImp':        round(float(tabnet_main_imp[i]),   6),
        'TABnet_AttentionImp_Norm':   round(float(main_imp_norm[i]),     4),
        'TABnet_Rank_Attention':      int(np.where(tabnet_main_rank == i)[0][0]) + 1,
        'TABnet_PermImp_Mean':        round(float(mean_perm[i]),         6),
        'TABnet_PermImp_Std':         round(float(std_perm[i]),          6),
        'TABnet_Rank_Perm':           int(np.where(perm_rank == i)[0][0]) + 1,
        'TABnet_PermImp_Norm':        round(float(perm_norm[i]),         4),
        'TABnet_RawAttention_Mean':   round(float(tabnet_attention[i]),  6),
    }
    # Merge existing 3-model importances if available
    if existing_imp is not None:
        match = existing_imp[existing_imp['Feature'] == feat]
        if not match.empty:
            m = match.iloc[0]
            row['XGBoost_MeanAbsSHAP']  = m.get('XGBoost_MeanAbsSHAP',  np.nan)
            row['XGBoost_Norm']         = m.get('XGBoost_Norm',          np.nan)
            row['XGBoost_Rank']         = m.get('XGBoost_Rank',          np.nan)
            row['LR_MeanAbsSHAP']       = m.get('LR_MeanAbsSHAP',        np.nan)
            row['LR_Norm']              = m.get('LR_Norm',               np.nan)
            row['LR_Rank']              = m.get('LR_Rank',               np.nan)
            row['MLP_PermImp_Mean']     = m.get('MLP_PermImp_Mean',      np.nan)
            row['MLP_Norm']             = m.get('MLP_Norm',              np.nan)
            row['MLP_Rank']             = m.get('MLP_Rank',              np.nan)
    rows.append(row)

imp_df = pd.DataFrame(rows).sort_values('TABnet_Rank_Attention')
imp_df.to_csv('outputs/table_12_tabnet_feature_importance.csv', index=False)
print("Table 12_tabnet saved: outputs/table_12_tabnet_feature_importance.csv")

print(f"\n  Feature importance (sorted by TABnet attention rank):")
print(f"  {'Rank':<5} {'Feature':<20} {'Theory':<7} "
      f"{'TABnet Att':<12} {'TABnet Perm':<12}")
print("  " + "-" * 62)
for _, row in imp_df.iterrows():
    print(f"  {row['TABnet_Rank_Attention']:<5} {row['Feature']:<20} "
          f"{row['Theory_Direction']:<7} "
          f"{row['TABnet_AttentionImp']:<12.5f} "
          f"{row['TABnet_PermImp_Mean']:<12.5f}")


# ─────────────────────────────────────────────
# STEP 6t.5 — Figure 14_tabnet: TABnet attention importance bar chart
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6t.5 — Figure 14_tabnet: Attention & Permutation importance")
print("=" * 60)

feat_order = imp_df['Feature'].tolist()   # sorted by TABnet attention rank
n_feats    = len(feat_order)
y_pos      = np.arange(n_feats)
bar_h      = 0.35

att_vals  = [float(imp_df.loc[imp_df['Feature'] == f, 'TABnet_AttentionImp_Norm'].values[0])
             for f in feat_order]
perm_vals = [float(imp_df.loc[imp_df['Feature'] == f, 'TABnet_PermImp_Norm'].values[0])
             for f in feat_order]

fig14, ax14 = plt.subplots(figsize=(10, 8))
ax14.barh(y_pos + bar_h/2, att_vals,  height=bar_h, color=COLORS['TABnet'],
          alpha=0.85, label='TABnet attention importance (normalised)')
ax14.barh(y_pos - bar_h/2, perm_vals, height=bar_h, color='#984ea3',
          alpha=0.85, label='TABnet permutation importance (normalised)')

ax14.set_yticks(y_pos)
ax14.set_yticklabels(feat_order, fontsize=9)
ax14.invert_yaxis()
ax14.set_xlabel('Normalised Importance (1.0 = most important)')
ax14.set_title(
    'Figure 14_tabnet: TABnet Feature Importance\n'
    'Attention-based vs Permutation Importance | Test Set (2023–2024)',
    fontsize=11
)
ax14.legend(fontsize=9, loc='lower right')
ax14.axvline(x=0, color='black', lw=0.6)
ax14.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/fig_14_tabnet_attention_feature_importance.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("Figure 14_tabnet saved: outputs/fig_14_tabnet_attention_feature_importance.png")


# ─────────────────────────────────────────────
# STEP 6t.6 — Figure 15_tabnet: 4-model importance comparison
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6t.6 — Figure 15_tabnet: Four-model importance comparison")
print("=" * 60)

if existing_imp is not None and 'XGBoost_Norm' in imp_df.columns:
    # Sort by XGBoost rank for consistent ordering with Script 06
    try:
        imp_df_sorted = imp_df.sort_values('XGBoost_Rank')
        feat_order_4  = imp_df_sorted['Feature'].tolist()
    except Exception:
        feat_order_4 = feat_order

    n_feats4 = len(feat_order_4)
    y_pos4   = np.arange(n_feats4)
    bar_h4   = 0.20

    def get_col(df, feat, col):
        v = df.loc[df['Feature'] == feat, col].values
        return float(v[0]) if len(v) > 0 and not pd.isna(v[0]) else 0.0

    xgb_vals4   = [get_col(imp_df_sorted, f, 'XGBoost_Norm')               for f in feat_order_4]
    lr_vals4    = [get_col(imp_df_sorted, f, 'LR_Norm')                     for f in feat_order_4]
    mlp_vals4   = [get_col(imp_df_sorted, f, 'MLP_Norm')                    for f in feat_order_4]
    tab_vals4   = [get_col(imp_df_sorted, f, 'TABnet_AttentionImp_Norm')    for f in feat_order_4]

    fig15, ax15 = plt.subplots(figsize=(11, 9))
    ax15.barh(y_pos4 + 1.5*bar_h4, xgb_vals4,  height=bar_h4, color=COLORS['XGBoost'],
              alpha=0.85, label='XGBoost (mean |SHAP|, normalised)')
    ax15.barh(y_pos4 + 0.5*bar_h4, lr_vals4,   height=bar_h4, color=COLORS['Logistic Regression'],
              alpha=0.85, label='Logistic Regression (mean |SHAP|, normalised)')
    ax15.barh(y_pos4 - 0.5*bar_h4, mlp_vals4,  height=bar_h4, color=COLORS['MLP'],
              alpha=0.85, label='MLP (permutation importance, normalised)')
    ax15.barh(y_pos4 - 1.5*bar_h4, tab_vals4,  height=bar_h4, color=COLORS['TABnet'],
              alpha=0.85, label='TABnet (attention importance, normalised)')

    ax15.set_yticks(y_pos4)
    ax15.set_yticklabels(feat_order_4, fontsize=9)
    ax15.invert_yaxis()
    ax15.set_xlabel('Normalised Feature Importance (1.0 = most important for each model)')
    ax15.set_title(
        'Figure 15_tabnet: Feature Importance Comparison — All Four Models\n'
        'Sorted by XGBoost SHAP rank | Test Set (2023–2024)',
        fontsize=11
    )
    ax15.legend(fontsize=8, loc='lower right')
    ax15.axvline(x=0, color='black', lw=0.6)
    ax15.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/fig_15_tabnet_feature_importance_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 15_tabnet saved: outputs/fig_15_tabnet_feature_importance_comparison.png")

    # Rank agreement: TABnet vs XGBoost
    try:
        from scipy.stats import spearmanr
        tab_ranks = imp_df_sorted['TABnet_Rank_Attention'].values
        xgb_ranks = imp_df_sorted['XGBoost_Rank'].values
        rho, pval = spearmanr(tab_ranks, xgb_ranks)
        print(f"\n  Spearman rank correlation (TABnet vs XGBoost): rho={rho:.3f}, p={pval:.3f}")
        print(f"  Interpretation: {'strong' if abs(rho)>=0.6 else 'moderate' if abs(rho)>=0.4 else 'weak'} "
              f"rank agreement")
    except Exception:
        pass
else:
    # Only TABnet data available — save TABnet-only figure
    fig15, ax15 = plt.subplots(figsize=(10, 8))
    ax15.barh(y_pos, att_vals, height=0.6, color=COLORS['TABnet'], alpha=0.85,
              label='TABnet attention importance (normalised)')
    ax15.set_yticks(y_pos)
    ax15.set_yticklabels(feat_order, fontsize=9)
    ax15.invert_yaxis()
    ax15.set_xlabel('Normalised Importance')
    ax15.set_title('Figure 15_tabnet: TABnet Feature Importance (Attention)\n'
                   'Test Set (2023–2024)', fontsize=11)
    ax15.legend(fontsize=9)
    ax15.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/fig_15_tabnet_feature_importance_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 15_tabnet (TABnet-only) saved: "
          "outputs/fig_15_tabnet_feature_importance_comparison.png")
    print("  NOTE: Run Script 06 first to include all 4 models in comparison figure.")


# ─────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Script 06_tabnet complete. All outputs saved to outputs/")
print("=" * 60)

print(f"\n  Top 5 features by TABnet attention rank:")
for rank in range(5):
    feat = feat_order[rank]
    row  = imp_df[imp_df['Feature'] == feat].iloc[0]
    print(f"    {rank+1}. {feat:<20}  "
          f"AttImp={row['TABnet_AttentionImp']:.5f}  "
          f"PermImp={row['TABnet_PermImp_Mean']:.5f}  "
          f"theory={row['Theory_Direction']}")

print(f"\n  Outputs:")
print(f"    table_12_tabnet_feature_importance.csv")
print(f"    fig_14_tabnet_attention_feature_importance.png")
print(f"    fig_15_tabnet_feature_importance_comparison.png")
print(f"\n  No-skill PR-AUC baseline: {NO_SKILL_PRAUC}")
