import os
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    f1_score, precision_score, recall_score,
    confusion_matrix,
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
ID_COLS        = ['gvkey', 'conm', 'tic', 'datadate', 'cusip', 'cik']
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
}
THRESHOLD_GRID = np.arange(0.001, 0.151, 0.001)


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


def find_output(filename):
    matches = list(Path('outputs').rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"'{filename}' not found anywhere under outputs/. "
            "Run the prerequisite script first."
        )
    return str(sorted(matches, key=lambda p: p.stat().st_mtime)[-1])


# ─────────────────────────────────────────────
# STEP 7.1 — Load data, models, and predictions
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 7.1 — Load data, models, and predictions")
print("=" * 60)

df = pd.read_excel(DATA_FILE)
df['fiscal_year'] = pd.to_datetime(df['datadate']).dt.year
df['sic_division'] = df['sic'].apply(sic_to_division)

df_val  = df[df['fiscal_year'] == 2022].copy().reset_index(drop=True)
df_test = df[df['fiscal_year'].between(2023, 2024)].copy().reset_index(drop=True)

X_val  = df_val[FEATURES].values;  y_val  = df_val[TARGET].values
X_test = df_test[FEATURES].values; y_test = df_test[TARGET].values

print(f"\n  Val  (2022)     : {len(y_val):,} obs | {y_val.sum()} pos")
print(f"  Test (2023–2024): {len(y_test):,} obs | {y_test.sum()} pos")

lr_path  = find_output('model_baseline_logistic.pkl')
xgb_path = find_output('model_xgb.pkl')
mlp_path = find_output('model_mlp_pipeline.pkl')

with open(lr_path,  'rb') as f: lr_pipeline  = pickle.load(f)
with open(xgb_path, 'rb') as f: model_xgb    = pickle.load(f)
with open(mlp_path, 'rb') as f: mlp_pipeline = pickle.load(f)

print(f"\n  Loaded: {lr_path}")
print(f"  Loaded: {xgb_path}")
print(f"  Loaded: {mlp_path}")

# All probabilities (test set)
test_probas = {
    'Logistic Regression': lr_pipeline.predict_proba(X_test)[:, 1],
    'XGBoost':             model_xgb.predict_proba(X_test)[:, 1],
    'MLP':                 mlp_pipeline.predict_proba(X_test)[:, 1],
}
val_probas = {
    'Logistic Regression': lr_pipeline.predict_proba(X_val)[:, 1],
    'XGBoost':             model_xgb.predict_proba(X_val)[:, 1],
    'MLP':                 mlp_pipeline.predict_proba(X_val)[:, 1],
}

# Identify best model and select threshold from val set (argmax F1)
try:
    metrics_df = pd.read_csv(find_output('table_10_test_metrics.csv'))
    best_model = metrics_df.loc[metrics_df['PR_AUC'].idxmax(), 'Model']
    best_prauc = metrics_df['PR_AUC'].max()
except Exception:
    best_model = 'XGBoost'
    best_prauc = np.nan

print(f"\n  Best model: {best_model} (Test PR-AUC = {best_prauc:.4f}, "
      f"lift = {best_prauc / NO_SKILL_PRAUC:.2f}x)")

# Best-model threshold: argmax F1 on validation set (same protocol as Script 05)
val_proba_best = val_probas[best_model]
f1_scores_val  = [f1_score(y_val, (val_proba_best >= t).astype(int), zero_division=0)
                  for t in THRESHOLD_GRID]
best_thresh = float(THRESHOLD_GRID[np.argmax(f1_scores_val)])
print(f"  Optimal threshold (val argmax F1): {best_thresh:.3f}")

proba_best = test_probas[best_model]
preds_best = (proba_best >= best_thresh).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, preds_best).ravel()
print(f"\n  Test set at threshold {best_thresh:.3f}: "
      f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")


# ─────────────────────────────────────────────
# STEP 7.2 — Sector-level analysis
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7.2 — Sector-level analysis (SIC divisions, test set)")
print("=" * 60)

df_test['proba_best']  = proba_best
df_test['pred_best']   = preds_best
df_test['error_type']  = 'TN'
df_test.loc[(df_test['pred_best'] == 1) & (df_test[TARGET] == 1), 'error_type'] = 'TP'
df_test.loc[(df_test['pred_best'] == 1) & (df_test[TARGET] == 0), 'error_type'] = 'FP'
df_test.loc[(df_test['pred_best'] == 0) & (df_test[TARGET] == 1), 'error_type'] = 'FN'

N_test    = len(y_test)
top10_n   = max(1, int(np.round(0.10 * N_test)))

sector_rows = []
for div, grp in df_test.groupby('sic_division', sort=False):
    n_firms   = len(grp)
    n_pos     = int(grp[TARGET].sum())
    base_rate = n_pos / n_firms if n_firms > 0 else 0

    # Sector-level mean probability
    mean_prob = grp['proba_best'].mean()

    # Precision@10% within sector
    n_top_sec = max(1, int(np.round(0.10 * n_firms)))
    sorted_sec = grp.sort_values('proba_best', ascending=False)
    pos_top10_sec = int(sorted_sec.iloc[:n_top_sec][TARGET].sum())
    exp_top10_sec = n_top_sec * base_rate
    lift_sec = (pos_top10_sec / n_top_sec) / base_rate if base_rate > 0 else np.nan

    # PR-AUC only if >=2 positives and <n_firms positives
    if 2 <= n_pos < n_firms:
        try:
            pr_auc_sec = average_precision_score(grp[TARGET], grp['proba_best'])
        except Exception:
            pr_auc_sec = np.nan
    else:
        pr_auc_sec = np.nan

    sector_rows.append({
        'SIC_Division':       div,
        'N_Firms':            n_firms,
        'N_Targets':          n_pos,
        'Base_Rate_pct':      round(base_rate * 100, 2),
        'Mean_PredProb':      round(mean_prob, 4),
        'Pos_in_Top10pct':    pos_top10_sec,
        'Top10pct_N':         n_top_sec,
        'Lift_at_10pct':      round(lift_sec, 3) if not np.isnan(lift_sec) else np.nan,
        'PR_AUC':             round(pr_auc_sec, 4) if not np.isnan(pr_auc_sec) else np.nan,
        'N_TP':               int((grp['error_type'] == 'TP').sum()),
        'N_FP':               int((grp['error_type'] == 'FP').sum()),
        'N_FN':               int((grp['error_type'] == 'FN').sum()),
    })

sector_df = (pd.DataFrame(sector_rows)
             .sort_values('N_Targets', ascending=False)
             .reset_index(drop=True))

sector_df.to_csv('outputs/table_13_sector_analysis.csv', index=False)
print("Table 13 saved: outputs/table_13_sector_analysis.csv\n")

print(f"  {'Division':<30} {'N_firms':>8} {'N_pos':>6} {'Rate%':>6} "
      f"{'Lift@10%':>10} {'PR-AUC':>8} {'TP':>4} {'FP':>5} {'FN':>4}")
print("  " + "-" * 85)
for _, row in sector_df.iterrows():
    lift_s = f"{row['Lift_at_10pct']:.2f}x" if not pd.isna(row['Lift_at_10pct']) else '  N/A'
    pr_s   = f"{row['PR_AUC']:.4f}"          if not pd.isna(row['PR_AUC'])         else '   N/A'
    print(f"  {row['SIC_Division']:<30} {row['N_Firms']:>8,} {row['N_Targets']:>6} "
          f"{row['Base_Rate_pct']:>6.2f} {lift_s:>10} {pr_s:>8} "
          f"{row['N_TP']:>4} {row['N_FP']:>5} {row['N_FN']:>4}")

# ── Figure 17: Sector analysis ────────────────────────────────────────────
fig17, axes17 = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Target rate by sector (bar)
ax_a = axes17[0]
plot_df = sector_df[sector_df['N_Targets'] >= 1].copy()
ax_a.barh(plot_df['SIC_Division'], plot_df['Base_Rate_pct'],
          color='#2c7bb6', alpha=0.75)
ax_a.axvline(x=NO_SKILL_PRAUC * 100, color='gray', linestyle='--', lw=1.5,
             label=f'Overall base rate ({NO_SKILL_PRAUC*100:.2f}%)')
ax_a.set_xlabel('M&A Target Base Rate (%)')
ax_a.set_title('Panel A: M&A Target Rate by SIC Division\n(Test Set 2023–2024)', fontsize=10)
ax_a.legend(fontsize=8)
ax_a.invert_yaxis()

# Add counts as text
for i, (_, row) in enumerate(plot_df.iterrows()):
    ax_a.text(row['Base_Rate_pct'] + 0.05, i,
              f"n={row['N_Targets']}/{row['N_Firms']}",
              va='center', fontsize=7.5, color='black')

# Panel B: Lift@10% by sector (bubble: size = N_firms)
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
plt.savefig('outputs/fig_17_sector_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nFigure 17 saved: outputs/fig_17_sector_analysis.png")


# ─────────────────────────────────────────────
# STEP 7.3 — Error characterisation (TP / FP / FN)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7.3 — Error characterisation: feature profiles for TP / FP / FN")
print("=" * 60)

groups = {'TP': 'True Positive', 'FP': 'False Positive', 'FN': 'False Negative'}

# Feature mean and median for each error group + overall positive class
error_rows = []
pos_mask = df_test[TARGET] == 1

for feat in FEATURES:
    row = {'Feature': feat}
    for grp_code, grp_label in groups.items():
        mask = df_test['error_type'] == grp_code
        vals = df_test.loc[mask, feat].dropna()
        row[f'{grp_code}_mean']   = round(vals.mean(),   4) if len(vals) > 0 else np.nan
        row[f'{grp_code}_median'] = round(vals.median(), 4) if len(vals) > 0 else np.nan
        row[f'{grp_code}_n']      = len(vals)

    # All positives (TP + FN)
    pos_vals = df_test.loc[pos_mask, feat].dropna()
    row['AllPos_mean']   = round(pos_vals.mean(),   4)
    row['AllPos_median'] = round(pos_vals.median(), 4)
    row['AllPos_n']      = len(pos_vals)

    # All test obs (context)
    all_vals = df_test[feat].dropna()
    row['Overall_mean']   = round(all_vals.mean(),   4)
    row['Overall_median'] = round(all_vals.median(), 4)

    error_rows.append(row)

error_df = pd.DataFrame(error_rows)
error_df.to_csv('outputs/table_14_error_profile.csv', index=False)
print("Table 14 saved: outputs/table_14_error_profile.csv")

print(f"\n  Group sizes:  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
print(f"\n  Feature profile (means) by error group:")
print(f"  {'Feature':<20} {'TP mean':>10} {'FP mean':>10} {'FN mean':>10} "
      f"{'AllPos mean':>12} {'Overall mean':>13}")
print("  " + "-" * 80)
for _, row in error_df.iterrows():
    print(f"  {row['Feature']:<20} "
          f"{row['TP_mean']:>10.4f} {row['FP_mean']:>10.4f} {row['FN_mean']:>10.4f} "
          f"{row['AllPos_mean']:>12.4f} {row['Overall_mean']:>13.4f}")

# ── Figure 18: Error characterisation heatmap ────────────────────────────
# Z-score feature means within each error group relative to overall mean / std
print("\n  Generating Figure 18: error characterisation ...")

overall_means = error_df.set_index('Feature')['Overall_mean']
overall_stds  = {f: df_test[f].std() for f in FEATURES}

heat_data = {}
for grp_code in ['TP', 'FP', 'FN']:
    col = error_df.set_index('Feature')[f'{grp_code}_mean']
    z   = (col - overall_means) / pd.Series(overall_stds)
    heat_data[grp_code] = z

heat_df = pd.DataFrame(heat_data)   # rows = features, cols = error groups

fig18, axes18 = plt.subplots(1, 2, figsize=(14, 7),
                              gridspec_kw={'width_ratios': [1, 2]})

# Panel A: heatmap of z-scores
ax_h = axes18[0]
sns.heatmap(
    heat_df,
    ax=ax_h,
    cmap='RdBu_r',
    center=0,
    vmin=-3, vmax=3,
    annot=True, fmt='.2f',
    linewidths=0.5,
    cbar_kws={'label': 'Z-score vs overall mean'},
)
ax_h.set_title(f'Panel A: Feature Z-scores by Error Group\n'
               f'(best model: {best_model}, threshold={best_thresh:.3f})', fontsize=10)
ax_h.set_xlabel('Error Group')
ax_h.set_ylabel('Feature')

# Panel B: TP vs FN means if FN > 0, else fall back to TP vs FP means
ax_bar = axes18[1]
tp_means = error_df.set_index('Feature')['TP_mean']

if fn > 0:
    compare_means = error_df.set_index('Feature')['FN_mean']
    compare_label = f'False Negative (n={fn})'
    panel_b_title = ('Panel B: TP vs FN Feature Means\n'
                     '(sorted by |TP mean - FN mean|, largest diff at top)')
else:
    # FN=0 at this threshold — compare TP vs FP to show what model over-predicts
    compare_means = error_df.set_index('Feature')['FP_mean']
    compare_label = f'False Positive (n={fp})'
    panel_b_title = ('Panel B: TP vs FP Feature Means\n'
                     '(FN=0 at this threshold; showing over-prediction profile)')
    print("  NOTE: FN=0 at selected threshold — Panel B shows TP vs FP profile.")

diff_abs   = (tp_means - compare_means).abs().sort_values(ascending=False)
feat_order = diff_abs.index.tolist()

y_pos = np.arange(len(feat_order))
bar_h = 0.35
tp_vals      = [tp_means[f]      for f in feat_order]
compare_vals = [compare_means[f] for f in feat_order]

ax_bar.barh(y_pos + bar_h / 2, tp_vals,      height=bar_h,
            color='#d7191c', alpha=0.8, label=f'True Positive (n={tp})')
ax_bar.barh(y_pos - bar_h / 2, compare_vals, height=bar_h,
            color='#2c7bb6', alpha=0.8, label=compare_label)
ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(feat_order, fontsize=9)
ax_bar.invert_yaxis()
ax_bar.axvline(0, color='black', lw=0.6)
ax_bar.set_xlabel('Feature Mean Value')
ax_bar.set_title(panel_b_title, fontsize=10)
ax_bar.legend(fontsize=9)
ax_bar.grid(axis='x', alpha=0.3)

fig18.suptitle('Figure 18: Error Characterisation — Test Set (2023–2024)', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/fig_18_error_characterization.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 18 saved: outputs/fig_18_error_characterization.png")


# ─────────────────────────────────────────────
# STEP 7.4 — Top-50 predicted M&A targets (Table 15)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7.4 — Top-50 predicted M&A targets (Table 15)")
print("=" * 60)

# Build a full prediction frame with firm identifiers
id_cols_present = [c for c in ['gvkey', 'conm', 'tic', 'sic', 'sic_division'] if c in df_test.columns]
top50_df = df_test[id_cols_present + [TARGET]].copy()
top50_df['Pred_Prob']   = proba_best
top50_df['Actual_Label'] = y_test
top50_df['Rank']        = top50_df['Pred_Prob'].rank(ascending=False, method='first').astype(int)
top50_df = top50_df.sort_values('Rank').head(50).reset_index(drop=True)
top50_df['Correct'] = (top50_df['Pred_Prob'] >= best_thresh).astype(int) == top50_df['Actual_Label']

top50_df.to_csv('outputs/table_15_top50_predicted_targets.csv', index=False)
print("Table 15 saved: outputs/table_15_top50_predicted_targets.csv")

n_tp_top50 = int(top50_df['Actual_Label'].sum())
print(f"\n  Top-50 actual M&A targets captured: {n_tp_top50} / {int(y_test.sum())} "
      f"({n_tp_top50/y_test.sum()*100:.1f}% of all test targets)")
print(f"\n  Top-20 predicted firms:")
print(f"  {'Rank':>5} {'Name':<35} {'Ticker':<8} {'Sector':<25} {'P(target)':>10} {'Actual':>8}")
print("  " + "-" * 95)
for _, row in top50_df.head(20).iterrows():
    conm_str = str(row.get('conm', 'N/A'))[:34]
    tic_str  = str(row.get('tic',  'N/A'))[:7]
    sec_str  = str(row.get('sic_division', 'N/A'))[:24]
    print(f"  {row['Rank']:>5} {conm_str:<35} {tic_str:<8} {sec_str:<25} "
          f"{row['Pred_Prob']:>10.4f} {'TARGET' if row['Actual_Label'] == 1 else '':>8}")


# ─────────────────────────────────────────────
# STEP 7.5 — Calibration (Figure 19)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7.5 — Probability calibration (Figure 19)")
print("=" * 60)

# sklearn calibration_curve: n_bins chosen so most bins have enough samples
N_BINS = 10

fig19, axes19 = plt.subplots(1, 2, figsize=(13, 5))
ax_cal, ax_hist = axes19

print(f"\n  Calibration (fraction positives per predicted-probability bin, n_bins={N_BINS}):")
for name, proba in test_probas.items():
    frac_pos, mean_pred = calibration_curve(
        y_test, proba, n_bins=N_BINS, strategy='quantile'
    )
    ax_cal.plot(mean_pred, frac_pos, marker='o', lw=2,
                color=COLORS[name], label=name, ms=5)

    # Expected calibration error (ECE)
    counts_per_bin, bin_edges = np.histogram(proba, bins=N_BINS)
    ece = 0.0
    for b_idx in range(N_BINS):
        lo, hi = bin_edges[b_idx], bin_edges[b_idx + 1]
        mask_b = (proba >= lo) & (proba < hi)
        if mask_b.sum() > 0:
            acc_b = y_test[mask_b].mean()
            conf_b = proba[mask_b].mean()
            ece += mask_b.sum() / len(proba) * abs(acc_b - conf_b)
    print(f"  {name:<25} ECE = {ece:.4f}")

ax_cal.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
ax_cal.set_xlabel('Mean Predicted Probability')
ax_cal.set_ylabel('Fraction of Positives')
ax_cal.set_title('Panel A: Reliability Diagram\n(quantile binning, n_bins=10)', fontsize=10)
ax_cal.legend(fontsize=9)
ax_cal.grid(alpha=0.3)

# Histogram of predicted probabilities (log scale y)
for name, proba in test_probas.items():
    ax_hist.hist(proba, bins=40, alpha=0.4, color=COLORS[name], label=name,
                 density=True)
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
plt.savefig('outputs/fig_19_calibration.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 19 saved: outputs/fig_19_calibration.png")


# ─────────────────────────────────────────────
# STEP 7.6 — Threshold sensitivity (Figure 20)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7.6 — Threshold sensitivity (Figure 20)")
print("=" * 60)

fig20, axes20 = plt.subplots(1, 2, figsize=(14, 5))

for name, proba in test_probas.items():
    f1s, precs, recs, lifts = [], [], [], []
    for t in THRESHOLD_GRID:
        preds_t = (proba >= t).astype(int)
        f1s.append(f1_score(y_test, preds_t, zero_division=0))
        precs.append(precision_score(y_test, preds_t, zero_division=0))
        recs.append(recall_score(y_test, preds_t, zero_division=0))
        # Lift: fraction of positives among predicted positives / base rate
        n_pred = preds_t.sum()
        if n_pred > 0:
            lifts.append((y_test[preds_t == 1].sum() / n_pred) / NO_SKILL_PRAUC)
        else:
            lifts.append(0)

    # Left panel: F1 / Precision / Recall
    axes20[0].plot(THRESHOLD_GRID, f1s,   color=COLORS[name], lw=2,
                   linestyle='-',  label=f'{name} F1')
    axes20[0].plot(THRESHOLD_GRID, precs, color=COLORS[name], lw=1.2,
                   linestyle='--', alpha=0.7)
    axes20[0].plot(THRESHOLD_GRID, recs,  color=COLORS[name], lw=1.2,
                   linestyle=':', alpha=0.7)

    # Right panel: lift vs threshold
    axes20[1].plot(THRESHOLD_GRID, lifts, color=COLORS[name], lw=2, label=name)

# Annotations
axes20[0].axvline(x=best_thresh, color='black', linestyle='--', lw=1.2,
                  label=f'{best_model} thresh ({best_thresh:.3f})')
axes20[0].set_xlabel('Threshold')
axes20[0].set_ylabel('Score')
axes20[0].set_title('Panel A: F1 (solid), Precision (dashed),\nRecall (dotted) vs Threshold',
                    fontsize=10)
axes20[0].legend(fontsize=8, loc='center right')
axes20[0].grid(alpha=0.3)
axes20[0].set_xlim(THRESHOLD_GRID[0], THRESHOLD_GRID[-1])
axes20[0].set_ylim(0, 1.02)

axes20[1].axhline(y=1.0, color='gray', linestyle='--', lw=1, label='No-skill (lift=1.0x)')
axes20[1].axvline(x=best_thresh, color='black', linestyle='--', lw=1.2,
                  label=f'{best_model} thresh ({best_thresh:.3f})')
axes20[1].set_xlabel('Threshold')
axes20[1].set_ylabel('Precision Lift = Precision / Base Rate')
axes20[1].set_title('Panel B: Precision Lift vs Threshold\n'
                    '(lift=1.0x is random screening)', fontsize=10)
axes20[1].legend(fontsize=8, loc='upper left')
axes20[1].grid(alpha=0.3)
axes20[1].set_xlim(THRESHOLD_GRID[0], THRESHOLD_GRID[-1])
axes20[1].set_ylim(bottom=0)

fig20.suptitle('Figure 20: Threshold Sensitivity — Test Set (2023–2024)', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/fig_20_threshold_sensitivity.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 20 saved: outputs/fig_20_threshold_sensitivity.png")


# ─────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Script 07 complete. All outputs saved to outputs/")
print("=" * 60)

print(f"\n  Best model: {best_model} | Test PR-AUC = {best_prauc:.4f} "
      f"| Lift = {best_prauc / NO_SKILL_PRAUC:.2f}x over no-skill")
print(f"  Threshold: {best_thresh:.3f} | TP={tp}  FP={fp}  FN={fn}  TN={tn}")

# Strongest sectors
best_sec = sector_df.dropna(subset=['Lift_at_10pct']).nlargest(3, 'Lift_at_10pct')
worst_sec = sector_df.dropna(subset=['Lift_at_10pct']).nsmallest(2, 'Lift_at_10pct')
print(f"\n  Highest model lift by sector:")
for _, row in best_sec.iterrows():
    print(f"    {row['SIC_Division']:<30} Lift@10% = {row['Lift_at_10pct']:.2f}x "
          f"(base rate = {row['Base_Rate_pct']:.2f}%)")
print(f"\n  Lowest model lift by sector:")
for _, row in worst_sec.iterrows():
    print(f"    {row['SIC_Division']:<30} Lift@10% = {row['Lift_at_10pct']:.2f}x "
          f"(base rate = {row['Base_Rate_pct']:.2f}%)")

print(f"\n  Top-50 captures {n_tp_top50}/{int(y_test.sum())} targets "
      f"({n_tp_top50 / y_test.sum() * 100:.1f}% recall at rank <= 50)")

print(f"\n  Outputs:")
print(f"    table_13_sector_analysis.csv")
print(f"    table_14_error_profile.csv")
print(f"    table_15_top50_predicted_targets.csv")
print(f"    fig_17_sector_analysis.png")
print(f"    fig_18_error_characterization.png")
print(f"    fig_19_calibration.png")
print(f"    fig_20_threshold_sensitivity.png")
print(f"\n  No-skill PR-AUC baseline: {NO_SKILL_PRAUC}")
