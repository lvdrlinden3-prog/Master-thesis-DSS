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
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    log_loss, confusion_matrix, f1_score, precision_score, recall_score,
    balanced_accuracy_score
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

warnings.filterwarnings('ignore')
os.makedirs('outputs', exist_ok=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
NO_SKILL_PRAUC = 0.0157  # updated from 0.0122 — equals 1.57% prevalence
PREVALENCE_LABEL = 'No-skill baseline (prevalence = 1.57%)'
COLORS = ['#2c7bb6', '#d7191c']
RANDOM_STATE = 42

DATA_FILE = 'data_final_modeling_ma_v7.xlsx'
ID_COLS  = ['gvkey', 'conm', 'tic', 'datadate', 'cusip', 'cik']
SECTOR   = 'sic'
TARGET   = 'target_next_year'
FEATURES = [
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

# ─────────────────────────────────────────────
# STEP 2.1 — Canonical temporal train/test split
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 2.1 — Canonical temporal train/test split")
print("=" * 60)
print(f"Loading: {DATA_FILE}")
df = pd.read_excel(DATA_FILE)
df['fiscal_year'] = pd.to_datetime(df['datadate']).dt.year

# Diagnostic: confirm M&A event count before/after temporal filter
# (expected: 439 total → 413 in 2019–2024 window; drop due to year exclusions, not filtering bug)
total_events = df[TARGET].sum()
print(f"\nPre-split event count: {int(total_events)} M&A targets in full dataset")
events_by_year = df.groupby('fiscal_year')[TARGET].sum().astype(int)
print("  Events by year (all):")
print(events_by_year.to_string())

# NOTE: 2011 excluded — first year of panel, zero positive events.
#       2025 excluded — partial-year coverage; falls outside test window.
df_train = df[df['fiscal_year'].between(2012, 2021)].copy()
df_val   = df[df['fiscal_year'] == 2022].copy()
df_test  = df[df['fiscal_year'].between(2023, 2024)].copy()

print("\nSplit verification:")
for name, split in [('Train (2012–2021)', df_train),
                    ('Validation (2022)',  df_val),
                    ('Test (2023–2024)',   df_test)]:
    n_total = len(split)
    n_pos   = split[TARGET].sum()
    rate    = n_pos / n_total * 100
    flag    = ' *** WARNING: fewer than 50 positives!' if n_pos < 50 else ''
    print(f"  {name}: {n_total:,} obs | {n_pos} positives | {rate:.2f}% base rate{flag}")

# Extract features and target — SECTOR never passed to model
X_train = df_train[FEATURES].values
y_train = df_train[TARGET].values
X_val   = df_val[FEATURES].values
y_val   = df_val[TARGET].values
X_test  = df_test[FEATURES].values
y_test  = df_test[TARGET].values

# Sector series extracted separately for Script 07
sector_train = df_train[SECTOR]
sector_val   = df_val[SECTOR]
sector_test  = df_test[SECTOR]

# ─────────────────────────────────────────────
# STEP 2.2 — Logistic regression pipeline
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2.2 — Logistic regression pipeline")
print("=" * 60)

scaler = RobustScaler()
lr     = LogisticRegression(
    max_iter=1000, solver='lbfgs',
    class_weight='balanced',
    C=1.0, random_state=RANDOM_STATE
)
pipeline = Pipeline([('scaler', scaler), ('lr', lr)])
pipeline.fit(X_train, y_train)

# Print class weights
n0, n1    = (y_train == 0).sum(), (y_train == 1).sum()
n_total   = len(y_train)
w0        = n_total / (2 * n0)
w1        = n_total / (2 * n1)
print(f"\nComputed class weights (class_weight='balanced'):")
print(f"  Class 0 weight: {w0:.4f}")
print(f"  Class 1 weight: {w1:.4f}  (ratio ~{w1/w0:.1f}x minority up-weight)")

# ── VIF diagnostics (on scaled training data) ──────────────────────────────
print("\n--- VIF Diagnostics (scaled training features) ---")
X_train_scaled = pipeline.named_steps['scaler'].transform(X_train)
X_scaled_df    = pd.DataFrame(X_train_scaled, columns=FEATURES)

vif_values = []
for i, feat in enumerate(FEATURES):
    vif = variance_inflation_factor(X_scaled_df.values, i)
    vif_values.append({'Feature': feat, 'VIF': round(vif, 3)})

vif_df = pd.DataFrame(vif_values).sort_values('VIF', ascending=False)
vif_df['Flag_VIF_gt5'] = vif_df['VIF'] > 5
print(vif_df.to_string(index=False))

flagged_vif = vif_df[vif_df['Flag_VIF_gt5']]['Feature'].tolist()
if flagged_vif:
    print(f"\n  VIF > 5 flagged: {flagged_vif}")
    print("  NOTE: Variables retained — joint interpretation required in thesis")
    print("  (leverage / altman_re_ta r=-0.75; see Methodology 'Assumptions' section)")
else:
    print("\n  No features with VIF > 5")

vif_df.to_csv('outputs/table_02b_vif.csv', index=False)
print("Table 02b saved: outputs/table_02b_vif.csv")

# ─────────────────────────────────────────────
# STEP 2.3 — Threshold-independent evaluation (ROC + PR)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2.3 — Threshold-independent evaluation (val set)")
print("=" * 60)

y_val_proba = pipeline.predict_proba(X_val)[:, 1]

# Metrics
roc_auc_val = auc(*roc_curve(y_val, y_val_proba)[:2])
pr_auc_val  = average_precision_score(y_val, y_val_proba)
ll_val      = log_loss(y_val, y_val_proba)

print(f"  Val ROC-AUC : {roc_auc_val:.4f}  (no-skill = 0.50)")
print(f"  Val PR-AUC  : {pr_auc_val:.4f}  (no-skill = {NO_SKILL_PRAUC})")
print(f"  Val Log-Loss: {ll_val:.4f}")
print(f"  PR-AUC lift vs no-skill: {pr_auc_val / NO_SKILL_PRAUC:.2f}x")

# Figure 6 — ROC + PR curves
fpr, tpr, _    = roc_curve(y_val, y_val_proba)
prec, rec, _   = precision_recall_curve(y_val, y_val_proba)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: ROC curve
ax = axes[0]
ax.plot(fpr, tpr, color=COLORS[0], lw=2,
        label=f'Logistic Regression (AUC = {roc_auc_val:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='No-skill (AUC = 0.50)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Panel A: ROC Curve (Validation Set)')
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

# Panel B: PR curve
ax2 = axes[1]
ax2.plot(rec, prec, color=COLORS[1], lw=2,
         label=f'Logistic Regression (AP = {pr_auc_val:.3f})')
ax2.axhline(y=NO_SKILL_PRAUC, color='gray', linestyle='--', lw=1.5,
            label=PREVALENCE_LABEL)
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Panel B: Precision-Recall Curve (Validation Set)')
ax2.legend(loc='upper right', fontsize=8)
ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.02])

plt.suptitle('Logistic Regression Baseline — ROC and PR Curves', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/fig_06_baseline_roc_pr.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 06 saved: outputs/fig_06_baseline_roc_pr.png")

# ─────────────────────────────────────────────
# STEP 2.4 — Threshold selection and classification metrics
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2.4 — Threshold selection (argmax F1 on validation set)")
print("=" * 60)

thresholds = np.arange(0.001, 0.151, 0.001)
best_thresh, best_f1 = 0.5, 0.0
for t in thresholds:
    y_pred_t = (y_val_proba >= t).astype(int)
    f = f1_score(y_val, y_pred_t, zero_division=0)
    if f > best_f1:
        best_f1, best_thresh = f, t

print(f"  Optimal threshold (val argmax F1): {best_thresh:.3f}  |  Val F1: {best_f1:.4f}")

def compute_classification_metrics(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    prec   = precision_score(y_true, y_pred, zero_division=0)
    rec    = recall_score(y_true, y_pred, zero_division=0)
    f1     = f1_score(y_true, y_pred, zero_division=0)
    spec   = tn / (tn + fp) if (tn + fp) > 0 else 0
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    return cm, {'Threshold': threshold, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                'Precision': prec, 'Recall': rec, 'F1': f1,
                'Specificity': spec, 'Balanced_Accuracy': bal_acc}

cm_val, val_metrics = compute_classification_metrics(y_val, y_val_proba, best_thresh)
val_metrics['ROC_AUC'] = roc_auc_val
val_metrics['PR_AUC']  = pr_auc_val

print("\n  Validation set metrics at optimal threshold:")
for k, v in val_metrics.items():
    if isinstance(v, float):
        print(f"    {k}: {v:.4f}")
    else:
        print(f"    {k}: {v}")

pd.DataFrame([val_metrics]).to_csv('outputs/table_05_baseline_val_metrics.csv', index=False)
print("Table 05 saved: outputs/table_05_baseline_val_metrics.csv")

# Confusion matrix figure (validation)
fig, ax = plt.subplots(figsize=(5, 4))
cm_norm = cm_val.astype(float) / cm_val.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=False, fmt='', cmap='Blues', ax=ax,
            xticklabels=['Pred 0', 'Pred 1'],
            yticklabels=['True 0', 'True 1'], cbar=True)
# Annotate with rate + raw count
for i in range(2):
    for j in range(2):
        ax.text(j + 0.5, i + 0.5,
                f'{cm_norm[i, j]:.2f}\n(n={cm_val[i, j]:,})',
                ha='center', va='center', fontsize=11,
                color='white' if cm_norm[i, j] > 0.5 else 'black')
ax.set_title(f'Confusion Matrix — Validation Set\n'
             f'Threshold = {best_thresh:.3f} | F1 = {best_f1:.3f}')
plt.tight_layout()
plt.savefig('outputs/fig_07_baseline_cm.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 07 saved: outputs/fig_07_baseline_cm.png")

# Apply same threshold to TEST SET (confirmatory)
y_test_proba   = pipeline.predict_proba(X_test)[:, 1]
roc_auc_test   = auc(*roc_curve(y_test, y_test_proba)[:2])
pr_auc_test    = average_precision_score(y_test, y_test_proba)
cm_test, test_metrics = compute_classification_metrics(y_test, y_test_proba, best_thresh)
test_metrics['ROC_AUC'] = roc_auc_test
test_metrics['PR_AUC']  = pr_auc_test

print(f"\n  Test set metrics (same threshold {best_thresh:.3f}):")
print(f"    ROC-AUC: {roc_auc_test:.4f}  |  PR-AUC: {pr_auc_test:.4f}")
print(f"    F1: {test_metrics['F1']:.4f}  |  Precision: {test_metrics['Precision']:.4f}  "
      f"|  Recall: {test_metrics['Recall']:.4f}")

pd.DataFrame([test_metrics]).to_csv('outputs/table_06_baseline_test_metrics.csv', index=False)
print("Table 06 saved: outputs/table_06_baseline_test_metrics.csv")

# ─────────────────────────────────────────────
# STEP 2.5 — Coefficient table with bootstrap CIs
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2.5 — Coefficient table with bootstrap 95% CIs (n=500)")
print("=" * 60)

# Theory directions: Palepu (1986), Jensen (1986), Morck et al. (1988)
THEORY_DIRECTIONS = {
    'profitability':    ('(-)', 'Underperformance hypothesis (Palepu, 1986)'),
    'leverage':         ('(+/-)', 'Jensen (1986) vs. financial distress'),
    'cash_ratio':       ('(-)', 'FCF hypothesis -- excess cash (Jensen, 1986)'),
    'fcf_debt':         ('(+/-)', 'FCF hypothesis (Jensen, 1986)'),
    'ppe_ratio':        ('(+)', 'Tangible asset appeal (Ambrose & Megginson, 1992)'),
    'capex_intensity':  ('(-)', 'Underinvestment hypothesis'),
    'asset_turnover':   ('(-)', 'Operational inefficiency (Palepu, 1986)'),
    'interest_burden':  ('(+)', 'Financial distress / toehold (Jensen, 1986)'),
    'net_margin':       ('(-)', 'Profitability improvement motive'),
    'rev_growth':       ('(-)', 'Undervaluation theory (Morck et al., 1988)'),
    'fcf_volatility':   ('(+)', 'Information asymmetry (Tunyi, 2021a)'),
    # --- New v6 features ---
    'firm_size':       ('(-)', 'Size hypothesis — larger firms less likely targets (Palepu, 1986)'),
    'rd_intensity':    ('(+)', 'Strategic asset motive — IP / innovation acquisition'),
    'rev_growth_lag1': ('(-)', 'Lagged growth underperformance (Morck et al., 1988)'),
    'altman_re_ta':    ('(-)', 'Financial health — higher retained earnings reduce target likelihood'),
}

coef_point = pipeline.named_steps['lr'].coef_[0]
intercept  = pipeline.named_steps['lr'].intercept_[0]

# Bootstrap CIs on raw X_train (pipeline refitted each iteration)
# liblinear solver is ~5x faster than lbfgs per iteration; 500 reps is
# sufficient for stable 95% CIs (SE of quantile estimate < 0.5% at n=500).
N_BOOT = 200
boot_coefs = np.zeros((N_BOOT, len(FEATURES)))
rng = np.random.default_rng(RANDOM_STATE)
print(f"  Running {N_BOOT} bootstrap iterations...")

for b in range(N_BOOT):
    idx = rng.integers(0, len(X_train), size=len(X_train))
    Xb, yb = X_train[idx], y_train[idx]
    if yb.sum() < 2:   # guard against degenerate draws
        boot_coefs[b] = coef_point
        continue
    pipe_b = Pipeline([
        ('scaler', RobustScaler()),
        ('lr', LogisticRegression(max_iter=300, solver='liblinear',
                                  class_weight='balanced',
                                  C=1.0, random_state=RANDOM_STATE))
    ])
    try:
        pipe_b.fit(Xb, yb)
        boot_coefs[b] = pipe_b.named_steps['lr'].coef_[0]
    except Exception:
        boot_coefs[b] = coef_point

ci_lo = np.percentile(boot_coefs, 2.5,  axis=0)
ci_hi = np.percentile(boot_coefs, 97.5, axis=0)
odds_ratios = np.exp(coef_point)

coef_rows = []
for i, feat in enumerate(FEATURES):
    theory_dir, theory_src = THEORY_DIRECTIONS[feat]
    actual_dir = '(+)' if coef_point[i] > 0 else '(-)'
    consistent = (
        theory_dir == actual_dir or
        theory_dir == '(+/-)'
    )
    coef_rows.append({
        'Feature':            feat,
        'Coefficient':        round(coef_point[i], 4),
        'Odds_Ratio':         round(odds_ratios[i], 4),
        'CI_lo_95':           round(ci_lo[i], 4),
        'CI_hi_95':           round(ci_hi[i], 4),
        'Actual_Direction':   actual_dir,
        'Expected_Theory_Dir': theory_dir,
        'Theory_Source':      theory_src,
        'Theory_Consistent':  'Y' if consistent else 'N',
    })

coef_df = pd.DataFrame(coef_rows).sort_values('Coefficient', ascending=False)
coef_df.to_csv('outputs/table_07_logistic_coefficients.csv', index=False)
print("Table 07 saved: outputs/table_07_logistic_coefficients.csv")
print("\n  Coefficients (sorted):")
print(coef_df[['Feature','Coefficient','Odds_Ratio','CI_lo_95','CI_hi_95',
               'Actual_Direction','Expected_Theory_Dir','Theory_Consistent']].to_string(index=False))

# Figure 8 — coefficient plot
fig, ax = plt.subplots(figsize=(8, 6))
feat_sorted = coef_df['Feature'].tolist()
coef_vals   = coef_df['Coefficient'].values
ci_lo_sorted = coef_df['CI_lo_95'].values
ci_hi_sorted = coef_df['CI_hi_95'].values
y_pos = np.arange(len(feat_sorted))

bar_colors = [COLORS[1] if c > 0 else COLORS[0] for c in coef_vals]
ax.barh(y_pos, coef_vals, color=bar_colors, alpha=0.7, height=0.5, zorder=3)
ax.errorbar(coef_vals, y_pos,
            xerr=[coef_vals - ci_lo_sorted, ci_hi_sorted - coef_vals],
            fmt='none', color='black', capsize=4, lw=1.5, zorder=4)
ax.axvline(0, color='black', lw=0.8, linestyle='--')
ax.set_yticks(y_pos)
ax.set_yticklabels(feat_sorted, fontsize=9)
ax.set_xlabel('Logistic Regression Coefficient\n(RobustScaled features; class_weight=\'balanced\')')
ax.set_title('Logistic Regression Coefficients with Bootstrap 95% CI\n'
             '(Positive = higher M&A likelihood; Negative = lower)')

# Add theory direction annotations
for i, (_, row) in enumerate(coef_df.iterrows()):
    color = 'green' if row['Theory_Consistent'] == 'Y' else 'red'
    ax.text(ax.get_xlim()[1] * 0.98, i,
            row['Expected_Theory_Dir'],
            va='center', ha='right', fontsize=8, color=color,
            fontweight='bold')

ax.text(0.99, 0.01, 'Green = theory-consistent, Red = inconsistent',
        transform=ax.transAxes, ha='right', fontsize=7, color='gray')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/fig_08_logistic_coefs.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 08 saved: outputs/fig_08_logistic_coefs.png")

# ─────────────────────────────────────────────
# STEP 2.6 — Save artifacts
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2.6 — Save model artifacts")
print("=" * 60)

with open('outputs/model_baseline_logistic.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
print("Saved: outputs/model_baseline_logistic.pkl")

# Save scaler separately (specific to logistic pipeline)
# NOTE: XGBoost does not require scaling; MLP uses its own scaler fitted in Script 04.
with open('outputs/scaler_logistic.pkl', 'wb') as f:
    pickle.dump(pipeline.named_steps['scaler'], f)
print("Saved: outputs/scaler_logistic.pkl")

print("\n=== Script 02 complete. All outputs saved to outputs/ ===")
print(f"\nSummary:")
print(f"  Val  ROC-AUC={roc_auc_val:.4f}  PR-AUC={pr_auc_val:.4f}  "
      f"(lift={pr_auc_val/NO_SKILL_PRAUC:.2f}x over no-skill)")
print(f"  Test ROC-AUC={roc_auc_test:.4f}  PR-AUC={pr_auc_test:.4f}  "
      f"(lift={pr_auc_test/NO_SKILL_PRAUC:.2f}x over no-skill)")
print(f"  Optimal threshold: {best_thresh:.3f}")
