import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_curve, auc, average_precision_score, precision_recall_curve,
    f1_score, precision_score, recall_score
)
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

warnings.filterwarnings('ignore')
os.makedirs('outputs', exist_ok=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
NO_SKILL_PRAUC   = 0.0157
PREVALENCE_LABEL = 'No-skill baseline (prevalence = 1.57%)'
RANDOM_STATE     = 42
SMOTE_SS_GRID    = [0.10, 0.20, 0.50]   # sampling_strategy candidates

DATA_FILE = 'data_final_modeling_ma_v7.xlsx'
TARGET    = 'target_next_year'
FEATURES  = [
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

# Fixed MLP hyperparameters — mid-range defaults to isolate resampling effect.
# MLP requires RobustScaler (feature-scale sensitive).
MLP_FIXED_PARAMS = dict(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    alpha=1e-3,
    learning_rate_init=1e-3,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=RANDOM_STATE,
)

STRATEGY_COLORS = {
    'S1_NoResample':        '#2c7bb6',
    'S2_RandomOverSampler': '#fdae61',
    'S3_RandomUnderSampler':'#abd9e9',
    'S4_SMOTE':             '#d7191c',
    'S5_SMOTEENN':          '#1a9641',
}

# ─────────────────────────────────────────────
# STEP 3c.1 — Load data (same canonical split as Scripts 02–03a/b)
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 3c.1 — Setup (canonical temporal split 2012–2024)")
print("=" * 60)

df = pd.read_excel(DATA_FILE)
df['fiscal_year'] = pd.to_datetime(df['datadate']).dt.year

df_train = df[df['fiscal_year'].between(2012, 2021)].copy()
df_val   = df[df['fiscal_year'] == 2022].copy()
df_test  = df[df['fiscal_year'].between(2023, 2024)].copy()

X_train = df_train[FEATURES].values
y_train = df_train[TARGET].values
X_val   = df_val[FEATURES].values
y_val   = df_val[TARGET].values
X_test  = df_test[FEATURES].values
y_test  = df_test[TARGET].values

n_pos_train = int(y_train.sum())
n_neg_train = int((y_train == 0).sum())

print(f"\nTraining set : {len(y_train):,} obs | {n_pos_train} positives | "
      f"{n_pos_train/len(y_train)*100:.2f}% base rate")
print(f"Validation set: {len(y_val):,} obs | {int(y_val.sum())} positives")
print(f"Test set      : {len(y_test):,} obs | {int(y_test.sum())} positives")
print(f"\nNote: MLP has no native class_weight support.")
print(f"      S1 = baseline (no resampling); S2–S5 rely entirely on the sampler.")

# SMOTE guard: k_neighbors must not exceed n_positives/5
K_NEIGHBORS = 5
max_k = n_pos_train // 5
assert K_NEIGHBORS <= max_k, \
    f"k_neighbors={K_NEIGHBORS} exceeds limit of {max_k} (n_pos/5)"
print(f"\nSMOTE k_neighbors={K_NEIGHBORS} confirmed safe "
      f"(n_positives={n_pos_train}, limit={max_k})")

print(f"\nFixed MLP hyperparameters (resampling effect isolated):")
for k, v in MLP_FIXED_PARAMS.items():
    if k not in ('random_state',):
        print(f"  {k}: {v}")

# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────
def fit_evaluate(X_tr, y_tr, X_v, y_v, sampler):
    """
    Resample (if sampler provided) → RobustScale → fit MLP → evaluate on val.
    MLP requires scaling; scaler is fitted on the (potentially resampled) training data.
    Returns metric dict and val probabilities.
    """
    if sampler is not None:
        X_res, y_res = sampler.fit_resample(X_tr, y_tr)
    else:
        X_res, y_res = X_tr.copy(), y_tr.copy()

    scaler = RobustScaler()
    X_res_sc = scaler.fit_transform(X_res)
    X_v_sc   = scaler.transform(X_v)

    mlp = MLPClassifier(**MLP_FIXED_PARAMS)
    mlp.fit(X_res_sc, y_res)
    y_proba = mlp.predict_proba(X_v_sc)[:, 1]

    roc_auc = auc(*roc_curve(y_v, y_proba)[:2])
    pr_auc  = average_precision_score(y_v, y_proba)

    # Optimal F1 threshold on val
    best_f1, best_thresh = 0.0, 0.5
    for t in np.arange(0.001, 0.151, 0.001):
        f = f1_score(y_v, (y_proba >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_thresh = f, t

    y_pred = (y_proba >= best_thresh).astype(int)

    return {
        'ROC_AUC':         roc_auc,
        'PR_AUC':          pr_auc,
        'F1':              best_f1,
        'Precision':       precision_score(y_v, y_pred, zero_division=0),
        'Recall':          recall_score(y_v, y_pred, zero_division=0),
        'Threshold':       best_thresh,
        'Lift_vs_NoSkill': pr_auc / NO_SKILL_PRAUC,
        'N_res_pos':       int(y_res.sum()),
        'N_res_neg':       int((y_res == 0).sum()),
    }, y_proba


# ─────────────────────────────────────────────
# STEP 3c.2 — SMOTE sensitivity scan → select best sampling_strategy
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3c.2 — SMOTE sensitivity scan (select best sampling_strategy)")
print("=" * 60)
print(f"  Grid : {SMOTE_SS_GRID}  (SMOTE, k={K_NEIGHBORS})")

smote_sensitivity = {}
for ss in tqdm(SMOTE_SS_GRID, desc="SMOTE sensitivity"):
    s = SMOTE(k_neighbors=K_NEIGHBORS, sampling_strategy=ss, random_state=RANDOM_STATE)
    res_s, _ = fit_evaluate(
        X_train.copy(), y_train.copy(), X_val, y_val, s
    )
    smote_sensitivity[ss] = res_s
    tqdm.write(f"    ss={ss:.2f}: PR-AUC={res_s['PR_AUC']:.4f} | "
               f"ROC-AUC={res_s['ROC_AUC']:.4f} | F1={res_s['F1']:.4f} | "
               f"N_pos_res={res_s['N_res_pos']}")

best_ss = max(smote_sensitivity, key=lambda s: smote_sensitivity[s]['PR_AUC'])
print(f"\n  Best sampling_strategy = {best_ss:.2f}  "
      f"(PR-AUC = {smote_sensitivity[best_ss]['PR_AUC']:.4f})")
print(f"  This value is applied to all resamplers (S2–S5).")

# ─────────────────────────────────────────────
# STEP 3c.3 — Five strategies (built with best_ss)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"STEP 3c.3 — Five imbalance-handling strategies (ss={best_ss:.2f})")
print("=" * 60)

STRATEGIES = {
    'S1_NoResample': {
        'sampler': None,
        'label':   'S1: No resampling\n(MLP baseline)',
        'short':   'S1\nNo resample',
    },
    'S2_RandomOverSampler': {
        'sampler': RandomOverSampler(sampling_strategy=best_ss, random_state=RANDOM_STATE),
        'label':   f'S2: RandomOverSampler\n(ss={best_ss:.2f})',
        'short':   'S2\nRandOver',
    },
    'S3_RandomUnderSampler': {
        'sampler': RandomUnderSampler(sampling_strategy=best_ss, random_state=RANDOM_STATE),
        'label':   f'S3: RandomUnderSampler\n(ss={best_ss:.2f})',
        'short':   'S3\nRandUnder',
    },
    'S4_SMOTE': {
        'sampler': SMOTE(k_neighbors=K_NEIGHBORS, sampling_strategy=best_ss,
                         random_state=RANDOM_STATE),
        'label':   f'S4: SMOTE\n(k={K_NEIGHBORS}, ss={best_ss:.2f})',
        'short':   'S4\nSMOTE',
    },
    'S5_SMOTEENN': {
        'sampler': SMOTEENN(
                       smote=SMOTE(k_neighbors=K_NEIGHBORS, sampling_strategy=best_ss,
                                   random_state=RANDOM_STATE),
                       random_state=RANDOM_STATE
                   ),
        'label':   f'S5: SMOTE+ENN\n(k={K_NEIGHBORS}, ss={best_ss:.2f})',
        'short':   'S5\nSMOTE+ENN',
    },
}

results = {}
probas  = {}

for name, cfg in tqdm(STRATEGIES.items(), desc="Strategies", total=len(STRATEGIES)):
    tqdm.write(f"\n  [{name}]")
    res, yp = fit_evaluate(
        X_train.copy(), y_train.copy(), X_val, y_val,
        cfg['sampler']
    )
    results[name] = res
    probas[name]  = yp
    tqdm.write(f"    Train after resample: {res['N_res_pos']} pos / {res['N_res_neg']} neg")
    tqdm.write(f"    Val PR-AUC : {res['PR_AUC']:.4f} | Lift over no-skill: {res['Lift_vs_NoSkill']:.2f}x")
    tqdm.write(f"    Val ROC-AUC: {res['ROC_AUC']:.4f} | F1: {res['F1']:.4f} "
               f"| Threshold: {res['Threshold']:.3f}")

# ─────────────────────────────────────────────
# STEP 3c.4 — Select winning strategy
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3c.4 — Winner selection (argmax Val PR-AUC)")
print("=" * 60)

best_name = max(results, key=lambda k: results[k]['PR_AUC'])
best_res  = results[best_name]

print(f"\n  Winner: {best_name}")
print(f"  Val PR-AUC : {best_res['PR_AUC']:.4f}  "
      f"(lift = {best_res['Lift_vs_NoSkill']:.2f}x over no-skill)")
print(f"  Val ROC-AUC: {best_res['ROC_AUC']:.4f}")
print(f"  Val F1     : {best_res['F1']:.4f}  (threshold = {best_res['Threshold']:.3f})")

# ─────────────────────────────────────────────
# Build and save comparison table
# ─────────────────────────────────────────────
rows = []
for name, res in results.items():
    rows.append({
        'Strategy':        name,
        'Description':     STRATEGIES[name]['label'].replace('\n', ' '),
        'N_resampled_pos': res['N_res_pos'],
        'N_resampled_neg': res['N_res_neg'],
        'Val_ROC_AUC':     round(res['ROC_AUC'],         4),
        'Val_PR_AUC':      round(res['PR_AUC'],          4),
        'Val_F1':          round(res['F1'],               4),
        'Val_Precision':   round(res['Precision'],        4),
        'Val_Recall':      round(res['Recall'],           4),
        'Threshold':       round(res['Threshold'],        3),
        'Lift_vs_NoSkill': round(res['Lift_vs_NoSkill'],  3),
        'Selected':        'Y' if name == best_name else 'N',
    })

for ss, res_s in smote_sensitivity.items():
    rows.append({
        'Strategy':        f'S4_SMOTE_ss{ss:.2f}',
        'Description':     f'S4 sensitivity (ss={ss:.2f})',
        'N_resampled_pos': res_s['N_res_pos'],
        'N_resampled_neg': res_s['N_res_neg'],
        'Val_ROC_AUC':     round(res_s['ROC_AUC'],                    4),
        'Val_PR_AUC':      round(res_s['PR_AUC'],                     4),
        'Val_F1':          round(res_s['F1'],                         4),
        'Val_Precision':   round(res_s['Precision'],                  4),
        'Val_Recall':      round(res_s['Recall'],                     4),
        'Threshold':       round(res_s['Threshold'],                  3),
        'Lift_vs_NoSkill': round(res_s['PR_AUC'] / NO_SKILL_PRAUC,   3),
        'Selected':        'N',
    })

comp_df = pd.DataFrame(rows)
comp_df.to_csv('outputs/table_08c_imbalance_mlp.csv', index=False)
print("\nTable 08c saved: outputs/table_08c_imbalance_mlp.csv")
print("\nStrategy summary (5 main strategies):")
print(comp_df[comp_df['Strategy'].str.startswith('S') & ~comp_df['Strategy'].str.contains('ss')][
    ['Strategy', 'Val_ROC_AUC', 'Val_PR_AUC', 'Val_F1',
     'Lift_vs_NoSkill', 'Selected']
].to_string(index=False))

# ─────────────────────────────────────────────
# STEP 3c.5 — Figure 09c
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3c.5 — Figure 09c: Four-panel comparison")
print("=" * 60)

strat_names  = list(STRATEGIES.keys())
short_labels = [STRATEGIES[n]['short'] for n in strat_names]
colors_bar   = [STRATEGY_COLORS[n] for n in strat_names]
pr_aucs      = [results[n]['PR_AUC']  for n in strat_names]
roc_aucs     = [results[n]['ROC_AUC'] for n in strat_names]
f1s          = [results[n]['F1']      for n in strat_names]
x            = np.arange(len(strat_names))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax_A, ax_B = axes[0]
ax_C, ax_D = axes[1]


def annotate_bars(ax, bars, values, fmt='{:.4f}', offset=0.0003):
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                fmt.format(val), ha='center', va='bottom', fontsize=7.5)


# ── Panel A: Val PR-AUC ──────────────────────────────────────────────────────
bars_A = ax_A.bar(x, pr_aucs, color=colors_bar, alpha=0.85, width=0.6, zorder=3)
ax_A.axhline(y=NO_SKILL_PRAUC, color='gray', linestyle='--', lw=1.5,
             label=PREVALENCE_LABEL)
ax_A.set_xticks(x)
ax_A.set_xticklabels(short_labels, fontsize=8)
ax_A.set_ylabel('Val PR-AUC')
ax_A.set_title('Panel A: PR-AUC by Strategy\n(primary metric — higher is better)',
               fontsize=10)
ax_A.legend(fontsize=7.5)
ax_A.grid(axis='y', alpha=0.3)
annotate_bars(ax_A, bars_A, pr_aucs)
for tick, name in zip(ax_A.get_xticklabels(), strat_names):
    if name == best_name:
        tick.set_fontweight('bold')
        tick.set_color('#d7191c')

# ── Panel B: Val ROC-AUC ─────────────────────────────────────────────────────
bars_B = ax_B.bar(x, roc_aucs, color=colors_bar, alpha=0.85, width=0.6, zorder=3)
ax_B.axhline(y=0.50, color='gray', linestyle='--', lw=1.5, label='No-skill (0.50)')
ax_B.set_xticks(x)
ax_B.set_xticklabels(short_labels, fontsize=8)
ax_B.set_ylabel('Val ROC-AUC')
ax_B.set_title('Panel B: ROC-AUC by Strategy', fontsize=10)
ax_B.legend(fontsize=7.5)
ax_B.grid(axis='y', alpha=0.3)
annotate_bars(ax_B, bars_B, roc_aucs, offset=0.002)
for tick, name in zip(ax_B.get_xticklabels(), strat_names):
    if name == best_name:
        tick.set_fontweight('bold')
        tick.set_color('#d7191c')

# ── Panel C: PR curves overlaid ──────────────────────────────────────────────
for name in strat_names:
    prec_c, rec_c, _ = precision_recall_curve(y_val, probas[name])
    ap = results[name]['PR_AUC']
    lw = 2.5 if name == best_name else 1.2
    ls = '-'  if name == best_name else '--'
    ax_C.plot(rec_c, prec_c,
              color=STRATEGY_COLORS[name], lw=lw, ls=ls,
              label=f"{STRATEGIES[name]['short'].replace(chr(10), ' ')} (AP={ap:.4f})",
              zorder=4 if name == best_name else 3)

ax_C.axhline(y=NO_SKILL_PRAUC, color='gray', linestyle=':', lw=1.5,
             label=PREVALENCE_LABEL)
ax_C.set_xlabel('Recall')
ax_C.set_ylabel('Precision')
ax_C.set_title('Panel C: Precision-Recall Curves (Validation Set)', fontsize=10)
ax_C.legend(fontsize=7, loc='upper right')
ax_C.set_xlim([0, 1])
ax_C.set_ylim([0, 0.25])
ax_C.grid(alpha=0.3)

# ── Panel D: SMOTE sensitivity (PR-AUC vs sampling_strategy) ─────────────────
ss_vals  = list(smote_sensitivity.keys())
ss_prs   = [smote_sensitivity[ss]['PR_AUC']  for ss in ss_vals]
ss_rocs  = [smote_sensitivity[ss]['ROC_AUC'] for ss in ss_vals]
ss_f1s   = [smote_sensitivity[ss]['F1']      for ss in ss_vals]

ss_x = np.arange(len(ss_vals))
w = 0.25
bars_pr  = ax_D.bar(ss_x - w,  ss_prs,  width=w, color='#d7191c', alpha=0.85,
                    label='PR-AUC', zorder=3)
bars_roc = ax_D.bar(ss_x,       ss_rocs, width=w, color='#2c7bb6', alpha=0.85,
                    label='ROC-AUC', zorder=3)
bars_f1  = ax_D.bar(ss_x + w,  ss_f1s,  width=w, color='#1a9641', alpha=0.85,
                    label='F1', zorder=3)

ax_D.axhline(y=NO_SKILL_PRAUC, color='gray', linestyle='--', lw=1.2,
             label=f'No-skill PR-AUC ({NO_SKILL_PRAUC})')
ax_D.set_xticks(ss_x)
ax_D.set_xticklabels([f'ss={ss:.2f}{"*" if ss == best_ss else ""}' for ss in ss_vals],
                     fontsize=9)
ax_D.set_ylabel('Metric value')
ax_D.set_title(f'Panel D: SMOTE Sensitivity\n(S4, k={K_NEIGHBORS}, * = selected ss)',
               fontsize=10)
ax_D.legend(fontsize=8)
ax_D.grid(axis='y', alpha=0.3)
for bar, val in zip(bars_pr, ss_prs):
    ax_D.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
              f'{val:.4f}', ha='center', va='bottom', fontsize=7)

fig.suptitle(
    'Class Imbalance Strategy Comparison — MLP, Validation Set\n'
    f'Winner (bold/red): {best_name}  |  '
    f'Val PR-AUC = {best_res["PR_AUC"]:.4f}  |  '
    f'Lift = {best_res["Lift_vs_NoSkill"]:.2f}× over no-skill  |  '
    f'ss={best_ss:.2f} (data-driven)',
    fontsize=11
)
plt.tight_layout()
plt.savefig('outputs/fig_09c_imbalance_mlp.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 09c saved: outputs/fig_09c_imbalance_mlp.png")

# ─────────────────────────────────────────────
# Persist selected strategy for Script 04
# ─────────────────────────────────────────────
# sampler_config strings are built from best_ss (not hard-coded)
sampler_cfg_str = {
    'S1_NoResample':         'NoResample',
    'S2_RandomOverSampler':  f'RandomOverSampler(sampling_strategy={best_ss:.2f})',
    'S3_RandomUnderSampler': f'RandomUnderSampler(sampling_strategy={best_ss:.2f})',
    'S4_SMOTE':              f'SMOTE(k_neighbors={K_NEIGHBORS}, sampling_strategy={best_ss:.2f})',
    'S5_SMOTEENN':           f'SMOTEENN(SMOTE(k_neighbors={K_NEIGHBORS}, sampling_strategy={best_ss:.2f}))',
}

with open('outputs/selected_imbalance_strategy_mlp.txt', 'w') as f:
    f.write(f"selected_strategy={best_name}\n")
    f.write(f"sampler_config={sampler_cfg_str[best_name]}\n")
    f.write(f"val_pr_auc={best_res['PR_AUC']:.6f}\n")
    f.write(f"val_roc_auc={best_res['ROC_AUC']:.6f}\n")
    f.write(f"val_f1={best_res['F1']:.6f}\n")
    f.write(f"lift_vs_noskill={best_res['Lift_vs_NoSkill']:.4f}\n")
    f.write(f"no_skill_prauc={NO_SKILL_PRAUC}\n")
    f.write(f"k_neighbors={K_NEIGHBORS}\n")
    f.write(f"best_sampling_strategy={best_ss:.2f}\n")

print("Selected strategy saved: outputs/selected_imbalance_strategy_mlp.txt")

print("\n=== Script 03c complete. All outputs saved to outputs/ ===")
print(f"\nSummary:")
print(f"  Winner     : {best_name}")
print(f"  Val PR-AUC : {best_res['PR_AUC']:.4f}  "
      f"(lift = {best_res['Lift_vs_NoSkill']:.2f}x over no-skill)")
print(f"  Val ROC-AUC: {best_res['ROC_AUC']:.4f}")
print(f"  Val F1     : {best_res['F1']:.4f}  (threshold = {best_res['Threshold']:.3f})")
print(f"  ss used    : {best_ss:.2f} (selected from sensitivity scan)")
print(f"\n  This strategy will be applied to MLP in Script 04.")
