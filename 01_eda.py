import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

warnings.filterwarnings('ignore')
os.makedirs('outputs', exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1.1 — Load data and define column taxonomy
# ─────────────────────────────────────────────

DATA_FILE = 'data_final_modeling_ma_v7.xlsx'
print(f"Loading dataset: {DATA_FILE}")
df = pd.read_excel(DATA_FILE)
print(f"  Raw shape: {df.shape}")

ID_COLS  = ['gvkey', 'conm', 'tic', 'datadate', 'cusip', 'cik']
SECTOR   = 'sic'   # SIC code; grouping only, never a model feature
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

# Extract fiscal year
df['fiscal_year'] = pd.to_datetime(df['datadate']).dt.year
print(f"  Fiscal years: {sorted(df['fiscal_year'].unique())}")

# SIC division mapping
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

df['sic_division'] = df[SECTOR].apply(sic_to_division)

# Check Finance & Insurance over-representation in targets
fi_total  = (df['sic_division'] == 'Finance & Insurance').sum()
fi_target = ((df['sic_division'] == 'Finance & Insurance') & (df[TARGET] == 1)).sum()
fi_rate   = fi_target / fi_total if fi_total > 0 else 0
overall_rate = df[TARGET].mean()
if fi_rate > 2 * overall_rate:
    print(f"  FLAG: Finance & Insurance over-represented in targets "
          f"(base rate {fi_rate:.2%} vs overall {overall_rate:.2%})")

# ─────────────────────────────────────────────
# STEP 1.2 — Dataset-level summary table
# ─────────────────────────────────────────────

n_obs        = len(df)
n_firms      = df['gvkey'].nunique()
n_ma         = (df[TARGET] == 1).sum()
n_non_ma     = (df[TARGET] == 0).sum()
base_rate    = n_ma / n_obs * 100
imb_ratio    = n_non_ma / n_ma
yr_min       = df['fiscal_year'].min()
yr_max       = df['fiscal_year'].max()
n_sic        = df[SECTOR].nunique()
n_divisions  = df['sic_division'].nunique()

summary = pd.DataFrame({
    'Metric': [
        'Total firm-year observations',
        'Unique firms (gvkey)',
        'M&A events (target_next_year == 1)',
        'Non-events (target_next_year == 0)',
        'Base rate (observation level, %)',
        'Imbalance ratio (neg:pos)',
        'Fiscal year range',
        'Unique SIC codes',
        'SIC divisions represented',
    ],
    'Value': [
        n_obs,
        n_firms,
        n_ma,
        n_non_ma,
        f'{base_rate:.2f}%',
        f'{imb_ratio:.1f}:1',
        f'{yr_min}–{yr_max}',
        n_sic,
        n_divisions,
    ]
})
summary.to_csv('outputs/table_01_dataset_summary.csv', index=False)
print("\nTable 01 saved: outputs/table_01_dataset_summary.csv")
print(summary.to_string(index=False))

# ─────────────────────────────────────────────
# STEP 1.3 — Descriptive statistics table
# ─────────────────────────────────────────────

def desc_stats(data, label_suffix=''):
    rows = []
    for feat in FEATURES:
        col = data[feat].dropna()
        rows.append({
            'feature':          feat,
            f'N{label_suffix}':      len(col),
            f'Mean{label_suffix}':   col.mean(),
            f'Std{label_suffix}':    col.std(),
            f'P5{label_suffix}':     col.quantile(0.05),
            f'P25{label_suffix}':    col.quantile(0.25),
            f'Median{label_suffix}': col.median(),
            f'P75{label_suffix}':    col.quantile(0.75),
            f'P95{label_suffix}':    col.quantile(0.95),
            f'Min{label_suffix}':    col.min(),
            f'Max{label_suffix}':    col.max(),
        })
    return pd.DataFrame(rows).set_index('feature')

df0 = df[df[TARGET] == 0]
df1 = df[df[TARGET] == 1]

stats_full = desc_stats(df,  '_full')
stats_0    = desc_stats(df0, '_class0')
stats_1    = desc_stats(df1, '_class1')

# Welch t-test and Mann-Whitney U
sig_rows = []
for feat in FEATURES:
    c0 = df0[feat].dropna()
    c1 = df1[feat].dropna()
    t_pval  = stats.ttest_ind(c0, c1, equal_var=False).pvalue
    mw_pval = stats.mannwhitneyu(c0, c1, alternative='two-sided').pvalue

    def star(p):
        if p < 0.01:  return '***'
        if p < 0.05:  return '**'
        if p < 0.10:  return '*'
        return ''

    sig_rows.append({
        'feature':         feat,
        'welch_t_pval':    t_pval,
        'welch_sig':       star(t_pval),
        'mannwhitney_pval': mw_pval,
        'mannwhitney_sig': star(mw_pval),
    })

sig_df = pd.DataFrame(sig_rows).set_index('feature')

desc_out = stats_full.join(stats_0).join(stats_1).join(sig_df)
desc_out.to_csv('outputs/table_02_descriptive_stats.csv')
print("\nTable 02 saved: outputs/table_02_descriptive_stats.csv")

# ─────────────────────────────────────────────
# STEP 1.4 — Class imbalance visualization
# ─────────────────────────────────────────────

COLORS = ['#2c7bb6', '#d7191c']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: log-scale bar chart
ax = axes[0]
counts   = [n_non_ma, n_ma]
labels   = ['Class 0\n(Non-target)', 'Class 1\n(M&A target)']
bars = ax.bar(labels, counts, color=COLORS, edgecolor='white', width=0.5)
ax.set_yscale('log')
ax.set_ylabel('Count (log scale)')
ax.set_title('Panel A: Class Distribution (Log Scale)')
for bar, cnt in zip(bars, counts):
    pct = cnt / n_obs * 100
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.4,
            f'{cnt:,}\n({pct:.2f}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylim(1, n_obs * 5)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

# Panel B: base rate per fiscal year with Wilson 95% CI
ax2 = axes[1]
year_df = (df.groupby('fiscal_year')
             .agg(total=(TARGET, 'count'), pos=(TARGET, 'sum'))
             .reset_index())
year_df['base_rate'] = year_df['pos'] / year_df['total'] * 100

ci_lo, ci_hi = [], []
for _, row in year_df.iterrows():
    lo, hi = proportion_confint(row['pos'], row['total'], alpha=0.05, method='wilson')
    ci_lo.append(lo * 100)
    ci_hi.append(hi * 100)
year_df['ci_lo'] = ci_lo
year_df['ci_hi'] = ci_hi

# Stacked bars (class 0 and class 1)
ax2.bar(year_df['fiscal_year'],
        year_df['total'] - year_df['pos'],
        label='Class 0', color=COLORS[0], alpha=0.7)
ax2.bar(year_df['fiscal_year'],
        year_df['pos'],
        bottom=year_df['total'] - year_df['pos'],
        label='Class 1', color=COLORS[1], alpha=0.9)

# Wilson CI on base rate (right axis)
ax2b = ax2.twinx()
ax2b.errorbar(year_df['fiscal_year'], year_df['base_rate'],
              yerr=[year_df['base_rate'] - year_df['ci_lo'],
                    year_df['ci_hi'] - year_df['base_rate']],
              fmt='o-', color='black', capsize=4, label='Base rate + 95% Wilson CI')
ax2b.set_ylabel('Base rate (%)')

ax2.set_xlabel('Fiscal Year of Observation (t)')
ax2.set_ylabel('Firm-year observations')
ax2.set_title('Panel B: Base Rate per Fiscal Year of Observation (t)\n(acquisition occurs in t + 1)')
ax2.legend(loc='upper left', fontsize=8)
ax2b.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('outputs/fig_01_class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 01 saved: outputs/fig_01_class_distribution.png")

# ─────────────────────────────────────────────
# STEP 1.5 — Feature distribution plots
# ─────────────────────────────────────────────

def winsorize(series, lower=0.01, upper=0.99):
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)

ncols = 4
nrows = 4  # 15 features → 4 rows × 4 cols (1 slot empty)
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 16))
axes_flat = axes.flatten()

for i, feat in enumerate(FEATURES):
    ax = axes_flat[i]
    c0_w = winsorize(df0[feat].dropna())
    c1_w = winsorize(df1[feat].dropna())
    c0_w.plot.kde(ax=ax, color=COLORS[0], alpha=0.5, label='Class 0' if i == 0 else '',
                  bw_method=0.3)
    c1_w.plot.kde(ax=ax, color=COLORS[1], alpha=0.5, label='Class 1 (bw×1.5)' if i == 0 else '',
                  bw_method=0.3 * 1.5)
    ax.set_title(feat, fontsize=9)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(labelsize=7)
    if i == 0:
        ax.legend(fontsize=8)

# Hide unused subplots
for j in range(len(FEATURES), len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle('Feature Distributions: Class 0 vs Class 1\n(winsorized 1%–99%)',
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('outputs/fig_02_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 02 saved: outputs/fig_02_feature_distributions.png")

# ─────────────────────────────────────────────
# STEP 1.6 — Pearson correlation matrix
# ─────────────────────────────────────────────

corr = df[FEATURES].corr()

# Find highly correlated pairs
high_corr = []
for i, f1 in enumerate(FEATURES):
    for j, f2 in enumerate(FEATURES):
        if i < j and abs(corr.loc[f1, f2]) > 0.7:
            high_corr.append(f'{f1} / {f2}: r={corr.loc[f1, f2]:.2f}')

fig, ax = plt.subplots(figsize=(10, 9))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # upper triangle (keep lower)
mask_upper = np.triu(np.ones_like(corr, dtype=bool))  # hide upper + diagonal
sns.heatmap(
    corr,
    mask=mask_upper,
    annot=True, fmt='.2f',
    cmap='RdBu_r', vmin=-1, vmax=1,
    ax=ax, linewidths=0.5,
    cbar_kws={'shrink': 0.8}
)
ax.set_title('Pearson Correlation Matrix (lower triangle)\n'
             + (f'|r|>0.7: {"; ".join(high_corr)}' if high_corr
                else 'No pairs with |r|>0.7'),
             fontsize=10)
plt.tight_layout()
plt.savefig('outputs/fig_03_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 03 saved: outputs/fig_03_correlation_matrix.png")
if high_corr:
    print(f"  High-correlation pairs (|r|>0.7): {high_corr}")
else:
    print("  No feature pairs with |r| > 0.7")

# ─────────────────────────────────────────────
# STEP 1.7 — Temporal M&A activity chart
# ─────────────────────────────────────────────

yr_df = year_df.copy()
yr_min_fig, yr_max_fig = int(yr_df['fiscal_year'].min()), int(yr_df['fiscal_year'].max())
n_years = yr_max_fig - yr_min_fig + 1

fig, ax = plt.subplots(figsize=(max(10, n_years * 0.8), 5))

# Left axis: bars for M&A event count
bars = ax.bar(yr_df['fiscal_year'], yr_df['pos'],
              color=COLORS[1], alpha=0.8, label='M&A events', zorder=3)
for bar, cnt in zip(bars, yr_df['pos']):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(int(cnt)), ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Fiscal Year of Observation (t)')
ax.set_ylabel('M&A Event Count')
ax.set_title(f'M&A Activity Over Time ({yr_min_fig}–{yr_max_fig})\n'
             f'Fiscal Year of Observation (t); acquisition occurs in t + 1')
ax.set_xticks(yr_df['fiscal_year'])

# COVID shading
ax.axvspan(2019.5, 2020.5, alpha=0.12, color='gray', label='COVID period (2020)')

# Right axis: base rate line + Wilson CI
ax2 = ax.twinx()
ax2.errorbar(yr_df['fiscal_year'], yr_df['base_rate'],
             yerr=[yr_df['base_rate'] - yr_df['ci_lo'],
                   yr_df['ci_hi'] - yr_df['base_rate']],
             fmt='o-', color='#1a9641', capsize=5, linewidth=2,
             label='Base rate % + 95% Wilson CI')
ax2.set_ylabel('Base Rate (%)')

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('outputs/fig_04_temporal_ma_activity.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 04 saved: outputs/fig_04_temporal_ma_activity.png")

# ─────────────────────────────────────────────
# STEP 1.8 — SIC sector distribution
# ─────────────────────────────────────────────

sector_df = (df.groupby('sic_division')
               .agg(n_total=(TARGET, 'count'), n_ma=(TARGET, 'sum'))
               .reset_index())
sector_df['base_rate_pct'] = sector_df['n_ma'] / sector_df['n_total'] * 100
sector_df = sector_df.sort_values('base_rate_pct', ascending=False)

# Flag Finance & Insurance if over-represented
sector_df['flag_fi'] = (
    (sector_df['sic_division'] == 'Finance & Insurance') &
    (sector_df['base_rate_pct'] > overall_rate * 100 * 2)
)

sector_df.to_csv('outputs/table_03_sector_summary.csv', index=False)
print("Table 03 saved: outputs/table_03_sector_summary.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: firm-year observations per division
ax = axes[0]
colors_a = ['#d7191c' if row['flag_fi'] else '#2c7bb6'
            for _, row in sector_df.iterrows()]
ax.barh(sector_df['sic_division'], sector_df['n_total'],
        color=colors_a, edgecolor='white')
ax.set_xlabel('Firm-year observations')
ax.set_title('Panel A: Observations per Sector')
ax.invert_yaxis()
for i, (_, row) in enumerate(sector_df.iterrows()):
    ax.text(row['n_total'] + 10, i,
            f"{int(row['n_total']):,}", va='center', fontsize=8)

# Panel B: M&A event count + base rate per division
ax2 = axes[1]
ax2.barh(sector_df['sic_division'], sector_df['n_ma'],
         color=colors_a, edgecolor='white')
ax2.set_xlabel('M&A Events')
ax2.set_title('Panel B: M&A Events & Base Rate per Sector\n(sorted by base rate ↓)')
ax2.invert_yaxis()
for i, (_, row) in enumerate(sector_df.iterrows()):
    ax2.text(row['n_ma'] + 0.1, i,
             f"{int(row['n_ma'])} / {int(row['n_total'])} ({row['base_rate_pct']:.2f}%)",
             va='center', fontsize=7.5)

plt.tight_layout()
plt.savefig('outputs/fig_05_sector_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 05 saved: outputs/fig_05_sector_distribution.png")

# ─────────────────────────────────────────────
# STEP 1.9 — Multivariate outlier report
# ─────────────────────────────────────────────

outlier_rows = []
for feat in FEATURES:
    col   = df[feat].dropna()
    z     = np.abs(stats.zscore(col))
    # Align z back with original index
    z_series = pd.Series(z, index=col.index)

    c0_idx  = df.loc[df[TARGET] == 0, feat].dropna().index
    c1_idx  = df.loc[df[TARGET] == 1, feat].dropna().index

    out0_n  = (z_series.loc[z_series.index.intersection(c0_idx)] > 3).sum()
    out1_n  = (z_series.loc[z_series.index.intersection(c1_idx)] > 3).sum()
    n0      = len(c0_idx)
    n1      = len(c1_idx)
    out0_pct = out0_n / n0 * 100 if n0 > 0 else 0
    out1_pct = out1_n / n1 * 100 if n1 > 0 else 0
    flag    = out1_pct > 2.0

    outlier_rows.append({
        'Feature':                 feat,
        'Outliers_class0_n':       out0_n,
        'Outliers_class0_pct':     f'{out0_pct:.2f}%',
        'Outliers_class1_n':       out1_n,
        'Outliers_class1_pct':     f'{out1_pct:.2f}%',
        'Flag_class1_outlier_rate_gt2pct': flag,
    })

outlier_df = pd.DataFrame(outlier_rows)
outlier_df.to_csv('outputs/table_04_outlier_report.csv', index=False)
print("Table 04 saved: outputs/table_04_outlier_report.csv")

flagged = outlier_df[outlier_df['Flag_class1_outlier_rate_gt2pct']]
if len(flagged):
    print(f"  Flagged features (class 1 outlier rate > 2%): "
          f"{flagged['Feature'].tolist()}")
else:
    print("  No features with class 1 outlier rate > 2%")

print("\n=== Script 01 complete. All outputs saved to outputs/ ===")
