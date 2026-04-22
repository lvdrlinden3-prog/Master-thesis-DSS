import pandas as pd
import numpy as np
import re
import sys

RELEVANT_MA_TYPES = {
    'Acquisitions of Remaining Interest',
    'Tender Offers',
    'Leveraged Buyouts',
    'Stake Purchase',
    'Privatizations'
}

def is_relevant_ma(ma_value):
    """Returns True if the M&A field contains at least one relevant deal type."""
    if pd.isna(ma_value):
        return False
    parts = [p.strip() for p in str(ma_value).split('|')]
    return any(p in RELEVANT_MA_TYPES for p in parts)


def clean_column_names(columns):
    """Extracts short codes from Compustat column names using regex."""
    new_cols = {}
    for col in columns:
        match = re.search(r'\((.*?)\)', col)
        if match:
            new_cols[col] = match.group(1).lower()
        else:
            new_cols[col] = col.strip().lower().replace(' ', '_')
    return new_cols


def read_excel_safe(filepath):
    """
    Reads an xlsx file using openpyxl, with a fallback that patches files
    exported without a sharedStrings.xml entry (common in Compustat/Refinitiv
    exports). No extra dependencies required beyond openpyxl.
    """
    import zipfile, io

    try:
        return pd.read_excel(filepath, engine='openpyxl')
    except KeyError:
        with open(filepath, 'rb') as f:
            data = f.read()

        buf = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(data)) as zin, \
             zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                zout.writestr(item, zin.read(item.filename))
            zout.writestr(
                'xl/sharedStrings.xml',
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
                ' count="0" uniqueCount="0"/>'
            )

        buf.seek(0)
        return pd.read_excel(buf, engine='openpyxl')


def merge_datasets(compustat_file, refinitiv_file):
    """
    Merges Compustat financial data with Refinitiv M&A data on 6-digit CUSIP.

    Compustat:  '(cusip) CUSIP' column — full CUSIP; first 6 chars = issuer code.
    Refinitiv:  'Target 6-digit CUSIP' column — issuer code used directly.
                'Date Announced' is the authoritative announcement date.
                'Deal Type' (pipe-separated) is filtered for relevant M&A types.

    Only relevant deals (matching RELEVANT_MA_TYPES) are carried into the merge.
    Compustat firms with no matching Refinitiv deal receive NaN for 'announced'
    and are treated as non-targets downstream.
    """
    print(f"  Loading Compustat: {compustat_file}")
    df_c = read_excel_safe(compustat_file)
    print(f"  Compustat rows: {len(df_c):,}")

    print(f"  Loading Refinitiv: {refinitiv_file}")
    df_r = read_excel_safe(refinitiv_file)
    print(f"  Refinitiv rows: {len(df_r):,}")

    # --- Compustat: clean column names, parse dates, extract 6-digit CUSIP ---
    df_c = df_c.rename(columns=clean_column_names(df_c.columns))
    df_c['datadate'] = pd.to_datetime(df_c['datadate'])
    df_c['cusip6'] = df_c['cusip'].astype(str).str[:6].str.upper().str.strip()

    # --- Refinitiv: locate columns by keyword (tries multiple fallbacks) ---
    def find_col(df, *keywords):
        """Return first column whose name contains any keyword (case-insensitive)."""
        for kw in keywords:
            matches = [c for c in df.columns if kw.lower() in c.lower()]
            if matches:
                return matches[0]
        return None

    # Prefer 'target 6-digit' to avoid accidentally matching an acquiror CUSIP col.
    cusip_col     = find_col(df_r, 'target 6-digit', 'target cusip', 'cusip')
    # Try 'date announced' first, then bare 'announced' as fallback.
    announced_col = find_col(df_r, 'date announced', 'announced')
    deal_type_col = find_col(df_r, 'deal type', 'm&a type', 'transaction type')

    missing = [label for label, col in [
                   ('Target 6-digit CUSIP', cusip_col),
                   ('Date Announced',       announced_col),
                   ('Deal Type',            deal_type_col),
               ] if col is None]
    if missing:
        print(f"ERROR: Required Refinitiv columns not found: {missing}")
        print(f"Available Refinitiv columns: {list(df_r.columns)}")
        sys.exit()

    print(f"  Refinitiv CUSIP col      : '{cusip_col}'")
    print(f"  Refinitiv announced col  : '{announced_col}'")
    print(f"  Refinitiv deal type col  : '{deal_type_col}'")

    df_r = df_r.rename(columns={
        cusip_col:     'cusip6_ref',
        announced_col: 'announced',
        deal_type_col: 'deal_type',
    })

    df_r['announced']  = pd.to_datetime(df_r['announced'], errors='coerce')
    # Truncate to 6 digits and normalise case/whitespace (defensive)
    df_r['cusip6_ref'] = df_r['cusip6_ref'].astype(str).str[:6].str.upper().str.strip()

    # --- Keep only relevant deal types; aggregate to earliest deal per CUSIP ---
    relevant_mask = df_r['deal_type'].apply(is_relevant_ma)
    df_r_relevant = (df_r[relevant_mask][['cusip6_ref', 'announced']]
                     .dropna(subset=['cusip6_ref', 'announced']))
    df_r_agg      = (df_r_relevant
                     .groupby('cusip6_ref')['announced']
                     .min()
                     .reset_index()
                     .rename(columns={'cusip6_ref': 'cusip6'}))

    print(f"  Relevant M&A deals in Refinitiv: {len(df_r_relevant):,} rows → "
          f"{len(df_r_agg):,} unique CUSIP6s")

    # --- Left join: every Compustat firm-year kept; non-targets get NaN ---
    df = df_c.merge(df_r_agg, on='cusip6', how='left')

    matched = df['announced'].notna().sum()
    print(f"  Compustat firm-years matched to a relevant M&A deal: "
          f"{matched:,} / {len(df):,}")

    return df


def preprocess_ma_data(compustat_file, refinitiv_file, output_file):
    print("Step 1: Merging Compustat and Refinitiv datasets...")
    df = merge_datasets(compustat_file, refinitiv_file)

    # ----------------------------------------------------------------
    # Step 2: M&A target labelling
    # ----------------------------------------------------------------
    print("Step 2: Identifying M&A target firms and applying lagging logic...")

    df['target_next_year'] = 0

    if 'announced' not in df.columns:
        print("ERROR: 'announced' column not found after merge.")
        print(f"Available columns: {list(df.columns)}")
        sys.exit()

    # Rows where 'announced' is non-null were matched to a relevant Refinitiv
    # deal (filtering for RELEVANT_MA_TYPES was done inside merge_datasets).
    # For each firm (gvkey), label the last fiscal-year-end BEFORE the
    # announcement date as the prediction target (target_next_year = 1).
    ma_dates = (df[df['announced'].notna()]
                .groupby('gvkey')['announced']
                .min())

    print(f"  Relevant M&A target firms found: {len(ma_dates)}")
    print(f"  Deal types: {', '.join(RELEVANT_MA_TYPES)}")

    for gvkey, ann_date in ma_dates.items():
        mask = (df['gvkey'] == gvkey) & (df['datadate'] < ann_date)
        if mask.any():
            idx = df[mask]['datadate'].idxmax()
            df.at[idx, 'target_next_year'] = 1

    # ----------------------------------------------------------------
    # Step 3: Drop rows missing core financials (unchanged from v5)
    # ----------------------------------------------------------------
    print("Step 3: Cleaning missing values (Critical Variables)...")
    essential_cols = ['at', 'ebitda', 'revt', 'oancf', 'lt', 'ni', 'gvkey', 'datadate']
    initial_len = len(df)
    df = df.dropna(subset=essential_cols)
    print(f"  Rows removed due to missing core financials: {initial_len - len(df):,}")

    # ----------------------------------------------------------------
    # Step 4: Feature Engineering
    # ----------------------------------------------------------------
    print("Step 4: Feature Engineering (Financial Ratios)...")

    # --- Denominators (safe division) ---
    at_adj  = df['at'].replace(0, np.nan)
    rev_adj = df['revt'].replace(0, np.nan)
    lt_adj  = df['lt'].replace(0, np.nan)
    lct_adj = df['lct'].replace(0, np.nan)   # for current_ratio

    # ---- ORIGINAL 11 FEATURES (v5, unchanged) ----
    df['profitability']   = df['ebitda'] / at_adj
    df['leverage']        = df['lt'] / at_adj
    df['cash_ratio']      = df['ch'].fillna(0) / at_adj
    df['fcf_debt']        = df['oancf'] / lt_adj
    df['ppe_ratio']       = df['ppent'].fillna(0) / at_adj
    df['capex_intensity'] = df['capx'].fillna(0) / at_adj
    df['asset_turnover']  = df['revt'] / at_adj
    df['interest_burden'] = df['xint'].fillna(0) / at_adj
    df['net_margin']      = df['ni'] / rev_adj

    df = df.sort_values(['gvkey', 'datadate'])
    df['rev_growth'] = df.groupby('gvkey')['revt'].pct_change()

    # ---- NEW FEATURES (v6) ----

    # 1. Firm size: log(total assets)
    #    Theory: Size hypothesis — smaller firms are disproportionately acquired
    #    (Palepu, 1986; Powell, 1997). No imputation needed; at > 0 guaranteed
    #    by essential_cols drop above.
    df['firm_size'] = np.log(at_adj)

    # 2. Book equity ratio: common equity / total assets
    #    Theory: Accounting-based undervaluation proxy — low book equity relative
    #    to assets signals potential undervaluation (Morck et al., 1988).
    #    ceq can be negative (distressed firms) — retain; no imputation.
    df['book_equity_ratio'] = df['ceq'].fillna(np.nan) / at_adj

    # 3. Current ratio: current assets / current liabilities
    #    Theory: Short-term liquidity — cash-rich, low-liability firms are
    #    self-financing post-acquisition (Ambrose & Megginson, 1992).
    #    ~29% of rows have null act/lct (financial sector firms with no
    #    working capital reporting). Propagate NaN — these rows are dropped
    #    at final cleanup if current_ratio is in features.
    df['current_ratio'] = df['act'].fillna(np.nan) / lct_adj

    # 4. R&D intensity: R&D expense / total assets
    #    Theory: Technology/IP acquisition motive. Firms with zero xrd genuinely
    #    do not report R&D spend — zero-imputed, not NaN.
    df['rd_intensity'] = df['xrd'].fillna(0) / at_adj

    # 5. Revenue growth lagged 1 year: one-period-prior rev_growth
    #    Theory: Sustained revenue decline is a stronger target signal than a
    #    single bad year (Powell, 1997; Tunyi & Ntim, 2016).
    df['rev_growth_lag1'] = df.groupby('gvkey')['rev_growth'].shift(1)

    # ---- ALTMAN Z-SCORE ACCOUNTING COMPONENTS ----
    # Constructed from accounting data only (no market value of equity).
    # Three of the five Altman components are purely balance-sheet / income-
    # statement based and can be computed from Compustat fundamentals.
    # The remaining two require market cap (excluded: market-based).
    # Theory: Financial distress proximity — lower Z-component values signal
    # acquisition attractiveness for distressed-firm acquirers (Altman, 1968;
    # Peel & Wilson, 1989).

    # Component 1: Working capital / total assets
    #    Working capital = current assets - current liabilities (wcap in Compustat).
    #    ~29% null for same financial-sector firms as act/lct.
    df['altman_wc_ta'] = df['wcap'].fillna(np.nan) / at_adj

    # Component 2: Retained earnings / total assets
    #    Proxy for cumulative profitability history. Negative values are valid
    #    (accumulated deficits). ~7% null.
    df['altman_re_ta'] = df['re'].fillna(np.nan) / at_adj

    # Component 3: EBIT / total assets
    #    Operating profitability independent of tax and capital structure.
    #    ~7% null. Note: overlaps conceptually with profitability (EBITDA/at)
    #    but uses EBIT not EBITDA — captures depreciation differently.
    df['altman_ebit_ta'] = df['ebit'].fillna(np.nan) / at_adj

    # ----------------------------------------------------------------
    # Step 5: FCF Volatility (unchanged from v5)
    # ----------------------------------------------------------------
    print("Step 5: Calculating 3-year FCF Volatility...")
    df['fcf_volatility'] = df.groupby('gvkey')['oancf'].transform(
        lambda x: x.rolling(window=3).std() / at_adj
    )

    # ----------------------------------------------------------------
    # Step 6: Final feature selection and cleanup
    # ----------------------------------------------------------------
    print("Step 6: Final Feature Cleanup...")

    # NOTE ON current_ratio / altman_wc_ta:
    # Both depend on act/lct/wcap which are null for ~29% of rows (primarily
    # financial sector firms). Including them in features will drop ~29% of
    # observations at the dropna step below. Two strategies are available:
    #
    #   Strategy A (CONSERVATIVE — recommended):
    #     Exclude current_ratio and altman_wc_ta from features. Retain all
    #     observations. Use the remaining 17 features.
    #
    #   Strategy B (AGGRESSIVE):
    #     Include current_ratio and altman_wc_ta. Accept ~29% row loss.
    #     Run sensitivity check: does model performance improve enough to
    #     justify the observation loss?
    #
    # v6 implements Strategy A by default. To switch to Strategy B, move
    # 'current_ratio' and 'altman_wc_ta' from FEATURES_OPTIONAL to FEATURES.

    FEATURES_CORE = [
        # --- Original 11 (v5) ---
        'profitability', 'leverage', 'cash_ratio', 'fcf_debt',
        'ppe_ratio', 'capex_intensity', 'asset_turnover',
        'interest_burden', 'net_margin', 'rev_growth', 'fcf_volatility',
        # --- New v6 (high coverage) ---
        'firm_size',          # log(at) — 0% null
        'book_equity_ratio',  # ceq/at — ~7% null
        'rd_intensity',       # xrd/at — 0% null (zero-imputed)
        'rev_growth_lag1',    # lagged rev_growth — ~1 extra year lost per firm
        'altman_re_ta',       # re/at — ~7% null
        'altman_ebit_ta',     # ebit/at — ~7% null
    ]

    FEATURES_OPTIONAL = [
        # Excluded by default due to ~29% null rate (financial sector firms)
        # Uncomment and move to FEATURES_CORE to test Strategy B
        # 'current_ratio',   # act/lct
        # 'altman_wc_ta',    # wcap/at
    ]

    FEATURES = FEATURES_CORE  # Change to FEATURES_CORE + FEATURES_OPTIONAL for Strategy B

    initial_len_final = len(df)
    df_final = df.dropna(subset=FEATURES + ['target_next_year'])
    df_final = df_final.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    print(f"  Rows removed during final cleanup (NaN / rolling window attrition): "
          f"{initial_len_final - len(df_final):,}")
    print(f"  Features in model: {len(FEATURES)} — {FEATURES}")

    # ----------------------------------------------------------------
    # Step 7: Save
    # ----------------------------------------------------------------
    print("Step 7: Saving Final Modeling Dataset...")

    # Preserve all ID and metadata columns plus features and target
    id_cols = ['gvkey', 'conm', 'tic', 'datadate', 'cusip', 'cusip6', 'cik', 'sic']
    keep_cols = [c for c in id_cols if c in df_final.columns] + FEATURES + ['target_next_year']
    df_out = df_final[keep_cols].copy()
    df_out.to_excel(output_file, index=False)

    # ----------------------------------------------------------------
    # Final reporting
    # ----------------------------------------------------------------
    unique_gvkeys  = df_out['gvkey'].nunique()
    total_obs      = len(df_out)
    total_targets  = int(df_out['target_next_year'].sum())
    base_rate_obs  = (total_targets / total_obs) * 100
    unique_targets = df_out[df_out['target_next_year'] == 1]['gvkey'].nunique()
    base_rate_firm = (unique_targets / unique_gvkeys) * 100

    print("\n--- PREPROCESSING REPORT v6 ---")
    print(f"Deal types included: {', '.join(RELEVANT_MA_TYPES)}")
    print(f"Total features: {len(FEATURES)}")
    print(f"  Original (v5): 11")
    print(f"  New (v6):      {len(FEATURES) - 11}")
    print(f"Final dataset size: {total_obs:,} firm-year observations.")
    print(f"Total unique firms (gvkeys): {unique_gvkeys:,}")
    print(f"Total M&A events (target_next_year=1): {total_targets}")
    print(f"Base Rate (Observation Level): {base_rate_obs:.2f}%")
    print(f"Base Rate (Firm Level): {base_rate_firm:.2f}%")
    print(f"Output saved as: {output_file}")
    print()
    print("--- NULL RATES FOR NEW FEATURES (post-cleanup) ---")
    for f in FEATURES[11:]:
        print(f"  {f}: 0 NaN (dropped at cleanup)")


if __name__ == "__main__":
    preprocess_ma_data(
        compustat_file='computstat.xlsx',
        refinitiv_file='refinitiv.xlsx',
        output_file='data_final_modeling_ma_v7.xlsx'
    )