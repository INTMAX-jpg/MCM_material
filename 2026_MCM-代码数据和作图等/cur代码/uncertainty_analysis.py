import pandas as pd
import os
import re
import numpy as np

# ==========================================
# Configuration
# ==========================================
lp_file = 'cur代码/百分比-区间估计.csv'
mc_file = 'cur代码/反向求百分比_置信区间.csv'
output_file = 'cur代码/粉丝投票预测-确定性分析.csv'

def main():
    if not os.path.exists(lp_file) or not os.path.exists(mc_file):
        print("Error: Input files not found.")
        return

    # 1. Read Data
    df_lp = pd.read_csv(lp_file)
    df_mc = pd.read_csv(mc_file)
    
    # 2. Merge Data
    # Merge on celebrity and season
    merge_keys = ['celebrity', 'season']
    
    # Select only necessary columns from LP to avoid clutter
    # We need week_i_delta
    lp_cols = merge_keys + [c for c in df_lp.columns if 'delta' in c]
    df_lp_subset = df_lp[lp_cols].copy()
    
    # Select necessary columns from MC
    # We need Mean, 25%, 75%
    # Columns are: 第i周_反向均值, 第i周_反向25%CI, 第i周_反向75%CI
    mc_cols = merge_keys + [c for c in df_mc.columns if '反向' in c]
    df_mc_subset = df_mc[mc_cols].copy()
    
    df_merged = pd.merge(df_lp_subset, df_mc_subset, on=merge_keys, how='inner')
    
    # 3. Calculate Uncertainty Metrics
    # Find max week
    max_week = 0
    for col in df_merged.columns:
        m = re.search(r'week_(\d+)_delta', col)
        if m:
            w = int(m.group(1))
            if w > max_week: max_week = w
            
    print(f"Analyzing up to Week {max_week}")
    
    # Create result dataframe
    # We'll keep the identification columns
    res_cols = merge_keys + ['placement', 'celebrity_industry', 'celebrity_homecountry/region']
    # Check if extra cols exist in original df_lp
    extra_cols = [c for c in res_cols if c not in df_merged.columns]
    if extra_cols:
        # Fetch from original df_lp
        temp = df_lp[merge_keys + extra_cols]
        df_merged = pd.merge(df_merged, temp, on=merge_keys, how='left')
    
    df_out = df_merged[res_cols].copy()
    
    for w in range(1, max_week + 1):
        col_delta = f'week_{w}_delta'
        col_mean = f'第{w}周_反向均值'
        col_25 = f'第{w}周_反向25%CI'
        col_75 = f'第{w}周_反向75%CI'
        
        if col_delta in df_merged.columns and col_mean in df_merged.columns:
            # 1. Feasible Range (Absolute Uncertainty)
            # Already in delta
            df_out[f'Week{w}_Feasible_Width'] = df_merged[col_delta]
            
            # 2. IQR (Statistical Uncertainty)
            iqr = df_merged[col_75] - df_merged[col_25]
            df_out[f'Week{w}_IQR'] = iqr
            
            # 3. Relative Uncertainty (CV-like)
            # IQR / Mean
            # Avoid division by zero
            mean_val = df_merged[col_mean]
            rel_unc = iqr / mean_val.replace(0, np.nan)
            df_out[f'Week{w}_Relative_Uncertainty'] = rel_unc
            
            # 4. Certainty Score (Normalized 0-1?)
            # Hard to normalize without global context. 
            # Let's just output the raw metrics for now.
            
    # Save
    df_out.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Successfully saved to {output_file}")
    print("Sample Output (first 5 rows):")
    print(df_out.head())

if __name__ == "__main__":
    main()
