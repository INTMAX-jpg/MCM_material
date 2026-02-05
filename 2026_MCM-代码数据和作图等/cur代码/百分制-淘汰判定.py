import pandas as pd
import os
import re
import numpy as np

# ==========================================
# Configuration
# ==========================================
input_file = 'cur代码/百分制-排名计算结果.csv'
output_file = 'cur代码/百分制-模型-淘汰结果.csv'

def main():
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    # 1. Read Data
    df = pd.read_csv(input_file)
    
    # 2. Identify Rank Columns
    # Format: week{i}排名
    rank_cols = []
    max_week = 0
    
    for col in df.columns:
        m = re.match(r'week(\d+)排名', col)
        if m:
            w = int(m.group(1))
            rank_cols.append((w, col))
            if w > max_week: max_week = w
            
    # Sort by week number
    rank_cols.sort(key=lambda x: x[0])
    
    # 3. Initialize Fail Columns
    # User requested: week_i_fail
    for w, _ in rank_cols:
        df[f'week_{w}_fail'] = 0 # Default 0 (not eliminated)
        
    # 4. Determine Eliminations
    seasons = df['season'].unique()
    
    for s in seasons:
        season_df = df[df['season'] == s]
        
        for w, rank_col in rank_cols:
            # Get ranks for this season and week
            # We only care about rows that have a rank (meaning they are still in competition)
            current_ranks = season_df[rank_col]
            
            # Filter out NaNs (already eliminated or didn't start)
            valid_ranks = current_ranks.dropna()
            
            if valid_ranks.empty:
                continue
                
            # Find the max rank (worst rank)
            # Higher number = worse rank (e.g. Rank 10 is worse than Rank 1)
            worst_rank = valid_ranks.max()
            
            # Identify candidates for elimination (all who have the worst rank)
            eliminated_indices = valid_ranks[valid_ranks == worst_rank].index
            
            # Mark them in the main dataframe
            # Note: week_i_fail is an integer column (0 or 1)
            df.loc[eliminated_indices, f'week_{w}_fail'] = 1
            
    # 5. Prepare Output
    # Columns:
    # - Base columns (from user request)
    # - week_i_fail columns
    
    base_cols = [
        'celebrity', 
        'ballroom_partner', 
        'celebrity_industry', 
        'celebrity_homestate', 
        'celebrity_homecountry/region', 
        'celebrity_age_during_season', 
        'season', 
        'placement'
    ]
    
    fail_cols = [f'week_{w}_fail' for w, _ in rank_cols]
    
    # Ensure base columns exist
    final_cols = [c for c in base_cols if c in df.columns] + fail_cols
    
    df_out = df[final_cols]
    
    # Save
    df_out.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Successfully saved to {output_file}")
    print("Sample Output (first 5 rows):")
    print(df_out.head())

if __name__ == "__main__":
    main()
