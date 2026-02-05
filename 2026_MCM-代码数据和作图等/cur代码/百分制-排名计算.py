import pandas as pd
import os
import re

# ==========================================
# Configuration
# ==========================================
judge_file = 'cur代码/百分比制-每周平均占比&排名.csv'
fan_file = 'cur代码/反向求百分比结果.csv'
output_file = 'cur代码/百分制-排名计算结果.csv'

def main():
    if not os.path.exists(judge_file) or not os.path.exists(fan_file):
        print("Error: Input files not found.")
        return

    # 1. Read Data
    df_judge = pd.read_csv(judge_file)
    df_fan = pd.read_csv(fan_file)
    
    # 2. Filter Seasons (3 to 27)
    df_judge = df_judge[(df_judge['season'] >= 3) & (df_judge['season'] <= 27)]
    df_fan = df_fan[(df_fan['season'] >= 3) & (df_fan['season'] <= 27)]
    
    # 3. Merge Data
    # We use the standard identification columns to merge
    # celebrity, season are the primary keys. 
    # To be safe, we merge on celebrity and season.
    
    # Rename fan columns to avoid confusion or prepare for easy access
    # Fan columns are like 'week1', 'week2'...
    # Judge columns are like 'week_1_judge_percentages'...
    
    # Let's keep the base info from df_judge (or df_fan, they should be similar)
    base_cols = [
        'celebrity', 'ballroom_partner', 'celebrity_industry', 
        'celebrity_homestate', 'celebrity_homecountry/region', 
        'celebrity_age_during_season', 'season', 'placement'
    ]
    
    # Ensure base cols exist
    for col in base_cols:
        if col not in df_judge.columns:
            print(f"Warning: Column {col} missing in judge file")
        if col not in df_fan.columns:
            print(f"Warning: Column {col} missing in fan file")
            
    # Merge
    # We select base cols + data columns from both
    # Actually, let's just merge everything on celebrity and season, and suffixes if needed
    df_merged = pd.merge(
        df_judge, 
        df_fan, 
        on=['celebrity', 'season'], 
        how='inner',
        suffixes=('_judge', '_fan')
    )
    
    # Resolve duplicated info columns (take from one side, drop the other)
    # The suffixes will affect columns like 'placement', 'ballroom_partner' if they exist in both
    # We want to reconstruct the final dataframe with clean base columns
    
    # Let's handle columns manually to be clean
    final_df = df_merged.copy()
    
    # Fix base columns names if they got suffixed
    for col in base_cols:
        if col not in final_df.columns:
            # check if col_judge exists
            if f'{col}_judge' in final_df.columns:
                final_df[col] = final_df[f'{col}_judge']
            elif f'{col}_fan' in final_df.columns:
                final_df[col] = final_df[f'{col}_fan']
    
    # Find max weeks
    # Scan for 'week_i_judge_percentages'
    max_week = 0
    for col in df_judge.columns:
        m = re.match(r'week_(\d+)_judge_percentages', col)
        if m:
            w = int(m.group(1))
            if w > max_week: max_week = w
            
    print(f"Max week found: {max_week}")
    
    # Calculate Ranks
    # We need to process season by season, week by week
    
    # Dictionary to store new rank columns
    rank_columns = {} 
    
    # Initialize rank columns with NaN
    for w in range(1, max_week + 1):
        final_df[f'week{w}排名'] = float('nan')
        
    # Iterate over each season
    seasons = final_df['season'].unique()
    
    for s in seasons:
        season_df = final_df[final_df['season'] == s]
        
        for w in range(1, max_week + 1):
            # Column names
            # Judge: week_{w}_judge_percentages (from original judge file)
            # Fan: week{w} (from original fan file)
            # Since we merged, we need to find these columns in final_df
            
            # Judge col might be 'week_1_judge_percentages' (if unique) or suffixed
            col_judge = f'week_{w}_judge_percentages'
            col_fan = f'week{w}'
            
            # Check if columns exist in merged df
            # Note: merge might have renamed them if they collided, but these names are distinct between two files
            # week_i_judge_percentages vs weeki
            # So they should remain as is.
            
            if col_judge in final_df.columns and col_fan in final_df.columns:
                # Get data for this season and week
                # We work on a slice indices
                indices = season_df.index
                
                # Extract values
                j_vals = final_df.loc[indices, col_judge]
                f_vals = final_df.loc[indices, col_fan]
                
                # Check valid data (not NaN)
                # Participants still in competition should have scores
                # We filter for rows where both are not NaN
                valid_mask = j_vals.notna() & f_vals.notna()
                
                if not valid_mask.any():
                    continue
                    
                valid_indices = indices[valid_mask]
                
                # Calculate Total Score
                # Formula: Judge% + Fan%
                # Input format: 0.08 -> 8%
                # So we multiply by 100
                
                scores = final_df.loc[valid_indices, col_judge] * 100 + final_df.loc[valid_indices, col_fan] * 100
                
                # Rank Descending
                # method='min' means if tie, use min rank? Or 'average'?
                # Usually standard competition ranking: 1, 2, 2, 4... -> method='min'
                # Or 1, 2, 3... -> method='first' (force unique)
                # User says: "总分数越高，排名越靠前"
                # Let's use standard dense ranking or min. Let's use 'min' (1, 2, 2, 4) or 'average'.
                # Given "每周排名最后的选手判定为被淘汰", ties might be an issue.
                # Let's use 'min' for now, consistent with typical leaderboards.
                ranks = scores.rank(ascending=False, method='min')
                
                # Update final_df
                final_df.loc[valid_indices, f'week{w}排名'] = ranks
                
    # 4. Prepare Output
    # Select columns
    output_cols = base_cols + [f'week{w}排名' for w in range(1, max_week + 1)]
    
    # Filter columns that exist
    output_cols = [c for c in output_cols if c in final_df.columns]
    
    df_out = final_df[output_cols]
    
    # Save
    df_out.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Successfully saved to {output_file}")
    
    # Verification
    print("Sample Output (first 5 rows):")
    print(df_out.head())

if __name__ == "__main__":
    main()
