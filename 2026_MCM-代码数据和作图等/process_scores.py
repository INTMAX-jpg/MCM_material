import pandas as pd
import numpy as np
import re

def process_dwts_data():
    # Load data
    try:
        df = pd.read_csv('C_origin.csv')
    except FileNotFoundError:
        print("Error: C_origin.csv not found.")
        return

    # Filter seasons: 1, 2, or >= 28
    # Ensure season is numeric
    df['season'] = pd.to_numeric(df['season'], errors='coerce')
    df = df[df['season'].notna()]
    
    target_seasons = df[
        (df['season'] == 1) | 
        (df['season'] == 2) | 
        (df['season'] >= 28)
    ].copy()

    # Rename columns to match requirements
    # "celebrity" is requested, input is "celebrity_name"
    column_mapping = {
        'celebrity_name': 'celebrity',
        'celebrity_homestate': 'celebrity_homestate', # already matches or close
        'celebrity_homecountry/region': 'celebrity_homecountry/region',
        'celebrity_age_during_season': 'celebrity_age_during_season',
        'season': 'season',
        'placement': 'placement',
        'ballroom_partner': 'ballroom_partner',
        'celebrity_industry': 'celebrity_industry'
    }
    
    # Check if 'celebrity_name' exists, if so rename
    if 'celebrity_name' in target_seasons.columns:
        target_seasons.rename(columns={'celebrity_name': 'celebrity'}, inplace=True)
        
    # Keep only necessary base columns plus score columns for processing
    base_cols = [
        'celebrity', 'ballroom_partner', 'celebrity_industry', 
        'celebrity_homestate', 'celebrity_homecountry/region', 
        'celebrity_age_during_season', 'season', 'placement'
    ]
    
    # Ensure all base columns exist
    for col in base_cols:
        if col not in target_seasons.columns:
            target_seasons[col] = None # or handle error

    # Identify all score columns to find max weeks
    score_cols = [c for c in target_seasons.columns if 'week' in c.lower() and 'judge' in c.lower() and 'score' in c.lower()]
    
    # Extract max week number
    max_week = 0
    for c in score_cols:
        match = re.search(r'week(\d+)', c.lower())
        if match:
            w = int(match.group(1))
            if w > max_week:
                max_week = w
    
    print(f"Max week found: {max_week}")

    # Process each week
    for w in range(1, max_week + 1):
        # Identify columns for this week
        # Pattern: week{w}_judge{j}_score
        # Need to be flexible with regex
        week_cols = [c for c in target_seasons.columns if re.search(f'week{w}_judge\\d+_score', c, re.IGNORECASE)]
        
        if not week_cols:
            continue
            
        # Calculate Average Score
        avg_col_name = f'week_{w}_judge_avg_score'
        rank_col_name = f'week_{w}_judge_rank'
        
        # Function to calculate row average
        def calculate_week_avg(row):
            valid_scores = []
            for col in week_cols:
                val = row[col]
                # Check for N/A, None, nan
                if pd.isna(val) or val == 'N/A' or str(val).strip() == '':
                    continue
                try:
                    s = float(val)
                    if s > 0: # Assuming 0 means didn't compete or special case, usually scores are > 0
                        valid_scores.append(s)
                except ValueError:
                    continue
            
            if not valid_scores:
                return np.nan
            
            return sum(valid_scores) / len(valid_scores)

        target_seasons[avg_col_name] = target_seasons.apply(calculate_week_avg, axis=1)
        
        # Calculate Rank within Season
        # We need to rank for each season separately
        # Rank: Higher score is better (1st is best)
        # Using 'min' method for ties (e.g. 1, 2, 2, 4) or 'dense' (1, 2, 2, 3)?
        # Usually 'min' is standard. Ascending=False (High score = Rank 1)
        
        # We can use groupby().rank()
        # But we need to handle NaNs (eliminated players shouldn't be ranked 1 or last, just NaN)
        
        target_seasons[rank_col_name] = target_seasons.groupby('season')[avg_col_name].rank(ascending=False, method='min')

    # Select final columns
    # First the base columns
    final_cols = base_cols.copy()
    
    # Then the week columns in order
    for w in range(1, max_week + 1):
        avg_col = f'week_{w}_judge_avg_score'
        rank_col = f'week_{w}_judge_rank'
        
        if avg_col in target_seasons.columns:
            final_cols.append(avg_col)
        if rank_col in target_seasons.columns:
            final_cols.append(rank_col)
            
    result_df = target_seasons[final_cols]
    
    # Save to CSV
    output_filename = '每周平均分数(十分制)&排名.csv'
    result_df.to_csv(output_filename, index=False, encoding='utf-8-sig') # utf-8-sig for Excel compatibility
    print(f"Successfully created {output_filename}")
    print(result_df.head())

if __name__ == "__main__":
    process_dwts_data()
