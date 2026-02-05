import pandas as pd
import numpy as np
import re

def process_dwts_percent():
    # Load data
    try:
        df = pd.read_csv('C_origin.csv')
    except FileNotFoundError:
        print("Error: C_origin.csv not found.")
        return

    # Filter seasons: 3 <= season <= 27
    df['season'] = pd.to_numeric(df['season'], errors='coerce')
    df = df[df['season'].notna()]
    
    target_seasons = df[
        (df['season'] >= 3) & 
        (df['season'] <= 27)
    ].copy()

    if target_seasons.empty:
        print("No data found for seasons 3-27.")
        return

    # Rename columns to match requirements
    column_mapping = {
        'celebrity_name': 'celebrity',
    }
    
    if 'celebrity_name' in target_seasons.columns:
        target_seasons.rename(columns={'celebrity_name': 'celebrity'}, inplace=True)
        
    # Base columns to keep
    base_cols = [
        'celebrity', 'ballroom_partner', 'celebrity_industry', 
        'celebrity_homestate', 'celebrity_homecountry/region', 
        'celebrity_age_during_season', 'season', 'placement'
    ]
    
    # Ensure all base columns exist
    for col in base_cols:
        if col not in target_seasons.columns:
            target_seasons[col] = None 

    # Identify all score columns to find max weeks
    score_cols = [c for c in target_seasons.columns if 'week' in c.lower() and 'judge' in c.lower() and 'score' in c.lower()]
    
    max_week = 0
    for c in score_cols:
        match = re.search(r'week(\d+)', c.lower())
        if match:
            w = int(match.group(1))
            if w > max_week:
                max_week = w
    
    print(f"Max week found: {max_week}")

    # Helper to calculate total score for a row/week
    def get_week_total_score(row, week_idx):
        # Find columns for this week
        cols = [c for c in score_cols if re.search(f'week{week_idx}_judge\\d+_score', c, re.IGNORECASE)]
        total = 0
        has_score = False
        for c in cols:
            val = row[c]
            if pd.isna(val) or val == 'N/A' or str(val).strip() == '':
                continue
            try:
                s = float(val)
                total += s
                has_score = True
            except:
                continue
        return total if has_score else 0

    # Process each week
    for w in range(1, max_week + 1):
        # 1. Calculate raw total score for each celebrity in this week
        # We store this temporarily to calculate the season sum
        temp_score_col = f'temp_week_{w}_total'
        target_seasons[temp_score_col] = target_seasons.apply(lambda row: get_week_total_score(row, w), axis=1)
        
        # 2. Calculate Season Total for this week (Sum of all celebrities' scores in that season)
        # Group by season and sum the temp score
        season_week_sums = target_seasons.groupby('season')[temp_score_col].transform('sum')
        
        # 3. Calculate Percentage
        # Percentage = Individual Score / Season Sum
        pct_col_name = f'week_{w}_judge_percentages'
        
        # Avoid division by zero
        target_seasons[pct_col_name] = np.where(
            season_week_sums > 0,
            target_seasons[temp_score_col] / season_week_sums,
            np.nan # Or 0? Usually if sum is 0, no one played, so NaN
        )
        
        # If the individual score was 0 (eliminated), the percentage is 0.
        # But if they didn't participate (NaN/N/A in input), get_week_total_score returns 0.
        # We might want to distinguish between "Scored 0" and "Not present".
        # But for now, 0 score -> 0 percentage is correct.
        
        # However, if a user has 0 score, we might want to set percentage to NaN if they are already eliminated?
        # The user didn't specify, but usually eliminated players don't get a rank.
        # Let's check if the raw score sum was 0.
        target_seasons.loc[target_seasons[temp_score_col] == 0, pct_col_name] = np.nan

        # 4. Calculate Rank
        # Rank based on percentage descending
        rank_col_name = f'week_{w}_judge_rank'
        target_seasons[rank_col_name] = target_seasons.groupby('season')[pct_col_name].rank(ascending=False, method='min')

    # Select final columns
    final_cols = base_cols.copy()
    
    for w in range(1, max_week + 1):
        pct_col = f'week_{w}_judge_percentages'
        rank_col = f'week_{w}_judge_rank'
        
        if pct_col in target_seasons.columns:
            final_cols.append(pct_col)
        if rank_col in target_seasons.columns:
            final_cols.append(rank_col)
            
    result_df = target_seasons[final_cols]
    
    # Save to CSV
    output_filename = '百分比制-每周平均占比&排名.csv'
    result_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"Successfully created {output_filename}")
    print(result_df.head())

if __name__ == "__main__":
    process_dwts_percent()
