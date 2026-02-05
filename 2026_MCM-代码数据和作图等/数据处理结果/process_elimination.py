import pandas as pd
import numpy as np
import re

def process_elimination():
    # Load data
    try:
        df = pd.read_csv('C_origin.csv')
    except FileNotFoundError:
        print("Error: C_origin.csv not found.")
        return

    # Standardize columns
    if 'celebrity_name' in df.columns:
        df.rename(columns={'celebrity_name': 'celebrity'}, inplace=True)

    # Base columns to keep
    base_cols = [
        'celebrity', 'ballroom_partner', 'celebrity_industry', 
        'celebrity_homestate', 'celebrity_homecountry/region', 
        'celebrity_age_during_season', 'season', 'placement'
    ]
    
    # Ensure all base columns exist
    for col in base_cols:
        if col not in df.columns:
            df[col] = None

    # Identify score columns to find max weeks
    score_cols = [c for c in df.columns if 'week' in c.lower() and 'judge' in c.lower() and 'score' in c.lower()]
    
    max_week = 0
    for c in score_cols:
        match = re.search(r'week(\d+)', c.lower())
        if match:
            w = int(match.group(1))
            if w > max_week:
                max_week = w
    
    print(f"Max week found: {max_week}")

    # Prepare result dataframe
    result_df = df[base_cols].copy()

    # Function to determine elimination week
    def get_fail_week(row):
        # Check placement first
        # If placement is 1, they never failed
        try:
            p = float(row['placement'])
            if p == 1:
                return None
        except:
            pass # Continue if placement is not numeric or missing

        # Find last active week
        last_active = 0
        for w in range(1, max_week + 1):
            # Check if any score exists for this week
            # Pattern: week{w}_judge{j}_score
            # We need to find columns that match this week
            cols = [c for c in score_cols if re.search(f'week{w}_judge\\d+_score', c, re.IGNORECASE)]
            
            has_score = False
            for c in cols:
                val = row.get(c)
                if not (pd.isna(val) or val == 'N/A' or str(val).strip() == ''):
                    try:
                        if float(val) > 0:
                            has_score = True
                            break
                    except:
                        pass
            
            if has_score:
                last_active = w
            
        # If they never played (last_active=0), maybe return None or handle?
        if last_active == 0:
            return None
            
        return last_active

    # Apply to find fail week for each row
    fail_weeks = df.apply(get_fail_week, axis=1)

    # Create columns week_i_fail
    for w in range(1, max_week + 1):
        col_name = f'week_{w}_fail'
        # Set to 1 if fail_week == w, else 0
        result_df[col_name] = (fail_weeks == w).astype(int)

    # Save to CSV
    output_filename = '第几week淘汰.csv'
    result_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"Successfully created {output_filename}")
    print(result_df.head())
    
    # Show some examples of failed weeks
    print("\nExample Failures:")
    print(result_df[result_df['week_3_fail'] == 1][['celebrity', 'season', 'week_3_fail']].head())

if __name__ == "__main__":
    process_elimination()
