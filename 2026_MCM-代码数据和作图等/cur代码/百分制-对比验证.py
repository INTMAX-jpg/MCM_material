import pandas as pd
import os
import re

# ==========================================
# Configuration
# ==========================================
model_file = 'cur代码/百分制-模型-淘汰结果.csv'
ground_truth_file = 'cur代码/第几week淘汰.csv'
output_file = 'cur代码/百分制淘汰-和真实情况对比.csv'

def main():
    if not os.path.exists(model_file) or not os.path.exists(ground_truth_file):
        print("Error: Input files not found.")
        return

    # 1. Read Data
    df_model = pd.read_csv(model_file)
    df_truth = pd.read_csv(ground_truth_file)
    
    # 2. Merge Data
    # Merge on celebrity and season
    # We want to keep all rows from model file (Season 3-27)
    # Ground truth file likely contains all seasons.
    
    # Check identifying columns
    merge_keys = ['celebrity', 'season']
    
    # Ensure they exist
    for k in merge_keys:
        if k not in df_model.columns or k not in df_truth.columns:
            print(f"Error: Missing key column {k}")
            return
            
    # Merge
    # Suffixes: _model for model prediction, _truth for ground truth
    df_merged = pd.merge(
        df_model, 
        df_truth, 
        on=merge_keys, 
        how='left', 
        suffixes=('', '_truth')
    )
    
    # 3. Compare and Update
    # Iterate over week columns in model file
    # Format: week_i_fail
    
    # Find all week fail columns in model
    week_cols = []
    max_week = 0
    for col in df_model.columns:
        m = re.match(r'week_(\d+)_fail', col)
        if m:
            w = int(m.group(1))
            week_cols.append((w, col))
            if w > max_week: max_week = w
            
    week_cols.sort(key=lambda x: x[0])
    
    # Create output dataframe based on model dataframe structure
    # We will modify the values in place or create a copy
    df_out = df_model.copy()
    
    # For each week column
    for w, col_model in week_cols:
        col_truth = f'week_{w}_fail_truth' # Based on merge suffix
        # The original truth file had 'week_i_fail', so after merge with empty suffix for left, 
        # it might have collided if we didn't specify suffixes correctly.
        # Wait, df_model has 'week_1_fail'. df_truth has 'week_1_fail'.
        # merge(df_model, df_truth, suffixes=('', '_truth'))
        # So df_model's col remains 'week_1_fail'.
        # df_truth's col becomes 'week_1_fail_truth'.
        
        if col_truth not in df_merged.columns:
            # Maybe the column naming in truth file is different?
            # Let's check truth file columns from Read tool output:
            # week_1_fail, week_2_fail...
            # So yes, it should be 'week_1_fail_truth'.
            
            # However, ground truth might not have all weeks if model goes up to 11 and truth only to 10?
            # Let's handle missing truth columns gracefully
            print(f"Warning: Truth column {col_truth} not found. Skipping week {w}.")
            # If truth is missing, we can't verify. Leave as is or mark unknown?
            # User said: "若淘汰的week匹配匹配，将对应格改成correct，反之，改成ERROR"
            # If no truth, we can't say correct or error.
            # Assuming files are consistent.
            continue
            
        # Comparison Logic
        # We need to compare specific cells where elimination happened?
        # User says: "若淘汰的week匹配匹配"
        # Does this mean:
        # Case 1: Model says Elim (1), Truth says Elim (1) -> Correct
        # Case 2: Model says Elim (1), Truth says Safe (0) -> Error
        # Case 3: Model says Safe (0), Truth says Elim (1) -> Error (Missed elimination)
        # Case 4: Model says Safe (0), Truth says Safe (0) -> Correct (Safe prediction correct)
        
        # User phrasing: "若淘汰的week匹配匹配，将对应格改成correct，反之，改成ERROR"
        # This implies we are looking at the cell in the output csv corresponding to that week.
        # So we replace 0/1 with 'correct'/'ERROR'.
        
        # Let's iterate rows
        for idx in df_out.index:
            val_model = df_merged.loc[idx, col_model]
            val_truth = df_merged.loc[idx, col_truth]
            
            # Handle NaNs (e.g. if celebrity not in truth file or data missing)
            if pd.isna(val_truth):
                # If truth is missing, we can't judge.
                # Maybe keep original value or empty string?
                # Let's keep original value (0/1) but convert column to object type first
                continue
                
            # Convert to int for comparison if possible
            try:
                vm = int(val_model) if pd.notna(val_model) else 0
                vt = int(val_truth) if pd.notna(val_truth) else 0
            except:
                vm = 0
                vt = 0
                
            # Logic:
            # If both are 1 (Eliminated in both) -> Correct
            # If both are 0 (Safe in both) -> Correct
            # If different -> ERROR
            
            # Wait, "若淘汰的week匹配" could specifically mean:
            # "If the week they were eliminated matches".
            # But the structure is a grid of weeks.
            # So for each cell:
            # - If model predicts elimination (1) AND truth is elimination (1) -> Correct
            # - If model predicts safe (0) AND truth is safe (0) -> Correct
            # - If mismatch -> ERROR
            
            # Let's apply this.
            
            # Note: We need to change the column type to string/object to hold 'correct'/'ERROR'
            
            if vm == vt:
                res = 'correct'
            else:
                res = 'ERROR'
                
            # We can't update in place efficiently if mixed types.
            # So let's store results in a temporary list or dict and update column at once.
            pass
            
    # Efficient update
    # Convert all week columns to object type
    for w, col in week_cols:
        df_out[col] = df_out[col].astype(object)
        
        col_truth = f'week_{w}_fail_truth'
        if col_truth not in df_merged.columns:
            continue
            
        # Vectorized comparison
        # 1. Exact match (0==0 or 1==1) -> 'correct'
        # 2. Mismatch -> 'ERROR'
        # 3. Handle NaNs in truth -> keep original or mark unknown? Let's assume 'correct' if we lack info? No, 'ERROR' or keep original.
        # Let's assume data is complete.
        
        # Create masks
        # Fill NaNs with -1 to ensure mismatch if data missing
        s_model = df_merged[col].fillna(-1).astype(int)
        s_truth = df_merged[col_truth].fillna(-2).astype(int)
        
        matches = (s_model == s_truth)
        
        df_out.loc[matches, col] = 'correct'
        df_out.loc[~matches, col] = 'ERROR'
        
    # Save
    df_out.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Successfully saved to {output_file}")
    print("Sample Output (first 5 rows):")
    print(df_out.head())

if __name__ == "__main__":
    main()
