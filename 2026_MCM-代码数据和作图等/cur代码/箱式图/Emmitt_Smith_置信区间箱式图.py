import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# ==========================================
# Configuration
# ==========================================
mc_file = 'cur代码/反向求百分比_置信区间.csv'
lp_file = 'cur代码/百分比-区间估计.csv'
output_file = 'cur代码/Emmitt_Smith_置信区间箱式图.png'

target_celebrity = "Emmitt Smith"
target_season = 3

# Colors
rank_colors = [
    (77/255, 103/255, 164/255),    # #4D67A4 (rank1)
    (140/255, 141/255, 197/255),   # #8C8DC5 (rank2)
    (186/255, 168/255, 210/255),   # #BAA8D2 (rank3)
    (237/255, 187/255, 199/255),   # #EDBBC7 (rank4)
    (255/255, 204/255, 154/255),   # #FFCC9A (rank5)
    (246/255, 162/255, 126/255),   # #F6A27E (rank6)
    (189/255, 115/255, 106/255),   # #BD736A (rank7)
    (121/255, 77/255, 72/255)      # #794D48 (rank8)
]

def main():
    if not os.path.exists(mc_file) or not os.path.exists(lp_file):
        print(f"Error: Input files not found.")
        return

    # 1. Read Data
    df_mc = pd.read_csv(mc_file)
    df_lp = pd.read_csv(lp_file)
    
    # 2. Filter for Emmitt Smith
    row_mc = df_mc[(df_mc['celebrity'] == target_celebrity) & (df_mc['season'] == target_season)]
    row_lp = df_lp[(df_lp['celebrity'] == target_celebrity) & (df_lp['season'] == target_season)]
    
    if row_mc.empty or row_lp.empty:
        print(f"Error: No data found for {target_celebrity} in Season {target_season}")
        return
        
    record_mc = row_mc.iloc[0]
    record_lp = row_lp.iloc[0]
    
    # 3. Extract Weekly Data
    # We need:
    # - Weeks
    # - Box Bottom: 25% CI from MC file
    # - Box Top: 75% CI from MC file (Wait, user said "用25%CI到50%CI区间的数据")
    #   User instruction: "用25%CI到50%CI区间的数据，但是图片的标识要写是“75%置信区间”"
    #   Interpretation: Box bottom = 25% CI, Box top = 75% CI. (Usually box is IQR, 25-75). 
    #   Let's check user text again: "用25%CI到50%CI区间的数据" -> This is weird. 
    #   Maybe typo for "25% to 75%"? Or user wants the box to represent 25%-50%?
    #   But then "图片的标识要写是“75%置信区间”" -> Usually 75% CI means central 75%? Or upper bound is 75% percentile?
    #   
    #   Let's re-read carefully:
    #   "箱体部分... 用25%CI到50%CI区间的数据"
    #   "但是图片的标识要写是“75%置信区间”"
    #   "箱式图的下限和上限，选用百分比-区间估计.csv中对应的区间" (This refers to whiskers/caps)
    #
    #   Standard Boxplot:
    #   - Box: Q1 (25%) to Q3 (75%). Median inside.
    #   - Whiskers: Min to Max (or 1.5 IQR).
    #
    #   User Request Interpretation:
    #   - Whiskers (Lower/Upper Caps): week_i_min_fan_percentage / week_i_max_fan_percentage from LP file.
    #   - Box: 
    #       - Bottom: week_i_reverse_25%CI (from MC file)
    #       - Top: week_i_reverse_75%CI (from MC file) -> Assuming "50%CI" was a typo for "75%" because standard box is 25-75, and user wants label "75% CI".
    #       - OR: User literally wants 25% to 50%.
    #       - Let's look at "75%置信区间" label. If box is 25-75, it covers 50% probability mass. 
    #       - If user meant "50% CI" (the central 50%), that IS 25%-75%.
    #       - So I will use 25% and 75% columns from MC file for the box.
    #
    
    weeks = []
    box_stats = [] # List of dicts for bxp
    
    # Scan columns
    # Find max week
    max_week = 0
    for col in df_mc.columns:
        m = re.match(r'第(\d+)周_反向均值', col)
        if m:
            w = int(m.group(1))
            if w > max_week: max_week = w
            
    print(f"Found weeks up to {max_week}")
    
    for w in range(1, max_week + 1):
        # MC columns
        col_mean = f'第{w}周_反向均值'
        col_25 = f'第{w}周_反向25%CI'
        col_75 = f'第{w}周_反向75%CI'
        
        # LP columns
        col_min = f'week_{w}_min_fan_percentage'
        col_max = f'week_{w}_max_fan_percentage'
        col_pt = f'week_{w}_point_estimate' # Use as median if needed, or use mean from MC
        
        if col_mean in df_mc.columns and pd.notna(record_mc[col_mean]):
            # Get values
            q1 = record_mc[col_25]
            q3 = record_mc[col_75]
            mean_val = record_mc[col_mean]
            
            # Whiskers from LP
            # Note: LP min/max might be wider or narrower than MC CI? 
            # Usually LP is the hard constraint [min, max]. MC explores within it.
            # So LP min <= MC min <= MC max <= LP max.
            # We use LP min/max as whiskers as requested.
            whislo = record_lp[col_min] if col_min in df_lp.columns and pd.notna(record_lp[col_min]) else q1
            whishi = record_lp[col_max] if col_max in df_lp.columns and pd.notna(record_lp[col_max]) else q3
            
            # Ensure consistency
            whislo = min(whislo, q1)
            whishi = max(whishi, q3)
            
            # Create stat dict for matplotlib.cbook.BoxplotStats or direct drawing
            # We can use ax.bxp which takes a list of dicts
            stats = {
                'whislo': whislo,    # Bottom whisker
                'q1': q1,            # Box bottom
                'med': mean_val,     # Median/Mean line (User didn't specify median, let's use mean as central line)
                'q3': q3,            # Box top
                'whishi': whishi,    # Top whisker
                'fliers': [],        # No outliers
                'label': str(w)      # X-axis label
            }
            
            weeks.append(w)
            box_stats.append(stats)
            
    if not weeks:
        print("Error: No weekly data found.")
        return
        
    # 4. Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Customizing the boxplot
    # ax.bxp returns a dictionary mapping each component to a list of the matplotlib.lines.Line2D instances created.
    bp = ax.bxp(box_stats, patch_artist=True, showmeans=False, showfliers=False, positions=weeks)
    
    # Style
    # Box color - Cycle through rank_colors
    for i, box in enumerate(bp['boxes']):
        color_idx = i % len(rank_colors)
        box.set_facecolor(rank_colors[color_idx])
        box.set_alpha(0.8)
        box.set_edgecolor('black')
        
    # Whisker color
    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.5)
        
    # Cap color
    for cap in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.5)
        
    # Median color
    for median in bp['medians']:
        median.set_color('white') # Make it visible on the dark box
        median.set_linewidth(2)
        
    # Labels
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Fan Vote Percentage', fontsize=12)
    ax.set_title(f'Fan Vote Prediction Boxplot (75% Confidence Interval): {target_celebrity} (Season {target_season})', fontsize=14)
    
    # Add grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Legend
    # Create a dummy patch for legend
    import matplotlib.patches as mpatches
    # Use rank1 color for legend representative
    patch = mpatches.Patch(color=rank_colors[0], label='75% Confidence Interval (MC 25%-75%)')
    line = plt.Line2D([0], [0], color='black', lw=1.5, label='Feasible Range (LP Min-Max)')
    ax.legend(handles=[patch, line], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved boxplot to {output_file}")

if __name__ == "__main__":
    main()
