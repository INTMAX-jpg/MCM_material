import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import numpy as np
from scipy.interpolate import make_interp_spline

# ==========================================
# Configuration
# ==========================================
mc_file = 'cur代码/反向求百分比_置信区间.csv'
lp_file = 'cur代码/百分比-区间估计.csv'
pe_file = 'cur代码/反向求百分比结果.csv'
output_file = 'cur代码/Emmith-加上曲线.png'

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
    if not os.path.exists(mc_file) or not os.path.exists(lp_file) or not os.path.exists(pe_file):
        print(f"Error: Input files not found.")
        return

    # 1. Read Data
    df_mc = pd.read_csv(mc_file)
    df_lp = pd.read_csv(lp_file)
    df_pe = pd.read_csv(pe_file)
    
    # 2. Filter for Emmitt Smith
    row_mc = df_mc[(df_mc['celebrity'] == target_celebrity) & (df_mc['season'] == target_season)]
    row_lp = df_lp[(df_lp['celebrity'] == target_celebrity) & (df_lp['season'] == target_season)]
    row_pe = df_pe[(df_pe['celebrity'] == target_celebrity) & (df_pe['season'] == target_season)]
    
    if row_mc.empty or row_lp.empty or row_pe.empty:
        print(f"Error: No data found for {target_celebrity} in Season {target_season}")
        return
        
    record_mc = row_mc.iloc[0]
    record_lp = row_lp.iloc[0]
    record_pe = row_pe.iloc[0]
    
    # 3. Extract Weekly Data
    weeks = []
    box_stats = [] # List of dicts for bxp
    point_estimates = []
    
    # Scan columns
    # Find max week
    max_week = 0
    for col in df_mc.columns:
        m = re.match(r'第(\d+)周_反向均值', col)
        if m:
            w = int(m.group(1))
            if w > max_week: max_week = w
            
    print(f"Found weeks up to {max_week}")
    
    valid_weeks_pe = []
    
    for w in range(1, max_week + 1):
        # MC columns
        col_mean = f'第{w}周_反向均值'
        col_25 = f'第{w}周_反向25%CI'
        col_75 = f'第{w}周_反向75%CI'
        
        # LP columns
        col_min = f'week_{w}_min_fan_percentage'
        col_max = f'week_{w}_max_fan_percentage'
        
        # PE columns (Point Estimate from 反向求百分比结果.csv)
        col_pe = f'week{w}'
        
        if col_mean in df_mc.columns and pd.notna(record_mc[col_mean]):
            # Get values
            q1 = record_mc[col_25]
            q3 = record_mc[col_75]
            mean_val = record_mc[col_mean]
            
            # Whiskers from LP
            whislo = record_lp[col_min] if col_min in df_lp.columns and pd.notna(record_lp[col_min]) else q1
            whishi = record_lp[col_max] if col_max in df_lp.columns and pd.notna(record_lp[col_max]) else q3
            
            # Ensure consistency
            whislo = min(whislo, q1)
            whishi = max(whishi, q3)
            
            # Create stat dict
            stats = {
                'whislo': whislo,    # Bottom whisker
                'q1': q1,            # Box bottom
                'med': mean_val,     # Median/Mean line
                'q3': q3,            # Box top
                'whishi': whishi,    # Top whisker
                'fliers': [],        # No outliers
                'label': str(w)      # X-axis label
            }
            
            weeks.append(w)
            box_stats.append(stats)
            
            # Get Point Estimate
            if col_pe in df_pe and pd.notna(record_pe[col_pe]):
                val = record_pe[col_pe]
                point_estimates.append(val)
                valid_weeks_pe.append(w)
            else:
                # If PE is missing for this week but box exists, append None or handle?
                # Assuming data consistency, but let's be safe
                pass
            
    if not weeks:
        print("Error: No weekly data found.")
        return
        
    # 4. Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Customizing the boxplot
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
        
    # 5. Add Smooth Curve for Point Estimates
    if len(valid_weeks_pe) >= 3:
        # Create smooth line using spline
        x_new = np.linspace(min(valid_weeks_pe), max(valid_weeks_pe), 300)
        spl = make_interp_spline(valid_weeks_pe, point_estimates, k=3) # Cubic spline
        y_smooth = spl(x_new)
        
        # Plot smooth line
        ax.plot(x_new, y_smooth, color='black', linewidth=2.5, linestyle='-', zorder=10, label='Point Estimate (Smoothed)')
        # Optional: Plot original points to show fit
        # ax.scatter(valid_weeks_pe, point_estimates, color='black', s=30, zorder=11)
    elif len(valid_weeks_pe) > 0:
        # Just plot line if not enough points for spline
        ax.plot(valid_weeks_pe, point_estimates, color='black', linewidth=2.5, linestyle='-', zorder=10, label='Point Estimate')
        
    # Labels
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Fan Vote Percentage', fontsize=12)
    ax.set_title(f'Fan Vote Prediction Boxplot with Point Estimate: {target_celebrity} (Season {target_season})', fontsize=14)
    
    # Add grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Legend
    import matplotlib.patches as mpatches
    # Use rank1 color for legend representative
    patch = mpatches.Patch(color=rank_colors[0], label='75% Confidence Interval (MC 25%-75%)')
    line_feasible = plt.Line2D([0], [0], color='black', lw=1.5, label='Feasible Range (LP Min-Max)')
    line_smooth = plt.Line2D([0], [0], color='black', lw=2.5, linestyle='-', label='Point Estimate (Smoothed)')
    
    ax.legend(handles=[patch, line_feasible, line_smooth], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved boxplot to {output_file}")

if __name__ == "__main__":
    main()
