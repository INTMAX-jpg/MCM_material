import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import make_interp_spline

# ==========================================
# Configuration
# ==========================================
input_file = 'cur代码/粉丝投票预测-确定性分析.csv'
output_file = 'cur代码/Emmitt_Smith_uncertainty.png'

# Colors specified by user
# rank2_color for IQR
# rank5_color for Relative Uncertainty
color_iqr = (140/255, 141/255, 197/255)   # #8C8DC5
color_rel = (255/255, 204/255, 154/255)   # #FFCC9A

def main():
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # 1. Read Data
    df = pd.read_csv(input_file)
    
    # 2. Filter for Emmitt Smith, Season 3
    target_celebrity = "Emmitt Smith"
    target_season = 3
    
    row = df[(df['celebrity'] == target_celebrity) & (df['season'] == target_season)]
    
    if row.empty:
        print(f"No data found for {target_celebrity} in Season {target_season}")
        return
    
    # 3. Extract Time Series Data
    weeks = []
    iqr = []
    relative_uncertainty = []
    
    # Iterate through columns to find week data
    # Assuming columns like 'Week1_Feasible_Width', etc.
    # Find max week
    max_week = 0
    import re
    for col in df.columns:
        m = re.match(r'Week(\d+)_Feasible_Width', col)
        if m:
            w = int(m.group(1))
            if w > max_week:
                max_week = w
    
    for w in range(1, max_week + 1):
        col_iqr = f'Week{w}_IQR'
        col_rel = f'Week{w}_Relative_Uncertainty'
        
        if col_iqr in row.columns and col_rel in row.columns:
            val_iqr = row[col_iqr].values[0]
            val_rel = row[col_rel].values[0]
            
            # Check for NaN
            if pd.notna(val_iqr) and pd.notna(val_rel):
                weeks.append(w)
                iqr.append(val_iqr)
                relative_uncertainty.append(val_rel)
            else:
                pass
                
    # 4. Plotting
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # X-axis
    x = np.array(weeks)
    
    # Smooth Curve Generation
    x_smooth = np.linspace(x.min(), x.max(), 300)
    
    # Spline for IQR
    spl_iqr = make_interp_spline(x, iqr, k=3)
    y_iqr_smooth = spl_iqr(x_smooth)
    
    # Spline for Relative Uncertainty
    spl_rel = make_interp_spline(x, relative_uncertainty, k=3)
    y_rel_smooth = spl_rel(x_smooth)
    
    # Plot Metric 1 on Left Axis (IQR)
    line1 = ax1.plot(x_smooth, y_iqr_smooth, linewidth=3, 
             color=color_iqr, label='IQR (Statistical Certainty)')
    # Also plot original points for clarity
    ax1.scatter(x, iqr, color=color_iqr, s=50, alpha=0.7)
             
    ax1.set_xlabel('Week', fontsize=12)
    ax1.set_ylabel('IQR Value (Probability)', fontsize=12)
    ax1.tick_params(axis='y')
    ax1.set_xticks(weeks)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Plot Metric 2 on Right Axis (Relative Uncertainty)
    ax2 = ax1.twinx()
    # Line 2: Relative Uncertainty
    line2 = ax2.plot(x_smooth, y_rel_smooth, linewidth=3, linestyle='-',
             color=color_rel, label='Relative Uncertainty (Score)')
    # Also plot original points for clarity
    ax2.scatter(x, relative_uncertainty, color=color_rel, s=50, marker='^', alpha=0.7)
             
    ax2.set_ylabel('Relative Score (Unitless)', fontsize=12)
    ax2.tick_params(axis='y')
    
    # Title
    plt.title(f'Certainty Trend for {target_celebrity} (Season {target_season})', fontsize=16)
    
    # Legend
    # Combine legends from both axes
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1),
              fancybox=True, shadow=True, ncol=2)
    
    plt.tight_layout()
    
    # Adjust layout to make room for legend
    plt.subplots_adjust(bottom=0.15)
    
    plt.savefig(output_file, dpi=300)
    print(f"Successfully saved plot to {output_file}")

if __name__ == "__main__":
    main()
