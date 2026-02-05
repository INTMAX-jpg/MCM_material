import pandas as pd
import numpy as np
import os
import re

# ==========================================
# Configuration
# ==========================================
input_file = 'cur代码/百分比-区间估计.csv'
original_file = 'cur代码/百分比制-每周平均占比&排名.csv'
output_file = 'cur代码/反向求百分比_置信区间.csv'
NUM_SIMULATIONS = 1000  # 模拟次数

def main():
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # Read data
    df = pd.read_csv(input_file)
    
    # Read original data to get judge percentages
    if os.path.exists(original_file):
        df_orig = pd.read_csv(original_file)
        # Merge judge percentage columns
        judge_cols = [c for c in df_orig.columns if 'judge_percentages' in c]
        merge_keys = ['celebrity', 'season']
        if all(k in df.columns for k in merge_keys) and all(k in df_orig.columns for k in merge_keys):
            df = pd.merge(df, df_orig[merge_keys + judge_cols], on=merge_keys, how='left')
            print("Merged judge percentages from original file.")
        else:
            print("Warning: Could not merge judge percentages. Keys missing.")
    else:
        print(f"Warning: {original_file} not found. Judge percentages may be missing.")
        return
    
    # 识别所有周数列
    week_cols = [c for c in df.columns if re.match(r'week_\d+_point_estimate', c)]
    weeks = sorted([int(re.match(r'week_(\d+)_point_estimate', c).group(1)) for c in week_cols])
    
    if not weeks:
        print("Error: No week data found.")
        return
        
    print(f"Found weeks: {weeks}")
    print(f"Running {NUM_SIMULATIONS} simulations per season...")
    
    # 初始化结果存储
    # 我们需要存储每一周的 mean, 25%, 75%
    # 结构：week_i_reverse_mean, week_i_reverse_25, week_i_reverse_75
    
    results_map = {} # (celebrity, season) -> {col_name: value}
    
    seasons = df['season'].unique()
    
    for season in seasons:
        season_mask = df['season'] == season
        season_df = df[season_mask].copy()
        
        # 获取有效周
        valid_weeks = []
        for w in weeks:
            col = f'week_{w}_point_estimate'
            if col in season_df.columns and season_df[col].notna().any():
                valid_weeks.append(w)
        
        if not valid_weeks:
            continue
            
        valid_weeks.sort()
        last_week = valid_weeks[-1]
        
        # 准备数据结构存储模拟结果
        # sim_results[week][celebrity_index] = array of N samples
        sim_results = {w: {} for w in valid_weeks}
        
        # 获取该赛季所有选手的索引
        season_indices = season_df.index.tolist()
        
        # 1. 初始化最后一周的样本
        # 对每个选手，从其 min/max/point 区间生成分布
        for idx in season_indices:
            row = season_df.loc[idx]
            w = last_week
            col_min = f'week_{w}_min_fan_percentage'
            col_max = f'week_{w}_max_fan_percentage'
            col_pt = f'week_{w}_point_estimate'
            
            if pd.isna(row[col_pt]):
                continue
                
            low = row[col_min] if pd.notna(row[col_min]) else row[col_pt]
            high = row[col_max] if pd.notna(row[col_max]) else row[col_pt]
            mode = row[col_pt]
            
            # 确保 low <= mode <= high
            low = min(low, mode)
            high = max(high, mode)
            
            # 如果区间太小，直接视为常数
            if high - low < 1e-6:
                samples = np.full(NUM_SIMULATIONS, mode)
            else:
                # 使用三角分布采样
                samples = np.random.triangular(left=low, mode=mode, right=high, size=NUM_SIMULATIONS)
            
            sim_results[last_week][idx] = samples
            
        # 2. 逆向推导
        for i in range(len(valid_weeks) - 2, -1, -1):
            current_week = valid_weeks[i]
            next_week = valid_weeks[i+1]
            
            # 识别本周被淘汰者
            # 淘汰者 = current_week 有 min/max, next_week 无
            col_lp_curr = f'week_{current_week}_point_estimate'
            col_lp_next = f'week_{next_week}_point_estimate'
            
            # 幸存者索引
            survivor_indices = [idx for idx in season_indices if idx in sim_results[next_week]]
            
            # 淘汰者索引
            eliminated_indices = []
            for idx in season_indices:
                if pd.notna(season_df.loc[idx, col_lp_curr]) and idx not in survivor_indices:
                    eliminated_indices.append(idx)
            
            # 生成淘汰者(k)的样本
            # 注意：如果有多个淘汰者，k是他们的和
            k_samples = np.zeros(NUM_SIMULATIONS)
            
            for idx in eliminated_indices:
                row = season_df.loc[idx]
                col_min = f'week_{current_week}_min_fan_percentage'
                col_max = f'week_{current_week}_max_fan_percentage'
                col_pt = f'week_{current_week}_point_estimate'
                
                low = row[col_min] if pd.notna(row[col_min]) else row[col_pt]
                high = row[col_max] if pd.notna(row[col_max]) else row[col_pt]
                mode = row[col_pt]
                
                low = min(low, mode)
                high = max(high, mode)
                
                if high - low < 1e-6:
                    s = np.full(NUM_SIMULATIONS, mode)
                else:
                    s = np.random.triangular(left=low, mode=mode, right=high, size=NUM_SIMULATIONS)
                
                # 记录被淘汰者本周的样本 (虽然不参与反推计算，但也是本周的结果)
                sim_results[current_week][idx] = s
                k_samples += s
                
            # 计算幸存者总份额 Sum x'
            sum_x_prime = np.zeros(NUM_SIMULATIONS)
            for idx in survivor_indices:
                sum_x_prime += sim_results[next_week][idx]
            
            # 避免除零
            sum_x_prime = np.maximum(sum_x_prime, 1e-9)
            
            # 计算幸存者本周的 x_i
            col_judge = f'week_{current_week}_judge_percentages'
            
            for idx in survivor_indices:
                x_prime = sim_results[next_week][idx]
                judge_pct_val = season_df.loc[idx, col_judge]
                
                # 如果评委分缺失，设为0
                if pd.isna(judge_pct_val): judge_pct_val = 0
                
                # 反向公式: x = x' - k * (0.3 * J + 0.7 * (x' / sum_x'))
                term_judge = 0.3 * judge_pct_val
                term_inherit = 0.7 * (x_prime / sum_x_prime)
                
                x_curr = x_prime - k_samples * (term_judge + term_inherit)
                
                # 存储结果
                sim_results[current_week][idx] = x_curr
                
        # 3. 统计结果并保存到 results_map
        for w in valid_weeks:
            for idx, samples in sim_results[w].items():
                mean_val = np.mean(samples)
                p25_val = np.percentile(samples, 25)
                p75_val = np.percentile(samples, 75)
                
                key = (season_df.loc[idx, 'celebrity'], season)
                if key not in results_map:
                    results_map[key] = {}
                    
                results_map[key][f'第{w}周_反向均值'] = mean_val
                results_map[key][f'第{w}周_反向25%CI'] = p25_val
                results_map[key][f'第{w}周_反向75%CI'] = p75_val
                
        print(f"Processed Season {season}")
        
    # 转换为 DataFrame
    # 先保留原始列
    base_cols = [
        'celebrity', 'ballroom_partner', 'celebrity_industry', 'celebrity_homestate', 
        'celebrity_homecountry/region', 'celebrity_age_during_season', 'season', 'placement'
    ]
    existing_base_cols = [c for c in base_cols if c in df.columns]
    df_out = df[existing_base_cols].copy()
    
    # 添加新列
    # 先收集所有可能的列名
    new_cols = []
    for w in weeks:
        new_cols.extend([f'第{w}周_反向均值', f'第{w}周_反向25%CI', f'第{w}周_反向75%CI'])
        
    for col in new_cols:
        df_out[col] = np.nan
        
    # 填充数据
    for idx, row in df_out.iterrows():
        key = (row['celebrity'], row['season'])
        if key in results_map:
            for col, val in results_map[key].items():
                df_out.at[idx, col] = val
                
    df_out.to_csv(output_file, index=False)
    print(f"Saved Monte Carlo results to {output_file}")

if __name__ == "__main__":
    main()
