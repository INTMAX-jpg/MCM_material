import pandas as pd
import numpy as np
import os
import re

# ==========================================
# Configuration
# ==========================================
input_file = 'cur代码/百分比-区间估计.csv'
original_file = 'cur代码/百分比制-每周平均占比&排名.csv'
output_file = 'cur代码/反向求百分比结果.csv'

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
        # Merge based on key columns (assuming celebrity and season are unique keys)
        # Use left merge to keep df rows
        merge_keys = ['celebrity', 'season']
        # Check if keys exist
        if all(k in df.columns for k in merge_keys) and all(k in df_orig.columns for k in merge_keys):
            df = pd.merge(df, df_orig[merge_keys + judge_cols], on=merge_keys, how='left')
            print("Merged judge percentages from original file.")
        else:
            print("Warning: Could not merge judge percentages. Keys missing.")
    else:
        print(f"Warning: {original_file} not found. Judge percentages may be missing.")
    
    # 识别所有周数列
    week_cols = [c for c in df.columns if re.match(r'week_\d+_point_estimate', c)]
    weeks = sorted([int(re.match(r'week_(\d+)_point_estimate', c).group(1)) for c in week_cols])
    
    if not weeks:
        print("Error: No week data found.")
        return
        
    max_week = max(weeks)
    print(f"Found weeks: {weeks}")
    
    # 初始化结果列
    for w in weeks:
        df[f'第{w}周的反向预测百分比xi'] = np.nan
        
    # 按 Season 分组处理
    seasons = df['season'].unique()
    
    for season in seasons:
        season_mask = df['season'] == season
        season_df = df[season_mask].copy()
        
        # 获取该 Season 包含的有效周数 (有些 Season 可能没有 Week 10, 11)
        # 检查该 Season 哪一周有非空数据
        valid_weeks = []
        for w in weeks:
            col = f'week_{w}_point_estimate'
            if col in season_df.columns and season_df[col].notna().any():
                valid_weeks.append(w)
        
        if not valid_weeks:
            continue
            
        valid_weeks.sort()
        last_week = valid_weeks[-1]
        
        # 1. 最后一周：直接取 LP 结果
        # 题目要求："最后一周的所有选手 x' 值：直接从此前线性规划的计算结果中获取"
        # 这里 x' 即为该周的 xi (对于最后一周而言，没有后续反推，直接视为最终状态)
        # 或者理解为：最后一周的“反向预测值”就是其本身
        col_lp_last = f'week_{last_week}_point_estimate'
        col_res_last = f'第{last_week}周的反向预测百分比xi'
        
        # 只对非空值赋值
        mask_last = season_df[col_lp_last].notna()
        season_df.loc[mask_last, col_res_last] = season_df.loc[mask_last, col_lp_last]
        
        # 2. 逆向推导 (从 last_week - 1 到 1)
        # 我们需要从 t+1 周推导 t 周
        # x_i_t 是未知数
        # x_i_t+1 (即公式中的 x') 是已知数 (来自上一轮反推结果，或者如果是 last_week 则来自 LP)
        
        for i in range(len(valid_weeks) - 2, -1, -1):
            current_week = valid_weeks[i]
            next_week = valid_weeks[i+1]
            
            col_res_curr = f'第{current_week}周的反向预测百分比xi'
            col_res_next = f'第{next_week}周的反向预测百分比xi' # 这是 x'
            
            # 获取本周(current_week)的 k 值：被淘汰选手的粉丝比例
            # 被淘汰选手 = 本周有数据，但下周没数据 (或者 placement 逻辑)
            # 题目明确：k 从此前线性规划的结果中提取对应周的数值
            # 我们查找在 current_week 有效，但在 next_week 无效的选手
            # 注意：LP 结果中，被淘汰者在淘汰周是有值的
            
            col_lp_curr = f'week_{current_week}_point_estimate'
            col_lp_next = f'week_{next_week}_point_estimate'
            col_judge_curr = f'week_{current_week}_judge_percentages'
            
            # 找出幸存者 (在 next_week 也有反向预测值的人)
            # 注意：x' 来自 col_res_next。如果某人在 next_week 已经被淘汰了，他不会有 col_res_next
            # 所以幸存者就是 col_res_next 非空的人
            survivors_mask = season_df[col_res_next].notna()
            
            # 找出本周被淘汰者
            # 定义：在 current_week LP 有值，但在 next_week LP 没值 (NaN)
            eliminated_mask = season_df[col_lp_curr].notna() & season_df[col_lp_next].isna()
            
            k_values = season_df.loc[eliminated_mask, col_lp_curr]
            k = k_values.sum() # 如果有多个被淘汰，累加他们的份额 (假设遗产是共享的)
            
            if k_values.empty and not survivors_mask.any():
                # 异常情况
                continue
                
            # 计算幸存者的 x_i (当前周)
            # 公式: x_i = x' * (1 - 0.7*k) - 0.3 * k * Judge_pct
            # 其中 x' 是 col_res_next
            # Judge_pct 是 col_judge_curr
            
            # Sum x_j' (下一周幸存者总和)
            # sum_x_prime = season_df.loc[survivors_mask, col_res_next].sum()
            # 题目公式: 0.7 * (x_i' / sum_x_prime)
            # 如果下一周已经是归一化的，sum_x_prime 应该接近 1
            # 但为了严谨，我们计算实际的 sum
            
            sum_x_prime = season_df.loc[survivors_mask, col_res_next].sum()
            
            if sum_x_prime == 0:
                 # 避免除以零
                 sum_x_prime = 1.0
            
            # 计算幸存者的值
            # x_i = x_prime - k * (0.3 * J + 0.7 * (x_prime / sum_x_prime))
            # 移项推导:
            # 原公式: x' = x + k * (0.3 * J + 0.7 * (x' / sum_x'))
            # x = x' - k * (0.3 * J + 0.7 * (x' / sum_x'))
            
            # 批量计算
            x_prime = season_df.loc[survivors_mask, col_res_next]
            judge_pct = season_df.loc[survivors_mask, col_judge_curr]
            
            # 确保 judge_pct 是数值
            if judge_pct.dtype == object:
                 judge_pct = pd.to_numeric(judge_pct, errors='coerce').fillna(0)
            
            # 计算项
            term_judge = 0.3 * judge_pct
            term_inherit = 0.7 * (x_prime / sum_x_prime)
            
            x_curr = x_prime - k * (term_judge + term_inherit)
            
            # 赋值给幸存者
            season_df.loc[survivors_mask, col_res_curr] = x_curr
            
            # 对于本周被淘汰者，他们的 x_i 就是他们的 LP 值 (因为他们没有下一周来反推)
            # 题目说 "计算排除该周被淘汰选手后，其余所有未淘汰选手的... x_i"
            # 但为了数据完整性，我们把被淘汰者的值也填上 (即 k 值)，或者留空？
            # 题目输出要求："第i周的反向预测百分比xi"
            # 如果只填幸存者，被淘汰者那一栏就是 NaN。这符合 "排除...后...其余..." 的描述。
            # 但被淘汰者在这一周确实有粉丝比例。
            # 既然 k 是已知的，我们不妨填入 k，表示他在这一周的真实比例。
            season_df.loc[eliminated_mask, col_res_curr] = season_df.loc[eliminated_mask, col_lp_curr]

        # 将计算好的 season_df 更新回主 df
        df.update(season_df)
        print(f"Processed Season {season}")

    # 保存结果
    # 按照要求的列顺序
    base_cols = [
        'celebrity', 'ballroom_partner', 'celebrity_industry', 'celebrity_homestate', 
        'celebrity_homecountry/region', 'celebrity_age_during_season', 'season', 'placement'
    ]
    
    # 确保这些列存在
    existing_base_cols = [c for c in base_cols if c in df.columns]
    
    result_cols = existing_base_cols + [f'第{w}周的反向预测百分比xi' for w in weeks]
    
    df_out = df[result_cols]
    df_out.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")

if __name__ == "__main__":
    main()
