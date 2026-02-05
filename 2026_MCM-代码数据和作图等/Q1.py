import pandas as pd
import numpy as np
from scipy.optimize import linprog
import itertools

def estimate_fan_votes(data_path):
    # 1. 加载数据
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        return "错误：未找到数据文件。请确保文件名正确。"

    # 预处理：计算每周评委总分（汇总 judge1 到 judge4 的分数）
    judge_cols = [c for c in df.columns if 'judge' in c and 'score' in c]
    df['total_judge_score'] = df[judge_cols].sum(axis=1)

    results_summary = []

    # 遍历每个赛季
    for season in df['season'].unique():
        s_data = df[df['season'] == season].copy()
        
        # A. 确定该赛季的合并规则
        # S1-2, S28-34: Rank; S3-27: Percent (根据题目假设)
        is_percent_rule = 3 <= season <= 27
        
        # B. 确定最后一周 (Final Week)
        # 寻找仍有选手得分 > 0 的最大周
        week_cols = [c for c in df.columns if 'week' in c and 'judge' in c]
        weeks = sorted(list(set([int(c.split('_')[0].replace('week', '')) for c in week_cols])))
        
        final_week = 1
        for w in reversed(weeks):
            w_score_cols = [c for c in judge_cols if f'week{w}' in c]
            if s_data[w_score_cols].sum().sum() > 0:
                final_week = w
                break
        
        # C. 提取最后一周的决赛选手 (Finalists)
        w_score_cols = [c for c in judge_cols if f'week{final_week}' in c]
        finalists = s_data[s_data[w_score_cols].sum(axis=1) > 0].copy()
        finalists = finalists.sort_values('placement') # 按最终名次排序
        
        k = len(finalists)
        if k == 0: continue
        
        # 提取评委总分
        finalists['current_judge_score'] = finalists[w_score_cols].sum(axis=1)
        judge_scores = finalists['current_judge_score'].values
        
        # --- 策略 1: 百分比合并制 (Linear Programming) ---
        if is_percent_rule:
            judge_pct = judge_scores / np.sum(judge_scores)
            fan_bounds = []
            
            # 对每个选手 i，求 fan_pct 的上下界
            for i in range(k):
                # 变量: x_0, x_1, ..., x_{k-1} 代表每个人的观众投票百分比
                # 约束 1: sum(x) = 1
                # 约束 2: x_i >= 0
                # 约束 3: (pj_m + x_m) >= (pj_{m+1} + x_{m+1}) + epsilon (名次约束)
                
                # linprog 目标函数 (求最小和最大)
                c_min = np.zeros(k); c_min[i] = 1
                
                # 等式约束: A_eq @ x = b_eq -> sum(x) = 1
                A_eq = [np.ones(k)]
                b_eq = [1]
                
                # 不等式约束: A_ub @ x <= b_ub -> (pj_m+1 - pj_m) + eps <= x_m - x_m+1
                # 即 x_{m+1} - x_m <= pj_m - pj_{m+1} - eps
                A_ub = []
                b_ub = []
                eps = 1e-5
                for m in range(k - 1):
                    row = np.zeros(k)
                    row[m] = -1
                    row[m+1] = 1
                    A_ub.append(row)
                    b_ub.append(judge_pct[m] - judge_pct[m+1] - eps)
                
                res_min = linprog(c_min, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1))
                res_max = linprog(-c_min, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1))
                
                f_min = res_min.x[i] if res_min.success else np.nan
                f_max = -res_max.fun if res_max.success else np.nan
                fan_bounds.append((f_min, f_max))

            # 点估计：最大熵 (简单实现：取区间中值并归一化)
            for i, name in enumerate(finalists['celebrity']):
                f_min, f_max = fan_bounds[i]
                results_summary.append({
                    'Season': season, 'Celebrity': name, 'Placement': i+1,
                    'Rule': 'Percent', 'Judge_Pct': f"{judge_pct[i]:.2%}",
                    'Est_Fan_Pct_Range': f"[{f_min:.2%}, {f_max:.2%}]",
                    'Uncertainty': f"{f_max - f_min:.4f}"
                })

        # --- 策略 2: 排名合并制 (Permutation Search) ---
        else:
            # 评委排名 (分数越高排名越小)
            # 处理相同分数：采用 min 排名
            j_ranks = finalists['current_judge_score'].rank(ascending=False, method='min').values
            
            possible_fan_ranks = [[] for _ in range(k)]
            valid_count = 0
            
            # 枚举所有可能的粉丝排名排列 [1, 2, ..., k]
            for f_rank_perm in itertools.permutations(range(1, k + 1)):
                # 计算总排名分: Judge Rank + Fan Rank
                combined_scores = j_ranks + np.array(f_rank_perm)
                
                # 检查产生的排名顺序是否与 placement [1, 2, ..., k] 一致
                # 注意：如果总分相同，通常看谁的粉丝分更高，这里简化为严格序
                temp_ranks = pd.Series(combined_scores).rank(method='min').values
                if np.array_equal(temp_ranks, np.arange(1, k + 1)):
                    valid_count += 1
                    for idx, r in enumerate(f_rank_perm):
                        possible_fan_ranks[idx].append(r)
            
            for i, name in enumerate(finalists['celebrity']):
                p_ranks = sorted(list(set(possible_fan_ranks[i])))
                results_summary.append({
                    'Season': season, 'Celebrity': name, 'Placement': i+1,
                    'Rule': 'Rank', 'Judge_Rank': int(j_ranks[i]),
                    'Est_Fan_Rank_Range': f"{p_ranks}",
                    'Uncertainty': f"Permutations: {valid_count}"
                })

    return pd.DataFrame(results_summary)

# 使用示例
results = estimate_fan_votes('C_origin.csv')
print(results.head(10))