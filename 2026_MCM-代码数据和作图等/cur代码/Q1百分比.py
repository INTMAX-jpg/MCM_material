import pandas as pd
import numpy as np
from scipy.optimize import linprog, minimize
import os

# ==========================================
# Configuration
# ==========================================
input_file = 'cur代码/百分比制-每周平均占比&排名.csv'
output_file = 'cur代码/百分比-区间估计.csv'
epsilon = 1e-5

# ==========================================
# Helper Functions
# ==========================================

def solve_fan_votes(judges_pct, ranks):
    """
    Solve for fan votes intervals and point estimate for a single week/group.
    
    Args:
        judges_pct (list): List of judge percentages for each contestant.
        ranks (list): List of ranks for each contestant.
        
    Returns:
        list of dicts: Each dict contains min, max, delta, point_est for a contestant.
    """
    n = len(judges_pct)
    if n == 0:
        return []

    # Identify indices
    indices = list(range(n))
    
    # 1. Linear Programming Setup
    # Variables: x_0, x_1, ..., x_{n-1} (fan percentages)
    # Constraints:
    #   sum(x) = 1 (Equality)
    #   x_i >= 0 (Bounds)
    #   If rank[i] < rank[j] (i is better), then Total[i] >= Total[j] + eps
    #   (J[i] + x[i]) >= (J[j] + x[j]) + eps
    #   x[j] - x[i] <= J[i] - J[j] - eps
    
    A_ub = []
    b_ub = []
    
    # Generate pairwise constraints based on Rank
    # To reduce complexity, we can group by rank and compare adjacent rank groups.
    # Group 1 (Rank 1) vs Group 2 (Rank 2)
    # All members of G1 must beat all members of G2.
    
    # Create rank groups
    rank_map = {}
    for idx, r in enumerate(ranks):
        if pd.isna(r): continue
        if r not in rank_map: rank_map[r] = []
        rank_map[r].append(idx)
        
    sorted_ranks = sorted(rank_map.keys())
    
    for k in range(len(sorted_ranks) - 1):
        r_better = sorted_ranks[k]
        r_worse = sorted_ranks[k+1]
        
        group_better = rank_map[r_better]
        group_worse = rank_map[r_worse]
        
        for idx_b in group_better:
            for idx_w in group_worse:
                # Constraint: x[idx_w] - x[idx_b] <= J[idx_b] - J[idx_w] - epsilon
                row = [0] * n
                row[idx_w] = 1
                row[idx_b] = -1
                A_ub.append(row)
                b_ub.append(judges_pct[idx_b] - judges_pct[idx_w] - epsilon)

    # Equality constraint: sum(x) = 1
    A_eq = [[1] * n]
    b_eq = [1]
    
    bounds = [(0, 1) for _ in range(n)]
    
    # Handle empty inequality constraints
    if not A_ub:
        A_ub_arg = None
        b_ub_arg = None
    else:
        A_ub_arg = A_ub
        b_ub_arg = b_ub
    
    results = []
    
    # Store feasible range for MaxEntropy initial guess
    feasible_bounds = []

    # Solve Min/Max for each contestant
    for i in range(n):
        # Min
        c_min = [0] * n
        c_min[i] = 1
        res_min = linprog(c_min, A_ub=A_ub_arg, b_ub=b_ub_arg, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        # Max
        c_max = [0] * n
        c_max[i] = -1
        res_max = linprog(c_max, A_ub=A_ub_arg, b_ub=b_ub_arg, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if res_min.success and res_max.success:
            val_min = max(0, res_min.fun)
            val_max = min(1, -res_max.fun)
            # Ensure min <= max (floating point errors)
            if val_min > val_max: val_min, val_max = val_max, val_min
            
            results.append({
                'min': val_min,
                'max': val_max,
                'delta': val_max - val_min
            })
            feasible_bounds.append((val_min, val_max))
        else:
            # If infeasible, return NaNs or fallback
            # This implies the judge ranking cannot be preserved even with extreme fan votes?
            # Or just solver issue.
            results.append({'min': np.nan, 'max': np.nan, 'delta': np.nan})
            feasible_bounds.append((0, 1))

    # 3. Point Estimate (Max Entropy)
    # Maximize H(x) = - sum(x * log(x))
    # Equivalent to Minimize sum(x * log(x))
    
    # Check if we have any valid results to run optimization
    if any(np.isnan(r['min']) for r in results):
         for r in results: r['point_est'] = np.nan
         return results

    def neg_entropy(x):
        # Add small constant to avoid log(0)
        x_safe = np.maximum(x, 1e-10)
        return np.sum(x_safe * np.log(x_safe))
    
    # Constraints for minimize
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}] # Sum = 1
    
    # Add inequality constraints: A_ub @ x <= b_ub
    # minimize expects constraints as c(x) >= 0. So b - Ax >= 0
    if A_ub:
        # We can pass linear constraints directly if using 'SLSQP' or 'trust-constr'
        # But let's wrap them
        cons.append({'type': 'ineq', 'fun': lambda x: np.array(b_ub) - np.dot(A_ub, x)})
        
    # Initial guess: Center of feasible bounds or uniform
    x0 = np.array([(b[0] + b[1])/2 for b in feasible_bounds])
    # Normalize x0 to sum to 1 just in case
    if np.sum(x0) > 0:
        x0 = x0 / np.sum(x0)
    else:
        x0 = np.ones(n) / n
        
    # Bounds for minimize
    # Use the tightened bounds calculated from LP? Yes, it helps convergence.
    # But strictly, 0-1 is fine as constraints handle it.
    # Using tightened bounds is safer.
    opt_bounds = feasible_bounds
    
    try:
        opt_res = minimize(neg_entropy, x0, bounds=opt_bounds, constraints=cons, method='SLSQP', tol=1e-6)
        
        if opt_res.success:
            points = opt_res.x
        else:
            # Fallback if optimization fails
            # print(f"MaxEntropy failed: {opt_res.message}")
            points = x0
    except Exception as e:
        print(f"Optimization error: {e}")
        points = x0

    for idx, r in enumerate(results):
        r['point_est'] = points[idx]
        
    return results

# ==========================================
# Main Execution
# ==========================================

def main():
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    df = pd.read_csv(input_file)
    
    # Identify week columns
    # Pattern: week_X_judge_percentages, week_X_judge_rank
    import re
    cols = df.columns
    week_nums = set()
    for c in cols:
        m = re.match(r'week_(\d+)_judge_percentages', c)
        if m:
            week_nums.add(int(m.group(1)))
    
    sorted_weeks = sorted(list(week_nums))
    print(f"Found weeks: {sorted_weeks}")
    
    # Initialize new columns
    for w in sorted_weeks:
        df[f'week_{w}_min_fan_percentage'] = np.nan
        df[f'week_{w}_max_fan_percentage'] = np.nan
        df[f'week_{w}_delta'] = np.nan
        df[f'week_{w}_point_estimate'] = np.nan
    
    # Iterate by Season and Week
    # We must process each week's competition group separately.
    # Usually, for a given season and week, there is a set of contestants.
    
    seasons = df['season'].unique()
    
    for season in seasons:
        season_df = df[df['season'] == season]
        
        for w in sorted_weeks:
            # Check if this week exists for this season
            pct_col = f'week_{w}_judge_percentages'
            rank_col = f'week_{w}_judge_rank'
            
            if pct_col not in df.columns: continue
            
            # Get contestants who have valid data for this week
            # We filter by non-NaN in judge_percentages
            # Note: We work on the indices of the original df to update it later
            
            # Filter rows for this season and non-null week data
            mask = (df['season'] == season) & (df[pct_col].notna()) & (df[rank_col].notna())
            active_indices = df.index[mask].tolist()
            
            if not active_indices:
                continue
                
            judges_pct = df.loc[active_indices, pct_col].values
            ranks = df.loc[active_indices, rank_col].values
            
            # Solve
            results = solve_fan_votes(judges_pct, ranks)
            
            # Update DataFrame
            for i, idx in enumerate(active_indices):
                res = results[i]
                df.at[idx, f'week_{w}_min_fan_percentage'] = res['min']
                df.at[idx, f'week_{w}_max_fan_percentage'] = res['max']
                df.at[idx, f'week_{w}_delta'] = res['delta']
                df.at[idx, f'week_{w}_point_estimate'] = res['point_est']
                
        print(f"Processed Season {season}")

    # Reorder columns: Original + New
    # Identify original columns
    original_cols = [c for c in df.columns if 'min_fan_percentage' not in c and 'max_fan_percentage' not in c and 'delta' not in c and 'point_estimate' not in c]
    
    # Identify new columns in order
    new_cols = []
    for w in sorted_weeks:
        new_cols.extend([
            f'week_{w}_min_fan_percentage',
            f'week_{w}_max_fan_percentage',
            f'week_{w}_delta',
            f'week_{w}_point_estimate'
        ])
        
    final_cols = original_cols + new_cols
    # Ensure all exist
    final_cols = [c for c in final_cols if c in df.columns]
    
    df = df[final_cols]
    
    df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")

if __name__ == "__main__":
    main()
