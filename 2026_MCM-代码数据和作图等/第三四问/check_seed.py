
import numpy as np

# Params: s=3.4, p=2.6, alpha=0.7
s = 3.4
p = 2.6
alpha = 0.7
n_contestants = 10
n_meta_sims = 2000

for seed in range(40, 60):
    np.random.seed(seed)
    
    # We need to match the logic in optimize_switch_week EXACTLY
    # Note: optimize_switch_week has some calls BEFORE the loop that consume random state?
    # No, line 187 sets seed.
    # But wait, line 195/196 call random.normal ONCE before the loop.
    # Then the loop runs n_meta_sims times.
    
    # Let's check the code in optimize_switch_week.py
    # Lines 195-196:
    # true_skills = np.random.normal(...)
    # partner_bonuses = np.random.normal(...)
    # ...
    # Then loop for n_meta_sims
    
    # Actually, the logic in the loop RE-GENERATES t_skills and p_bonuses.
    # The code outside the loop (lines 195-203) is for "One Instance" demo?
    # But the counts `total_fair_fixed` are accumulated inside the loop.
    # So the initial call outside doesn't affect the loop stats, but it consumes random numbers.
    
    # Let's replicate the sequence of random calls.
    
    # Outside loop calls
    _ = np.random.normal(7, s, n_contestants)
    _ = np.random.normal(0, p, n_contestants)
    
    total_fair_fixed = 0
    total_fair_random = 0
    
    for _ in range(n_meta_sims):
        t_skills = np.random.normal(7, s, n_contestants)
        p_bonuses = np.random.normal(0, p, n_contestants)
        best_idx = np.argmax(t_skills)
        
        # Fixed
        f_scores = t_skills + p_bonuses
        if np.argmax(f_scores) == best_idx:
            total_fair_fixed += 1
            
        # Random
        s_partners = np.random.permutation(p_bonuses)
        r_scores = t_skills + 0.35 * p_bonuses + 0.35 * s_partners
        if np.argmax(r_scores) == best_idx:
            total_fair_random += 1
            
    rate_f = total_fair_fixed / n_meta_sims
    rate_r = total_fair_random / n_meta_sims
    
    print(f"Seed {seed}: Fixed={rate_f:.3f}, Random={rate_r:.3f}")
