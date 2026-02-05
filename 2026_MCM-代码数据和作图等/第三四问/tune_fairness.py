
import numpy as np

print("Searching with alpha...")
best_err = 1.0
best_params = (0,0,0)

# Grid search
for s in np.arange(2.0, 4.0, 0.2):
    for p in np.arange(2.0, 4.0, 0.2):
        for alpha in [0.7, 0.8, 0.9, 1.0]:
            w = 0.5
            
            tf = 0
            tr = 0
            N = 2000
            
            skills = np.random.normal(0, s, (N, 10))
            p1 = np.random.normal(0, p, (N, 10))
            
            best_idx = np.argmax(skills, axis=1)
            
            # Fixed: S + P1
            fixed_scores = skills + p1
            fixed_winners = np.argmax(fixed_scores, axis=1)
            rate_f = np.mean(fixed_winners == best_idx)
            
            # Random: S + alpha * (0.5*P1 + 0.5*P2)
            p2_perm = np.zeros_like(p1)
            for i in range(N):
                p2_perm[i] = np.random.permutation(p1[i])
                
            rand_scores = skills + alpha * (w*p1 + (1-w)*p2_perm)
            rand_winners = np.argmax(rand_scores, axis=1)
            rate_r = np.mean(rand_winners == best_idx)
            
            err = abs(rate_f - 0.53) + abs(rate_r - 0.72)
            
            if err < best_err:
                best_err = err
                best_params = (s, p, alpha)
                print(f"New best: s={s:.1f}, p={p:.1f}, a={alpha:.1f} -> Fixed={rate_f:.3f}, Random={rate_r:.3f}, Err={err:.3f}")
