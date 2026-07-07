#!/usr/bin/env python3
"""
Confidence-interval and significance computations for:
"Real-Time Automated Estrus Behavior Detection..." (IEEE Access submission)

Run:  python3 compute_cis.py

Every input below is a COUNT taken directly from the paper's evaluation.
Confirm each count matches your real results before reporting the output.
Method matches the Methods section:
  - proportions (recall/accuracy): nonparametric bootstrap, 10,000 iters, percentile CI
  - precision/recall/F1: JOINT bootstrap over the pooled {TP, FP, FN} instances
  - FP rates: exact Poisson interval on the count, divided by hours
  - single- vs multi-camera recall: exact McNemar test
"""

import numpy as np
from scipy.stats import chi2, binomtest

SEED = 42
N_BOOT = 10_000
rng = np.random.default_rng(SEED)

def boot_proportion(successes, n, label):
    """Bootstrap CI for a simple proportion successes/n."""
    data = np.array([1]*successes + [0]*(n-successes))
    est = data.mean()
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        boots[i] = rng.choice(data, size=n, replace=True).mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"{label:32s}: {est*100:5.1f}%  (95% CI {lo*100:4.1f}-{hi*100:5.1f}%)   [{successes}/{n}]")
    return est, lo, hi

def boot_prf(tp, fp, fn, label):
    """JOINT bootstrap of precision, recall, F1 over pooled TP/FP/FN instances."""
    pool = np.array([0]*tp + [1]*fp + [2]*fn)   # 0=TP, 1=FP, 2=FN
    def prf(arr):
        t = np.sum(arr==0); f = np.sum(arr==1); m = np.sum(arr==2)
        P = t/(t+f) if (t+f)>0 else np.nan
        R = t/(t+m) if (t+m)>0 else np.nan
        F = 2*P*R/(P+R) if (P and R and not np.isnan(P) and not np.isnan(R) and (P+R)>0) else np.nan
        return P, R, F
    P0,R0,F0 = prf(pool)
    Ps=np.empty(N_BOOT); Rs=np.empty(N_BOOT); Fs=np.empty(N_BOOT)
    for i in range(N_BOOT):
        s = rng.choice(pool, size=len(pool), replace=True)
        Ps[i],Rs[i],Fs[i] = prf(s)
    def ci(a): return np.nanpercentile(a,[2.5,97.5])
    pl,ph = ci(Ps); rl,rh = ci(Rs); fl,fh = ci(Fs)
    print(f"\n{label}  (TP={tp}, FP={fp}, FN={fn})")
    print(f"   Precision: {P0:.3f}  (95% CI {pl:.2f}-{ph:.2f})")
    print(f"   Recall   : {R0:.3f}  (95% CI {rl:.2f}-{rh:.2f})")
    print(f"   F1       : {F0:.3f}  (95% CI {fl:.2f}-{fh:.2f})")

def poisson_rate_ci(k, hours, label):
    """Exact Poisson 95% CI on a count k, expressed as a rate per hour."""
    lo_k = chi2.ppf(0.025, 2*k)/2 if k>0 else 0.0
    hi_k = chi2.ppf(0.975, 2*k+2)/2
    print(f"{label:32s}: {k/hours:4.2f} FP/h  (95% CI {lo_k/hours:4.2f}-{hi_k/hours:4.2f})   [{k} in {hours} h]")

print("="*70)
print(f"SEED = {SEED} | {N_BOOT:,} bootstrap iterations")
print("="*70)
print("\n--- Bootstrap CIs: proportions ---")
boot_proportion(18, 22, "Event-level recall")
boot_proportion(11, 22, "Per-camera recall")
boot_proportion(11, 12, "Mounter ID accuracy")
boot_proportion(7, 12, "Mountee ID accuracy")
boot_proportion(18, 24, "Overall ID accuracy")

print("\n--- Bootstrap CIs: precision / recall / F1 (joint) ---")
# Confirmed only: 12 TP, 5 FP; recall denom = 18 GT events -> FN = 18-12 = 6
boot_prf(12, 5, 6, "Confirmed only")
# Confirmed + possible: 15 TP, 9 FP; recall denom = 18 -> FN = 3
boot_prf(15, 9, 3, "Confirmed + possible")

print("\n--- Exact Poisson CIs: false-positive rates ---")
poisson_rate_ci(1, 10.5, "Confirmed-tier FP rate")
poisson_rate_ci(9, 10.5, "Overall FP rate")

print("\n--- Exact McNemar test: single-camera vs fused recall ---")
b, c = 7, 0
p = binomtest(min(b,c), n=b+c, p=0.5).pvalue
print(f"b={b}, c={c}  ->  exact two-sided p = {p:.4f}")
print("="*70)
