import math
import numpy as np

def hac_mean_test(values: np.ndarray, lag: int = 10) -> dict[str, float | int]:
    values = values[np.isfinite(values)]
    n = len(values)
    mean = float(values.mean())
    centered = values - mean
    gamma0 = float(np.dot(centered, centered) / n)
    long_run = gamma0
    used_lag = min(lag, n - 1)
    for offset in range(1, used_lag + 1):
        covariance = float(np.dot(centered[offset:], centered[:-offset]) / n)
        weight = 1.0 - offset / (used_lag + 1.0)
        long_run += 2.0 * weight * covariance
    long_run = max(long_run, 0.0)
    se = math.sqrt(long_run / n) if n else math.nan
    z = mean / se if se > 0 else math.nan
    p = normal_two_sided_p(z) if math.isfinite(z) else math.nan
    return {
        "days": n,
        "mean_difference": mean,
        "hac_lag": used_lag,
        "hac_se": se,
        "z": z,
        "p_value": p,
    }

def normal_two_sided_p(z):
    return math.erfc(abs(z)/math.sqrt(2))

def moving_block_interval(values, block=20, replicates=5000, seed=20261001):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or not np.isfinite(x).all() or len(x)<block or block<1:
        raise ValueError("Expected finite daily values and 1 <= block <= n")
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,len(x)-block+1,size=(replicates,int(np.ceil(len(x)/block))))
    idx=(starts[:,:,None]+np.arange(block)).reshape(replicates,-1)[:,:len(x)]
    return np.quantile(x[idx].mean(1),[.025,.975]).tolist()
