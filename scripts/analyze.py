import _paths  # noqa: F401  (repository root on path, run from root)
import json, numpy as np, pickle
from scipy import stats
R = [json.loads(l) for l in open('results/results.jsonl')]
FDS = ['FD001', 'FD002', 'FD003', 'FD004']
METH = ['FOUCAL', 'FOUCAL-noMondrian', 'IT2-raw', 'IT2-split', 'IT2-MSE', 'T1-TSK', 'Ens-MLP', 'HGB-CQR', 'LSTM-CQR']
S = {}
for r in R: S.setdefault(r['data'], {}).setdefault(r['method'], []).append(r)
for fd in S:
    for m in S[fd]: S[fd][m].sort(key=lambda r: r['seed'])
def arr(fd, m, k): return np.array([r[k] for r in S[fd][m] if k in r], float)
def ms(a, p): return f"{a.mean():.{p}f} ± {a.std(ddof=1):.{p}f}" if len(a) > 1 else f"{a.mean():.{p}f}"
def pval(fd, m, k):
    a, b = arr(fd, 'FOUCAL', k), arr(fd, m, k); n = min(len(a), len(b))
    return float(stats.ttest_rel(a[:n], b[:n]).pvalue) if n >= 2 else float('nan')
summary = {fd: {m: {k: arr(fd, m, k).tolist() for k in S[fd][m][0] if isinstance(S[fd][m][0][k], (int, float))} for m in S[fd]} for fd in S}
json.dump(summary, open('results/summary.json', 'w'), indent=1)
if __name__ == '__main__':
    for fd in FDS:
        if fd not in S: continue
        print('==', fd)
        for m in METH:
            if m not in S[fd]: continue
            print(f"  {m:18s} n={len(S[fd][m])} RMSE {ms(arr(fd,m,'RMSE'),2)} cov {ms(arr(fd,m,'cov'),3)} vcov {ms(arr(fd,m,'val_cov'),3)} W {ms(arr(fd,m,'width'),1)} Wink {ms(arr(fd,m,'winkler'),1)} vWink {ms(arr(fd,m,'val_winkler'),1)} wb {ms(arr(fd,m,'worst_band_cov'),3)} vwb {ms(arr(fd,m,'val_worst_band_cov'),3)} rho {ms(arr(fd,m,'rho'),2)}" + (f" wr {ms(arr(fd,m,'worst_regime_cov'),3)} vwr {ms(arr(fd,m,'val_worst_regime_cov'),3)}" if 'worst_regime_cov' in S[fd][m][0] else ''))
