import _paths  # noqa: F401  (repository root on path, run from root)
import json, numpy as np
from scipy import stats
from analyze import S, arr, ms, FDS
T, V = {}, {}
seeds = min(len(S[fd]['FOUCAL']) for fd in FDS)
LAB = {'FOUCAL': 'FOUCAL (proposed)', 'T1-TSK': 'T1-TSK', 'Ens-MLP': 'Ens-MLP', 'HGB-CQR': 'HGB-CQR', 'LSTM-CQR': 'LSTM-CQR',
       'IT2-raw': 'IT2-raw (no calibration)', 'IT2-split': 'IT2-split (unnormalised scores)', 'IT2-MSE': 'IT2-MSE (no likelihood term)', 'FOUCAL-noMondrian': 'FOUCAL-noMondrian (pooled quantile)'}
MAIN = ['FOUCAL', 'T1-TSK', 'Ens-MLP', 'HGB-CQR', 'LSTM-CQR']
T['T_RMSE'] = dict(head=['Method', 'FD001', 'FD002', 'FD003', 'FD004'], rows=[[LAB[m]] + [ms(arr(fd, m, 'RMSE'), 2) for fd in FDS] for m in MAIN])
rows = []
for fd in FDS:
    best = min(MAIN, key=lambda m: arr(fd, m, 'winkler').mean())
    for m in MAIN:
        f = lambda k, p: f"{arr(fd, m, k).mean():.{p}f}"
        isv = f('winkler', 1); isv = f"**{isv}**" if m == best else isv
        rows.append([fd if m == 'FOUCAL' else '', LAB[m], f('cov', 3), f('val_cov', 3), f('width', 1), isv, f('rho', 2)])
T['T_INT'] = dict(head=['Subset', 'Method', 'Cov', 'Val cov', 'Width', 'IS', 'ρ'], rows=rows)
ABL = ['FOUCAL', 'FOUCAL-noMondrian', 'IT2-MSE', 'IT2-split', 'IT2-raw']
T['T_ABL'] = dict(head=['Variant', 'FD001', 'FD002', 'FD003', 'FD004'], rows=[[LAB[m]] + [f"{arr(fd, m, 'winkler').mean():.1f} ({arr(fd, m, 'cov').mean():.2f})" for fd in FDS] for m in ABL])
st = []
for fd in FDS:
    for m in ['T1-TSK', 'Ens-MLP', 'HGB-CQR', 'LSTM-CQR']:
        a, b = arr(fd, 'FOUCAL', 'winkler'), arr(fd, m, 'winkler'); n = min(len(a), len(b)); p = stats.ttest_rel(a[:n], b[:n]).pvalue
        st.append([fd, m, f"{(a[:n] - b[:n]).mean():+.1f}", '< 0.001' if p < 0.001 else f"{p:.3f}"])
T['T_STATS'] = dict(head=['Subset', 'Baseline', 'Mean difference in IS', 'p value'], rows=st)
rl = json.load(open('results/rules.json'))['FD001']['rules']
T['T_RULES'] = dict(head=['IF (antecedents)', 'THEN RUL (cycles)', 'FOU ratio', 'Support'], rows=[[r['ante'], f"{r['rul']:.0f}", f"{r['fou']:.2f}", f"{r['support']:.3f}"] for r in rl])
# placeholders
for fd in FDS:
    for m in S[fd]:
        for k in S[fd][m][0]:
            if isinstance(S[fd][m][0][k], (int, float)): V[f'{fd}|{m}|{k}'] = float(arr(fd, m, k).mean())
rank = [sorted(MAIN, key=lambda m: arr(fd, m, 'winkler').mean()).index('FOUCAL') for fd in FDS]
V['nbest'] = sum(r <= 1 for r in rank)
V['nnarrow'] = sum(min(MAIN, key=lambda m: arr(fd, m, 'width').mean()) == 'FOUCAL' for fd in FDS)
red = np.mean([1 - arr(fd, 'FOUCAL', 'width').mean() / arr(fd, 'T1-TSK', 'width').mean() for fd in FDS])
V['wred'] = f"{100 * red:.0f}% on average"
for k, m in [('rho_foucal', 'FOUCAL'), ('rho_t1', 'T1-TSK'), ('rho_ens', 'Ens-MLP')]: V[k] = f"{np.mean([arr(fd, m, 'rho').mean() for fd in FDS]):.2f}"
c = [arr(fd, 'FOUCAL', 'cov').mean() for fd in FDS]; vc = [arr(fd, 'FOUCAL', 'val_cov').mean() for fd in FDS]
V.update(covmin=f"{min(c):.2f}", covmax=f"{max(c):.2f}", vcovmin=f"{min(vc):.2f}", vcovmax=f"{max(vc):.2f}")
rc = [arr(fd, 'IT2-raw', 'cov').mean() for fd in FDS]; V.update(raw_covmin=f"{min(rc):.2f}", raw_covmax=f"{max(rc):.2f}")
B = json.load(open('results/bands.json')); mi = B['methods']
V.update(b_cov0=f"{B['cov'][mi.index('FOUCAL')][0]:.2f}", b_wid0=f"{B['wid'][mi.index('FOUCAL')][0]:.0f}", b_wid0_t1=f"{B['wid'][mi.index('T1-TSK')][0]:.0f}", b_wid0_ens=f"{B['wid'][mi.index('Ens-MLP')][0]:.0f}")
r0 = S['FD001']['FOUCAL'][0]; V.update(nseed=str(seeds), params=f"{r0['params']:,}", lat=f"{np.mean([r['latency_ms'] for fd in FDS for r in S[fd]['FOUCAL']]):.2f}")
# dataset table
from foucal import data as Da
dr = []
for fd in FDS:
    d = Da.load(fd, 0); dr.append([fd, '6' if Da.MULTI[fd] else '1', '2' if fd in ('FD003', 'FD004') else '1', f"{len(d['tr']['y']):,} / {len(d['va']['y']):,} / {len(d['ca']['y']):,}", str(len(d['te']['y']))])
T['DATASET_ROWS'] = dr
json.dump(dict(T=T, V=V), open('results/tables.json', 'w'), indent=1, ensure_ascii=False)
print('seeds', seeds, {k: V[k] for k in ['nbest', 'nnarrow', 'wred', 'rho_foucal', 'rho_t1', 'covmin', 'covmax', 'vcovmin', 'vcovmax', 'b_cov0', 'b_wid0', 'b_wid0_t1']})
