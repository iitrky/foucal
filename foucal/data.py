"""C-MAPSS loading: sliding windows, per operating regime normalisation, fuzzy summary features."""
import os, numpy as np, pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
D = os.environ.get('FOU_DATA', str(Path(__file__).resolve().parents[1] / 'data'))
SENS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
NAMES = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'htBleed', 'W31', 'W32']
MULTI = {'FD001': False, 'FD002': True, 'FD003': False, 'FD004': True}
COLS = ['unit', 'cycle', 'o1', 'o2', 'o3'] + [f's{i}' for i in range(1, 22)]

def _read(f): return pd.read_csv(f'{D}/{f}', sep=r'\s+', header=None, names=COLS)

def load(fd, seed, W=30, cap=125, stride=2):
    tr, te = _read(f'train_{fd}.txt'), _read(f'test_{fd}.txt'); rul = np.loadtxt(f'{D}/RUL_{fd}.txt')
    sc = [f's{i}' for i in SENS]
    # operating regime: 6 clusters of the operating settings for FD002/FD004, one regime otherwise
    if MULTI[fd]:
        km = KMeans(6, n_init=10, random_state=0).fit(tr[['o1', 'o2', 'o3']].values)
        tr['reg'] = km.predict(tr[['o1', 'o2', 'o3']].values); te['reg'] = km.predict(te[['o1', 'o2', 'o3']].values)
    else:
        tr['reg'] = 0; te['reg'] = 0
    units = tr.unit.unique(); rng = np.random.RandomState(seed); rng.shuffle(units)
    nh = max(1, len(units) // 10); va_u, ca_u, tr_u = units[:nh], units[nh:2 * nh + nh], units[2 * nh + nh:]
    # regime-wise z-score with statistics of training engines only
    ref = tr[tr.unit.isin(tr_u)]
    mu = ref.groupby('reg')[sc].mean(); sd = ref.groupby('reg')[sc].std().replace(0, 1).fillna(1)
    for df in (tr, te):
        df[sc] = (df[sc].values - mu.loc[df.reg].values) / sd.loc[df.reg].values
    def build(df, us, last_only=False, true_rul=None, step=1):
        X, y, g, u_ = [], [], [], []
        for k, u in enumerate(us):
            d = df[df.unit == u]; a = d[sc].values; rg = d.reg.values
            if len(a) < W: a = np.vstack([np.repeat(a[:1], W - len(a), 0), a]); rg = np.concatenate([np.repeat(rg[:1], W - len(rg)), rg])
            if last_only:
                X.append(a[-W:]); y.append(min(true_rul[k], cap)); g.append(rg[-1]); u_.append(u); continue
            r = np.minimum(d.cycle.max() - d.cycle.values, cap)
            for i in range(0, len(a) - W + 1, step):
                X.append(a[i:i + W]); y.append(r[i + W - 1]); g.append(rg[i + W - 1]); u_.append(u)
        return np.array(X, np.float32), np.array(y, np.float32), np.array(g), np.array(u_)
    out = {}
    out['tr'] = build(tr, tr_u, step=stride); out['va'] = build(tr, va_u); out['ca'] = build(tr, ca_u)
    out['te'] = build(te, sorted(te.unit.unique()), last_only=True, true_rul=rul)
    # fuzzy summary features: recent level (mean of last 5 cycles) and trend (least squares slope) per sensor
    t = np.arange(W) - (W - 1) / 2; t = t / np.sum(t ** 2)
    def feats(X): return np.concatenate([X[:, -5:].mean(1), np.einsum('nwj,w->nj', X, t) * W], 1)
    F = {k: feats(v[0]) for k, v in out.items()}
    lo, hi = F['tr'].min(0), F['tr'].max(0); s = np.where(hi - lo > 1e-9, hi - lo, 1)
    for k in out:
        X, y, g, u = out[k]
        out[k] = dict(X=np.clip(X, -6, 6), F=np.clip((F[k] - lo) / s, -0.25, 1.25).astype(np.float32), y=y, g=g, u=u)
    out['names'] = [f'{n} level' for n in NAMES] + [f'{n} trend' for n in NAMES]
    out['cap'] = cap
    return out
