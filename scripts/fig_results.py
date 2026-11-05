import _paths  # noqa: F401  (repository root on path, run from root)
import json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = [json.loads(l) for l in open('results/results.jsonl')]
get = lambda fd, m, s: next(r for r in R if r['data'] == fd and r['method'] == m and r['seed'] == s)
# Fig 3: FD002 seed 0 intervals, FOUCAL vs T1-TSK
fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
for ax, m, lab in zip(axs, ['FOUCAL', 'T1-TSK'], ['(a) FOUCAL', '(b) T1-TSK with split conformal']):
    p = get('FD002', m, 0)['pred']; y = np.array(p['y']); o = np.argsort(y, kind='stable')
    lo, hi, yh = np.array(p['lo'])[o], np.array(p['hi'])[o], np.array(p['yh'])[o]; x = np.arange(len(y))
    cov = np.mean((y[o] >= lo) & (y[o] <= hi))
    ax.fill_between(x, lo, hi, color='#9ecae1', alpha=0.65, step='mid', label=f'90% interval (coverage {cov:.2f}, mean width {np.mean(hi - lo):.1f})')
    ax.plot(x, y[o], 'k-', lw=1.3, label='true RUL'); ax.plot(x, yh, '.', ms=3, color='#d62728', label='predicted RUL')
    ax.set_title(lab, fontsize=10); ax.set_xlabel('FD002 test engine (sorted by true RUL)'); ax.legend(fontsize=7, loc='upper left'); ax.set_ylim(-5, 160)
axs[0].set_ylabel('RUL (cycles)')
plt.tight_layout(); plt.savefig('figures/Fig3_intervals_fd002.png', dpi=300); plt.close()
# Fig 4: coverage and width per true RUL band, pooled over subsets and seeds
methods = ['FOUCAL', 'T1-TSK', 'Ens-MLP', 'HGB-CQR', 'LSTM-CQR']; bands = [(0, 50), (50, 100), (100, 126)]; blab = ['≤ 50', '51–100', '> 100']
cov = np.zeros((len(methods), 3)); wid = np.zeros((len(methods), 3))
for i, m in enumerate(methods):
    Y, L, H = [], [], []
    for r in R:
        if r['method'] == m: Y += r['pred']['y']; L += r['pred']['lo']; H += r['pred']['hi']
    Y, L, H = map(np.array, (Y, L, H))
    for j, (a, b) in enumerate(bands):
        s = (Y > a if a else Y >= 0) & (Y <= b - (1 if b == 126 else 0)) if j < 2 else (Y > 100)
        if j == 0: s = Y <= 50
        if j == 1: s = (Y > 50) & (Y <= 100)
        cov[i, j] = np.mean((Y[s] >= L[s]) & (Y[s] <= H[s])); wid[i, j] = np.mean(H[s] - L[s])
fig, axs = plt.subplots(1, 2, figsize=(11, 3.3)); w = 0.16; hatch = ['', '//', '..', 'xx', '\\\\']
cols = ['#08519c', '#bdbdbd', '#74c476', '#fd8d3c', '#9e9ac8']
for i, m in enumerate(methods):
    axs[0].bar(np.arange(3) + (i - 2) * w, cov[i], w, label=m, color=cols[i], hatch=hatch[i], edgecolor='k', lw=0.4)
    axs[1].bar(np.arange(3) + (i - 2) * w, wid[i], w, label=m, color=cols[i], hatch=hatch[i], edgecolor='k', lw=0.4)
axs[0].axhline(0.9, color='k', ls='--', lw=1); axs[0].set_ylim(0.5, 1.0); axs[0].set_ylabel('coverage'); axs[0].set_title('(a) Coverage by true RUL band', fontsize=10)
axs[1].set_ylabel('mean interval width (cycles)'); axs[1].set_title('(b) Width by true RUL band', fontsize=10)
for ax in axs: ax.set_xticks(range(3)); ax.set_xticklabels([f'RUL {b}' for b in blab])
axs[1].legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.savefig('figures/Fig4_band_coverage.png', dpi=300); plt.close()
json.dump(dict(methods=methods, cov=cov.tolist(), wid=wid.tolist()), open('results/bands.json', 'w'))
print(np.round(cov, 3)); print(np.round(wid, 1))
