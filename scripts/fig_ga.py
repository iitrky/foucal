import _paths  # noqa: F401
import json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
R = [json.loads(l) for l in open('results/results.jsonl')]
fig = plt.figure(figsize=(9, 4)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 9); ax.set_ylim(0, 4); ax.axis('off')
def box(x, y, w, h, t, c, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.03,rounding_size=0.12', fc=c, ec='#333', lw=1))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs, weight='bold' if bold else 'normal')
def arr(a, b): ax.annotate('', b, a, arrowprops=dict(arrowstyle='->', lw=1.4, color='#333'))
ax.text(4.5, 3.78, 'FOUCAL: calibrated and explainable remaining useful life intervals', ha='center', fontsize=10.5, weight='bold')
box(0.1, 2.0, 1.5, 1.3, 'Turbofan sensor\nwindow\n(NASA C-MAPSS)', '#e8eef7')
# FOU inset
box(1.95, 1.75, 2.0, 1.8, '', '#fdf1dc')
ia = fig.add_axes([0.245, 0.585, 0.16, 0.17]); x = np.linspace(0, 1, 200)
ia.fill_between(x, np.exp(-(x - .5) ** 2 / (2 * .09 ** 2)), np.exp(-(x - .5) ** 2 / (2 * .16 ** 2)), color='#9ecae1'); ia.plot(x, np.exp(-(x - .5) ** 2 / (2 * .16 ** 2)), color='#08519c'); ia.plot(x, np.exp(-(x - .5) ** 2 / (2 * .09 ** 2)), color='#d62728')
ia.set_xticks([]); ia.set_yticks([]); ia.set_facecolor('#fdf1dc')
ax.text(2.95, 3.3, 'Interval type 2 fuzzy rules', ha='center', fontsize=8.5, weight='bold')
ax.text(2.95, 1.92, 'FOU trained as error scale\nexact Karnik Mendel interval', ha='center', fontsize=7.8)
box(4.3, 2.0, 1.75, 1.3, 'Conformal calibration\nscore |y − ŷ| / (FOU width)\nper health state\n(CRITICAL / DEGRADING /\nHEALTHY)', '#e6f4ea', 7.6)
arr((1.6, 2.65), (1.95, 2.65)); arr((3.95, 2.65), (4.3, 2.65)); arr((6.05, 2.65), (6.3, 2.65))
# result inset
p = next(r for r in R if r['data'] == 'FD002' and r['method'] == 'FOUCAL' and r['seed'] == 0)['pred']
y = np.array(p['y']); o = np.argsort(y, kind='stable')
ib = fig.add_axes([0.75, 0.42, 0.235, 0.42])
ib.fill_between(np.arange(len(y)), np.array(p['lo'])[o], np.array(p['hi'])[o], color='#9ecae1', step='mid'); ib.plot(y[o], 'k-', lw=1); ib.plot(np.array(p['yh'])[o], '.', ms=1.5, color='#d62728')
ib.set_title('Calibrated 90% RUL intervals (FD002)', fontsize=7.5); ib.tick_params(labelsize=6); ib.set_ylabel('RUL (cycles)', fontsize=6.5); ib.set_xlabel('test engines', fontsize=6.5)
box(0.1, 0.15, 8.8, 1.25, '', '#f7f7f7')
ax.text(0.3, 1.08, 'Results on all four C-MAPSS subsets', fontsize=8.5, weight='bold')
ax.text(0.3, 0.32, '•  narrowest calibrated intervals on every subset\n•  best or second best interval score against deep ensemble, LSTM, gradient boosting and type 1 fuzzy\n•  near end of life: 24-cycle intervals at 0.90 coverage vs 41 cycles for type 1 fuzzy\n•  every wide interval explained by linguistic rules and their footprint of uncertainty', fontsize=7.6, va='bottom')
plt.savefig('figures/Graphical_abstract.png', dpi=300); plt.close()
from PIL import Image; print(Image.open('figures/Graphical_abstract.png').size)
