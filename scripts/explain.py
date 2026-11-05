import _paths  # noqa: F401  (repository root on path, run from root)
import pickle, json, numpy as np, jax, jax.numpy as jnp
from foucal import data as Da, model as Mo
def rules(fd, top=6, nf=3):
    p = pickle.load(open(f'checkpoints/foucal_{fd}_s0.pkl', 'rb')); d = Da.load(fd, 0); names = d['names']
    F = jnp.asarray(d['te']['F']); f_lo, f_hi = Mo.firing(p, F)
    sup = np.asarray(f_hi / f_hi.sum(1, keepdims=True)).mean(0)
    a = np.asarray(jax.nn.softplus(p['a'])); c = np.asarray(p['c']); ratio = 1 + np.asarray(jax.nn.softplus(p['dl']))
    cons = (np.asarray(p['c']) * np.asarray(p['w'])).sum(1) + np.asarray(p['b'])
    out = []
    for r in np.argsort(-sup)[:top]:
        js = np.argsort(-a[r])[:nf]
        term = lambda v: 'LOW' if v < 1 / 3 else ('HIGH' if v > 2 / 3 else 'MEDIUM')
        out.append(dict(rule=int(r) + 1, ante=' AND '.join(f'{names[j]} is {term(c[r, j])}' for j in js),
                        rul=float(np.clip(cons[r] * 125, 0, 125)), fou=float(ratio[r, js].mean()), support=float(sup[r])))
    return out, float(ratio.mean()), float(ratio.min()), float(ratio.max())
if __name__ == '__main__':
    res = {}
    for fd in ['FD001', 'FD002']:
        o, m, lo, hi = rules(fd); res[fd] = dict(rules=o, fou_mean=m, fou_min=lo, fou_max=hi)
        print(fd, round(m, 2), round(lo, 2), round(hi, 2)); [print('  ', x) for x in o]
    json.dump(res, open('results/rules.json', 'w'), indent=1)
