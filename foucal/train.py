"""Training, conformal calibration and evaluation for FOUCAL and all baselines."""
import time, numpy as np, jax, jax.numpy as jnp, optax
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from . import model as Mo, data as Da

ALPHA = 0.10; GAMMA = 0.02
STATES = (50.0, 100.0)   # linguistic health states on predicted RUL: CRITICAL <= 50 < DEGRADING <= 100 < HEALTHY
def health_state(yh): return np.digitize(np.clip(yh, 0, 125), STATES, right=True)   # miscoverage level; width floor in normalised RUL units (2.5 cycles)

def qhat(scores, alpha=ALPHA):
    n = len(scores); k = min(n, int(np.ceil((n + 1) * (1 - alpha))))
    return float(np.sort(scores)[k - 1])

def adam_loop(params, loss_fn, Xs, y, Xv, yv, pred_fn, epochs, lr, bs, seed):
    n = len(y); steps = epochs * int(np.ceil(n / bs))
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(optax.cosine_decay_schedule(lr, steps)))
    st = opt.init(params); rng = np.random.RandomState(seed)
    @jax.jit
    def step(p, st, xb, yb):
        L, g = jax.value_and_grad(loss_fn)(p, xb, yb); u, st = opt.update(g, st, p); return optax.apply_updates(p, u), st, L
    best, bp = 1e9, params
    for ep in range(epochs):
        idx = rng.permutation(n)
        for i in range(0, n, bs):
            b = idx[i:i + bs]; params, st, L = step(params, st, Xs[b], y[b])
        v = float(np.sqrt(np.mean((np.clip(pred_fn(params, Xv), 0, 1) - yv) ** 2)))
        if v < best: best, bp = v, jax.tree_util.tree_map(lambda a: a, params)
    return bp

# ---------------- fuzzy models ----------------
def fit_tsk(d, seed, it2=True, nll=True, R=16, epochs=80, lam=0.1):
    F, y = d['tr']['F'], d['tr']['y'] / d['cap']
    km = KMeans(R, n_init=4, random_state=seed).fit(F[np.random.RandomState(seed).choice(len(F), min(len(F), 5000), replace=False)])
    p = Mo.init(jax.random.PRNGKey(seed), R, F.shape[1], km.cluster_centers_, it2=it2)
    def loss(p, Fb, yb):
        yh, w, _ = Mo.forward(p, Fb); L = jnp.mean((yh - yb) ** 2)
        if nll:
            s = w / 2 + 0.01; L = L + lam * 1e-2 * jnp.mean(jnp.log(s) + (yb - jax.lax.stop_gradient(yh)) ** 2 / (2 * s ** 2))
        return L + 1e-4 * jnp.mean(jax.nn.softplus(p['a']))
    fwd = jax.jit(lambda p, F: Mo.forward(p, F)[:2])
    pred = lambda p, F: np.asarray(fwd(p, F)[0])
    p = adam_loop(p, loss, F, y, d['va']['F'], d['va']['y'] / d['cap'], pred, epochs, 3e-3, 512, seed)
    out = lambda F: tuple(np.asarray(a) for a in fwd(p, jnp.asarray(F)))
    return p, out

# ---------------- deep ensemble MLP ----------------
def fit_ens(d, seed, M=5, epochs=60):
    F, y = d['tr']['F'], d['tr']['y'] / d['cap']; Dm = F.shape[1]; nets = []
    def mlp(p, x):
        h = jax.nn.relu(x @ p['W1'] + p['b1']); h = jax.nn.relu(h @ p['W2'] + p['b2']); return (h @ p['W3'] + p['b3'])[:, 0]
    for m in range(M):
        k = jax.random.split(jax.random.PRNGKey(100 * seed + m), 3)
        p = {'W1': jax.random.normal(k[0], (Dm, 64)) * np.sqrt(2 / Dm), 'b1': jnp.zeros(64), 'W2': jax.random.normal(k[1], (64, 64)) * np.sqrt(2 / 64),
             'b2': jnp.zeros(64), 'W3': jax.random.normal(k[2], (64, 1)) * 0.05, 'b3': jnp.full((1,), 0.5)}
        f = jax.jit(mlp); p = adam_loop(p, lambda p, x, y: jnp.mean((mlp(p, x) - y) ** 2), F, y, d['va']['F'], d['va']['y'] / d['cap'],
                                       lambda p, x: np.asarray(f(p, x)), epochs, 2e-3, 256, 100 * seed + m)
        nets.append((f, p))
    def out(F):
        P = np.stack([np.asarray(f(p, jnp.asarray(F))) for f, p in nets]); return P.mean(0), P.std(0)
    return nets, out

# ---------------- LSTM with quantile heads (for CQR) ----------------
QS = np.array([ALPHA / 2, 0.5, 1 - ALPHA / 2], np.float32)
def fit_lstm_q(d, seed, hid=48, epochs=30):
    X, y = d['tr']['X'], d['tr']['y'] / d['cap']; J = X.shape[2]
    k = jax.random.split(jax.random.PRNGKey(seed), 2)
    p = {'Wg': jax.random.normal(k[0], (J + hid, 4 * hid)) * np.sqrt(1 / (J + hid)), 'bg': jnp.zeros(4 * hid),
         'Wo': jax.random.normal(k[1], (hid, 3)) * 0.05, 'bo': jnp.full((3,), 0.5)}
    def net(p, x):
        B = x.shape[0]
        def stp(c, xt):
            h, cc = c; z = jnp.concatenate([xt, h], -1) @ p['Wg'] + p['bg']; i, f, g, o = jnp.split(z, 4, -1)
            cc = jax.nn.sigmoid(f + 1) * cc + jax.nn.sigmoid(i) * jnp.tanh(g); h = jax.nn.sigmoid(o) * jnp.tanh(cc); return (h, cc), None
        (h, _), _ = jax.lax.scan(stp, (jnp.zeros((B, hid)), jnp.zeros((B, hid))), x.transpose(1, 0, 2))
        return h @ p['Wo'] + p['bo']
    def loss(p, x, yb):
        e = yb[:, None] - net(p, x); return jnp.mean(jnp.maximum(QS * e, (QS - 1) * e))
    f = jax.jit(net)
    p = adam_loop(p, loss, X, y, d['va']['X'], d['va']['y'] / d['cap'], lambda p, x: np.asarray(f(p, x))[:, 1], epochs, 2e-3, 256, seed)
    def out(X):
        o = np.concatenate([np.asarray(f(p, jnp.asarray(X[i:i + 4096]))) for i in range(0, len(X), 4096)]); o.sort(1); return o
    return p, out

def fit_hgb_q(d, seed):
    F, y = d['tr']['F'], d['tr']['y'] / d['cap']
    ms = [HistGradientBoostingRegressor(loss='quantile', quantile=q, max_iter=400, learning_rate=0.05, random_state=seed, early_stopping=True).fit(F, y) for q in QS]
    def out(F): o = np.stack([m.predict(F) for m in ms], 1); o.sort(1); return o
    return ms, out

# ---------------- metrics ----------------
def evaluate(y, yh, lo, hi, groups=None, alpha=ALPHA):
    yh = np.clip(yh, 0, 125); lo = np.clip(lo, 0, 125); hi = np.clip(hi, 0, 125); e = yh - y
    cov = (y >= lo) & (y <= hi); width = hi - lo
    wink = width + (2 / alpha) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
    bands = [(y <= 50), (y > 50) & (y <= 100), (y > 100)]
    bcov = [float(cov[b].mean()) for b in bands if b.sum() >= 5]
    m = dict(RMSE=float(np.sqrt(np.mean(e ** 2))), MAE=float(np.mean(np.abs(e))),
             Score=float(np.sum(np.where(e < 0, np.exp(-e / 13) - 1, np.exp(e / 10) - 1))),
             cov=float(cov.mean()), width=float(width.mean()), winkler=float(wink.mean()),
             worst_band_cov=float(min(bcov)), rho=float(spearmanr(width, np.abs(e)).correlation) if width.std() > 1e-9 else 0.0)
    if groups is not None and len(np.unique(groups)) > 1:
        m['worst_regime_cov'] = float(min(cov[groups == g].mean() for g in np.unique(groups) if (groups == g).sum() >= 5))
    return m

_CACHE = {}
def run(fd, method, seed, fast=False):
    key_d = ('data', fd, seed)
    if key_d not in _CACHE: _CACHE.clear(); _CACHE[key_d] = Da.load(fd, seed)
    d = _CACHE[key_d]; C = d['cap']; t0 = time.time(); ca = d['ca']; yc = ca['y']
    res = {}; sets = {'va': d['va'], 'te': d['te']}
    if method in ('FOUCAL', 'FOUCAL-R', 'FOUCAL-noMondrian', 'IT2-raw', 'IT2-split', 'IT2-MSE', 'T1-TSK'):
        it2 = method != 'T1-TSK'; nll = method not in ('IT2-MSE', 'T1-TSK')
        kt = ('tsk', fd, seed, it2, nll)
        if kt not in _CACHE: _CACHE[kt] = fit_tsk(d, seed, it2=it2, nll=nll, epochs=20 if fast else 80)
        p, out = _CACHE[kt]
        o = {k: tuple(a * C for a in out(v['F'])) for k, v in [('ca', ca)] + list(sets.items())}
        mc, wc = o['ca']; sc_all = np.abs(yc - np.clip(mc, 0, C)) / (wc + GAMMA * C)
        def grp(m, S):
            hs = health_state(m)
            if method == 'FOUCAL-R': return hs * 10 + S['g']
            if method == 'FOUCAL': return hs
            return np.zeros(len(m), int)
        gc = grp(mc, ca); qs = {}
        for g in np.unique(gc):
            sel = gc == g; qs[int(g)] = qhat(sc_all[sel]) if sel.sum() >= 30 else qhat(sc_all)
        q_split = qhat(np.abs(yc - np.clip(mc, 0, C)))
        def intervals(k):
            m, w = o[k]
            if method == 'IT2-raw': return m, m - w / 2, m + w / 2
            if method in ('IT2-split', 'T1-TSK'): return m, m - q_split, m + q_split
            g = grp(m, sets[k]); q = np.array([qs.get(int(x), qhat(sc_all)) for x in g])
            return m, m - q * (w + GAMMA * C), m + q * (w + GAMMA * C)
        res['q_by_group'] = qs
        res['params'] = int(sum(np.prod(a.shape) for a in jax.tree_util.tree_leaves(p))); res['_p'] = p
        te = d['te']; out(te['F'][:1]); t1 = time.perf_counter()
        for i in range(200): out(te['F'][i % len(te['y'])][None])
        res['latency_ms'] = (time.perf_counter() - t1) / 200 * 1e3
    elif method == 'Ens-MLP':
        _, out = fit_ens(d, seed, epochs=15 if fast else 60)
        o = {k: tuple(a * C for a in out(v['F'])) for k, v in [('ca', ca)] + list(sets.items())}
        mc, sc = o['ca']; q = qhat(np.abs(yc - np.clip(mc, 0, C)) / (sc + GAMMA * C))
        def intervals(k): m, s_ = o[k]; return m, m - q * (s_ + GAMMA * C), m + q * (s_ + GAMMA * C)
    elif method in ('LSTM-CQR', 'HGB-CQR'):
        if method == 'LSTM-CQR': _, out = fit_lstm_q(d, seed, epochs=5 if fast else 30); key = 'X'
        else: _, out = fit_hgb_q(d, seed); key = 'F'
        o = {k: out(v[key]) * C for k, v in [('ca', ca)] + list(sets.items())}
        q = qhat(np.maximum(o['ca'][:, 0] - yc, yc - o['ca'][:, 2]))
        def intervals(k): oo = o[k]; return oo[:, 1], oo[:, 0] - q, oo[:, 2] + q
    for k, S in sets.items():
        m, lo, hi = intervals(k)
        ev = evaluate(S['y'], m, lo, hi, S['g'] if Da.MULTI[fd] else None)
        if k == 'te': res.update(ev); res['pred'] = dict(y=S['y'].tolist(), yh=np.clip(m, 0, C).tolist(), lo=np.clip(lo, 0, C).tolist(), hi=np.clip(hi, 0, C).tolist(), g=S['g'].tolist())
        else: res.update({'val_' + kk: v for kk, v in ev.items() if kk in ('cov', 'width', 'winkler', 'worst_band_cov', 'worst_regime_cov')})
    res['train_s'] = time.time() - t0
    return res
