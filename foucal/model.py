"""Interval type 2 Takagi Sugeno model with exact Karnik Mendel type reduction (JAX)."""
import jax, jax.numpy as jnp, numpy as np

def init(key, R, D, centers, it2=True):
    k1, k2 = jax.random.split(key)
    return {'c': jnp.asarray(centers),                       # (R, D) rule centres
            'ls': jnp.full((R, D), np.log(0.25)),           # log lower (narrow) width sigma_lo
            'dl': jnp.full((R, D), -1.0 if it2 else -1e9),  # FOU spread: sigma_hi = sigma_lo (1 + softplus(dl))
            'a': jnp.zeros((R, D)),                          # antecedent importance a = softplus(a)
            'w': jax.random.normal(k1, (R, D)) * 0.01, 'b': jnp.full((R,), 0.5)}

def firing(p, F):
    """log lower and upper firing strengths, (N, R)."""
    sl = jnp.exp(p['ls']); su = sl * (1.0 + jax.nn.softplus(p['dl'])); a = jax.nn.softplus(p['a'])
    d2 = (F[:, None, :] - p['c'][None]) ** 2                # (N, R, D)
    lf_lo = -jnp.sum(a * d2 / (2 * sl ** 2), -1)            # narrow width -> lower membership
    lf_hi = -jnp.sum(a * d2 / (2 * su ** 2), -1)            # wide width -> upper membership
    m = jnp.max(lf_hi, -1, keepdims=True)                   # common scale; KM ratios are scale invariant
    return jnp.exp(lf_lo - m), jnp.exp(lf_hi - m)

def consequents(p, F): return F @ p['w'].T + p['b']        # first order TSK consequents, (N, R)

def km(f_lo, f_hi, y):
    """Exact Karnik Mendel interval [y_l, y_r] by enumerating all R+1 switch points (vectorised)."""
    o = jnp.argsort(y, -1); ys = jnp.take_along_axis(y, o, -1)
    lo = jnp.take_along_axis(f_lo, o, -1); hi = jnp.take_along_axis(f_hi, o, -1)
    R = y.shape[-1]; tri = jnp.tril(jnp.ones((R + 1, R)), -1)  # row k: first k rules
    # y_l: upper firing on rules with small consequents, lower on the rest
    wl = tri[None] * hi[:, None] + (1 - tri[None]) * lo[:, None]
    wr = tri[None] * lo[:, None] + (1 - tri[None]) * hi[:, None]
    yl = jnp.sum(wl * ys[:, None], -1) / (jnp.sum(wl, -1) + 1e-12)
    yr = jnp.sum(wr * ys[:, None], -1) / (jnp.sum(wr, -1) + 1e-12)
    return jnp.min(yl, -1), jnp.max(yr, -1)

def forward(p, F):
    f_lo, f_hi = firing(p, F); y = consequents(p, F)
    yl, yr = km(f_lo, f_hi, y)
    return (yl + yr) / 2, yr - yl, (f_lo, f_hi, y)
