import numpy as np, scipy
from scipy.signal import whittaker_henderson
print("scipy", scipy.__version__, "numpy", np.__version__)

rng = np.random.default_rng(0)
n = 30
x = np.arange(n)
y = np.sin(2 * np.pi * x / n) + 0.1 * rng.standard_normal(n)
w = rng.uniform(0.5, 3.0, n)

def wls_line(y, w, deg):
    # minimizes sum_i w_i * (y_i - p(x_i))**2, the WH objective for lamb -> inf
    V = np.vander(x, deg + 1)
    sw = np.sqrt(w)
    c = np.linalg.lstsq(sw[:, None] * V, sw * y, rcond=None)[0]
    return V @ c

ref = wls_line(y, w, 1)
for lamb in [1e6, 1e8, 1e10, 1e16, 1e20]:
    res = whittaker_henderson(y, lamb=lamb, order=2, weights=w)
    print(f"lamb={lamb:.0e}: max|WH - weighted LS line| = {np.max(np.abs(res.x - ref)):.3e}")

# numpy Polynomial.fit's `w` multiplies the unsquared residual
p_w = np.polynomial.Polynomial.fit(x, y, 1, w=w)(x)
p_sqrtw = np.polynomial.Polynomial.fit(x, y, 1, w=np.sqrt(w))(x)
r20 = whittaker_henderson(y, lamb=1e20, order=2, weights=w).x
print("WH(1e20) vs Polynomial.fit(w=w):      ", np.max(np.abs(r20 - p_w)))
print("WH(1e20) vs Polynomial.fit(w=sqrt(w)):", np.max(np.abs(r20 - p_sqrtw)))
r10 = whittaker_henderson(y, lamb=1e10, order=2, weights=w).x
print("WH(1e10) vs WH(1e20):", np.max(np.abs(r10 - r20)))

# scaling invariance: (lamb, w) -> (c*lamb, c*w) must not change the result
a = whittaker_henderson(y, lamb=1e20, order=2, weights=w).x
b = whittaker_henderson(y, lamb=4e20, order=2, weights=4 * w).x
print("scale invariance 1e20 vs 4e20,4w:", np.max(np.abs(a - b)))

# runner-up: binned_statistic min/max with NaN
from scipy.stats import binned_statistic
v = np.array([1.0, np.nan, 3.0])
xx = np.array([0.1, 0.2, 0.3])
for s in ["min", "max", "mean", "median", np.min, np.max]:
    print("binned_statistic", s, binned_statistic(xx, v, statistic=s, bins=1).statistic)
