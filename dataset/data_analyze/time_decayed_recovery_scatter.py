import numpy as np
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except Exception:
    pass

np.random.seed(7)


# ============================================================
# 1. Nominal curve: one main wave with slight irregularity
# ============================================================
def nominal_curve(x):
    """
    Keep the overall one-period trend, but add small harmonic
    perturbations so it does not look like a perfectly regular sine.
    """
    return (
        0.18 * np.sin(2 * np.pi * x)                  # main trend
        + 0.022 * np.sin(4 * np.pi * x + 0.55)       # mild local variation
        - 0.010 * np.cos(3 * np.pi * x - 0.35)       # asymmetry
    )


# ============================================================
# 2. Smooth decay profile
# ============================================================
def smootherstep(s):
    s = np.clip(s, 0.0, 1.0)
    return 6 * s**5 - 15 * s**4 + 10 * s**3


# ============================================================
# 3. Config
# ============================================================
x_current = 0.31
x_decay_end = 0.70     # slightly faster decay
delta0 = 0.10

def deviation_width(x):
    x = np.asarray(x)
    d = np.empty_like(x, dtype=float)

    hist = x <= x_current
    d[hist] = delta0

    trans = (x > x_current) & (x < x_decay_end)
    s = (x[trans] - x_current) / (x_decay_end - x_current)
    d[trans] = delta0 * (1.0 - smootherstep(s))

    tail = x >= x_decay_end
    d[tail] = 0.0

    return d


# ============================================================
# 4. Color interpolation
# ============================================================
def blend(c1, c2, t):
    c1 = np.asarray(c1)
    c2 = np.asarray(c2)
    return tuple((1 - t) * c1 + t * c2)

light_gray = np.array([0.80, 0.82, 0.85])
mid_blue   = np.array([0.55, 0.72, 0.92])
deep_blue  = np.array([0.16, 0.38, 0.67])


# ============================================================
# 5. Build dense and evenly filled scatter band
# ============================================================

# Reduce horizontal density so it matches the vertical density better
x_cols = np.linspace(0.02, 1.00, 40)
dx = x_cols[1] - x_cols[0]

xs_all, ys_all, cs_all = [], [], []

max_layers = 5

for x in x_cols:
    y0 = nominal_curve(x)
    d = deviation_width(np.array([x]))[0]

    frac = d / delta0 if delta0 > 0 else 0.0

    # 5 -> 4 -> 3 -> 2 -> 1 layers as the band narrows
    n_layers = max(
        1,
        int(np.round(1 + (max_layers - 1) * frac))
    )

    if n_layers == 1:
        u_levels = np.array([0.0])
    else:
        u_levels = np.linspace(-1.0, 1.0, n_layers)

    for j, u in enumerate(u_levels):
        stagger = 0.0
        if n_layers > 1:
            stagger = ((j % 2) - 0.5) * 0.38 * dx

        xj = x + stagger + 0.06 * dx * np.random.randn()
        yj = (
            y0
            + u * d
            + 0.007 * max(frac, 0.12) * np.random.randn()
        )

        if x <= x_current:
            t = x / x_current
            c = blend(light_gray, mid_blue, 0.60 * t)
        else:
            t = (x - x_current) / (1.0 - x_current)
            c = blend(mid_blue, deep_blue, t)

        xs_all.append(xj)
        ys_all.append(yj)
        cs_all.append(c)

# ============================================================
# 6. Reference curve only for determining plot limits
# ============================================================
x_curve = np.linspace(0.0, 1.0, 600)
y_curve = nominal_curve(x_curve)


# ============================================================
# 7. Plot
# ============================================================
fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=180)

# Scatter only
ax.scatter(
    xs_all,
    ys_all,
    s=68,
    c=cs_all,
    edgecolors="none",
    zorder=3
)

# Clean layout
ax.set_xlim(-0.01, 1.02)

margin = 0.025
ax.set_ylim(
    y_curve.min() - delta0 - margin,
    y_curve.max() + delta0 + margin
)

ax.set_xticks([])
ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout(pad=0.25)
plt.show()