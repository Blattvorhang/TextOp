import numpy as np
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except Exception:
    pass


# ============================================================
# 1. Main trajectory
# ============================================================
def nominal_curve(x):
    """
    Slightly irregular periodic trajectory:
    preserves the global sinusoidal trend but avoids looking
    like a perfectly regular trigonometric function.
    """
    return (
        0.18 * np.sin(2 * np.pi * x)
        + 0.022 * np.sin(4 * np.pi * x + 0.55)
        - 0.010 * np.cos(3 * np.pi * x - 0.35)
    )


# ============================================================
# 2. Quintic smootherstep
#    zero velocity and acceleration at both endpoints
# ============================================================
def smootherstep(s):
    s = np.clip(s, 0.0, 1.0)
    return 6 * s**5 - 15 * s**4 + 10 * s**3


# ============================================================
# 3. Deviation profile
# ============================================================
x_current = 0.31
x_decay_end = 0.70
delta0 = 0.10


def deviation_width(x):
    x = np.asarray(x)
    d = np.zeros_like(x, dtype=float)

    # History: constant deviation
    hist = x <= x_current
    d[hist] = delta0

    # Recovery: smoothly decay deviation to zero
    trans = (x > x_current) & (x < x_decay_end)
    s = (x[trans] - x_current) / (x_decay_end - x_current)

    d[trans] = delta0 * (1.0 - smootherstep(s))

    # Fully recovered
    d[x >= x_decay_end] = 0.0

    return d


# ============================================================
# 4. Generate trajectory and band
# ============================================================
x = np.linspace(0.0, 1.0, 800)

y = nominal_curve(x)
delta = deviation_width(x)

y_upper = y + delta
y_lower = y - delta


# ============================================================
# 5. Plot
# ============================================================
fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=180)

# Deviation band
ax.fill_between(
    x,
    y_lower,
    y_upper,
    alpha=0.25,
    linewidth=0,
    zorder=1,
)

# Main trajectory
ax.plot(
    x,
    y,
    linewidth=3.0,
    zorder=3,
)

# Clean layout
ax.set_xlim(-0.01, 1.02)

margin = 0.025
ax.set_ylim(
    y.min() - delta0 - margin,
    y.max() + delta0 + margin,
)

ax.set_xticks([])
ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout(pad=0.25)
plt.show()