"""Fig E: stability. Final energy ratio E(t_f)/E0 vs omega0*h near the limits
omega0*h = 2 (VV) and 2*sqrt(2) ~ 2.83 (RK4).
Linear model (where the limits are exact), beta = 0, theta0 = 0.1, t_f = 200.
"""
import numpy as np
import matplotlib.pyplot as plt
import pendulum as p
from style import COLOR, STEPS, INK, HALF, save

THETA0 = 0.1
T_F = 200.0
h_list = np.linspace(0.5, 3.2, 109)          # grid spacing 0.025

fig, ax = plt.subplots(figsize=(HALF, 3.0))
E0 = p.energy(THETA0, 0.0, True)
ratios = {}
for name, step in STEPS.items():
    ratio = []
    for h in h_list:
        with np.errstate(over="ignore", invalid="ignore"):
            t, th, om = p.integrate(step, THETA0, 0.0, h, T_F, 0.0, True)
            ratio.append(p.energy(th[-1], om[-1], True) / E0)
    ratio = np.array(ratio)
    ratios[name] = ratio
    ax.semilogy(h_list, ratio, "o-", ms=1.8, lw=0.8, color=COLOR[name], label=name,
                zorder=3 if name == "VV" else 2)
    first = h_list[np.argmax(ratio > 10)]
    i_min = np.argmin(ratio)
    print(f"{name:3}: E/E0 > 10 first at h = {first:.3f}; min E/E0 = {ratio[i_min]:.1e} at h = {h_list[i_min]:.2f}")

for x, lab in ((2.0, "ω₀h = 2"), (2 * np.sqrt(2), "ω₀h = 2√2")):
    ax.axvline(x, color=INK, ls=":", lw=0.9)
    if x > 2.5:
        ax.text(x + 0.04, 1e-55, lab, rotation=90, ha="left", va="bottom", fontsize=7, color=INK)
        continue
    ax.text(x - 0.04, 1e-55, lab, rotation=90, ha="right", va="bottom", fontsize=7, color=INK)
ax.axhline(1, color=INK, lw=0.7, zorder=1)
ax.set(xlabel="ω₀h", ylabel=rf"$E(t_f)\,/\,E_0$,  $t_f\approx{T_F:g}$", ylim=(1e-60, 1e100),
       title=f"Stability: linear model, β = 0")
ax.set_yticks([1e-60, 1e-40, 1e-20, 1, 1e20, 1e40])
ax.set_yticks([1e-60, 1e-30, 1, 1e30, 1e60, 1e90])
ax.set_yticks([], minor=True)
ax.legend(loc="lower left", fontsize=7)

# inset: VV below its limit on a linear scale, with the analytic band minimum 1 - h^2/4
X0, W = 0.15, 0.4                                # inset footprint in axes fraction
ins = ax.inset_axes([X0, 0.66, W, 0.22])
lo, hi = ax.get_xlim()
x_lo, x_hi = lo + X0 * (hi - lo), lo + (X0 + W) * (hi - lo)    # same h range as the main axis below it
m = (h_list >= x_lo) & (h_list < 2)
ins.plot(h_list[m], ratios["VV"][m], "o", ms=1.5, color=COLOR["VV"])
ins.plot(h_list[m], 1 - h_list[m] ** 2 / 4, color=INK, lw=0.7, ls=":")   # band: [1 - h^2/4, 1]
ins.set_ylim(0, 1.1)
ins.set_xlim(x_lo, x_hi)
ins.tick_params(labelsize=7, pad=1)
ins.set_xticks([1, 1.5])
ins.axhline(1, color=INK, lw=0.7, ls=":")
ins.set_yticks([0, 0.5, 1])
ins.set_title("VV, linear y-scale", fontsize=6.5, color=INK, pad=1.5)
fig.tight_layout()
save(fig, "fig_E_stability")
