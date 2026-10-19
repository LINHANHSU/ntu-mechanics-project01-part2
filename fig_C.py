"""Fig C: long-time behaviour with beta = 0 (what a symplectic method buys you).

Methods are compared at EQUAL COST: VV uses 1 force evaluation per step, RK4 uses 4,
so VV at h and RK4 at 4h cost the same per unit time. Pairs: (0.05, 0.2), (0.025, 0.1).
(a) Relative energy error |E - E0| / E0 vs t, shown as the maximum over log-spaced
    time bins so VV's fast oscillation shows up as a band and is not aliased.
(b) Phase error from upward zero-crossing times compared with the exact period:
    delta_phi_n = 2 pi (t_n - t_n_exact) / T_exact.
Nonlinear model, theta0 = 1.0.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk
import pendulum as p
from style import COLOR, STEPS, INK, FULL, GUIDE, shade, save

THETA0 = 1.0
T_MAX = 1.0e4
PAIRS = ((0.05, 0.2), (0.025, 0.1))     # (h_VV, h_RK4) at equal evaluations per unit time
T_EXACT = 4 * ellipk(np.sin(THETA0 / 2) ** 2)      # period in units of 1/omega0


def period_of_energy(E):
    amp = np.arccos(1 - E)
    return 4 * ellipk(np.sin(amp / 2) ** 2)


def binned_max(t, y, n_bins=150):
    edges = np.geomspace(t[1], t[-1], n_bins + 1)
    idx = np.searchsorted(t, edges)
    tc, yc = [], []
    for a, b in zip(idx[:-1], idx[1:]):
        if b > a:
            tc.append(np.sqrt(t[a] * t[b - 1]))
            yc.append(np.max(y[a:b]))
    return np.array(tc), np.array(yc)


fig, (ax_E, ax_ph) = plt.subplots(1, 2, figsize=(FULL, 2.8))
E0 = p.energy(THETA0, 0.0, False)

runs = {}
for k, pair in enumerate(PAIRS):
    for name, h in zip(("VV", "RK4"), pair):
        c = shade(COLOR[name], k + 1, len(PAIRS) + 1)
        t, th, om = p.integrate(STEPS[name], THETA0, 0.0, h, T_MAX, 0.0, False)
        E = p.energy(th, om, False)
        dE = (E - E0) / E0
        tc = p.crossing_times(t, th, om)
        n = np.arange(len(tc))
        dphi = 2 * np.pi * (tc - (n + 0.75) * T_EXACT) / T_EXACT
        runs[name, h] = (t, E, dE, tc, dphi)

        tb, eb = binned_max(t, np.abs(dE))
        lab = f"{name}, h={h}  ({round(1 / h * (1 if name == 'VV' else 4))} evals per unit t)"
        ax_E.loglog(tb, eb, color=c, label=lab)
        ax_ph.loglog(tc, np.abs(dphi), color=c, label=lab)

        m = tb > 1e2
        slope = np.polyfit(np.log(tb[m]), np.log(eb[m]), 1)[0]
        print(f"{name:3} h={h:<5}: max|dE/E0| = {np.max(np.abs(dE)):.2e}, final dE/E0 = {dE[-1]:+.2e}, "
              f"growth exponent (t>100) = {slope:.2f}, phase error at t=1e4 = {dphi[-1]:+.3e} rad")

Omega = 2 * np.pi / T_EXACT
print(f"Linear-oscillator RK4 drift -Omega^6 h^5 t/72 with Omega={Omega:.3f}, h=0.1, t=1e4: {-Omega**6 * 0.1**5 * 1e4 / 72:.2e}")
print("\nEqual-cost comparison:")
for h_vv, h_rk in PAIRS:
    _, _, dE_v, tc_v, ph_v = runs["VV", h_vv]
    t_r, E_r, dE_r, tc_r, ph_r = runs["RK4", h_rk]
    band = np.max(np.abs(dE_v))
    t_E = t_r[np.argmax(np.abs(dE_r) > band)]
    m = min(len(ph_v), len(ph_r))
    worse = np.abs(ph_r[:m]) > np.abs(ph_v[:m])
    # last time RK4's phase was still the better one
    t_ph = tc_r[np.where(~worse)[0][-1] + 1] if (~worse).any() else tc_r[0]
    print(f"  VV h={h_vv} vs RK4 h={h_rk}: RK4 energy error exceeds VV band ({band:.1e}) at t = {t_E:.0f}; "
          f"RK4 phase error larger than VV from t = {t_ph:.0f}")
    print(f"     at t=1e4: |dE/E0| VV {band:.1e} vs RK4 {abs(dE_r[-1]):.1e};  "
          f"|dphi| VV {abs(ph_v[-1]):.2f} vs RK4 {abs(ph_r[-1]):.2f} rad")
    # mechanism check: phase drift predicted from RK4's own energy loss via T(E)
    # extra phase gained when the period is T(E(t)) instead of T_exact
    pred_phase = -2 * np.pi * np.trapezoid(1 / period_of_energy(E_r) - 1 / T_EXACT, t_r)
    print(f"     RK4 phase drift predicted from energy loss alone: {pred_phase:+.3e} rad "
          f"(measured {ph_r[-1]:+.3e})")

ax_E.set_xlim(5 * T_EXACT, T_MAX * 1.3)        # start at one period: earlier bins hold too few samples
ax_E.set(xlabel="t  (units of 1/ω₀)", ylabel=r"$|E-E_0|\,/\,E_0$  (max per time bin)",
         title=f"(a) Energy error, nonlinear, θ₀ = {THETA0:g}, β = 0")
tt = np.array([10, T_MAX])
ax_ph.loglog(tt, 1e-4 * tt / 10, **GUIDE)
ax_ph.annotate("∝ t", xy=(3e3, 1e-4 * 3e3 / 10), xytext=(4e3, 3e-3), color=INK,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.5))
tt2 = np.array([5e2, T_MAX])
ax_ph.loglog(tt2, 1.5e-2 * (tt2 / T_MAX) ** 2, **GUIDE)
ax_ph.annotate("∝ t²", xy=(2e3, 1.5e-2 * (2e3 / T_MAX) ** 2), xytext=(2.5e3, 6e-5), color=INK,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.5))
ax_ph.set_xlim(5 * T_EXACT, T_MAX * 1.3)
ax_ph.set(xlabel="t  (units of 1/ω₀)", ylabel=r"$|\Delta\varphi|$  (rad)", title="(b) Phase error vs exact period")

ax_E.legend(loc="lower right", fontsize=7, handlelength=1.5, borderaxespad=0.3)
ax_E.set_ylim(1e-7, 0.1)
fig.tight_layout()
save(fig, "fig_C_long_time")
