"""Fig D: with damping, is the energy loss physical or numerical?

(a) E(t) for both methods on a log-y axis, with the linear-model envelope E0 exp(-2 beta t).
(b) Max budget residual |E + W_diss - E0| over 0 <= t <= 40 vs h: it shrinks like
    h^2 (VV) and h^4 (RK4), so it is numerical. Also shown: with the trapezoid rule
    for W_diss, RK4's residual is capped at h^2 by the quadrature, not the integrator.
Nonlinear model, theta0 = 2.0.
"""
import numpy as np
import matplotlib.pyplot as plt
import pendulum as p
from style import COLOR, STEPS, INK, LS_BETA, FULL, GUIDE, save

THETA0 = 2.0
E0 = p.energy(THETA0, 0.0, False)
from scipy.special import ellipk
ellipk_period = ellipk(np.sin(THETA0 / 2) ** 2)          # quarter period; 4x = one period

fig, (ax_E, ax_h) = plt.subplots(1, 2, figsize=(FULL, 2.9))

# --- (a) energy decay ---
h = 0.1
T_a = 60.0
t_env = np.linspace(0, T_a, 300)
for beta in (0.05, 0.15):
    for name, step in STEPS.items():
        t, th, om = p.integrate(step, THETA0, 0.0, h, T_a, beta, False)
        E = p.energy(th, om, False)
        ratio = E / (E0 * np.exp(-2 * beta * t))
        print(f"{name:3} beta={beta}: E(t={T_a:g})/E0 = {E[-1] / E0:.1e}; "
              f"E / linear envelope: min {ratio[1:].min():.2f}, max {ratio.max():.2f}")
        ax_E.semilogy(t, E, color=COLOR[name], ls=LS_BETA[beta],
                      lw=1.5 if name == "VV" else 0.8, label=f"{name}, β={beta:g}")
    ax_E.semilogy(t_env, E0 * np.exp(-2 * beta * t_env), color=INK, ls=":", lw=0.9)
    # envelope lines are identified in the caption (labels would collide with the curves)
ax_E.set(xlabel="t  (units of 1/ω₀)", ylabel="E  (units of m l² ω₀²)", ylim=(1e-9, 5),
         title=f"(a) Energy decay, nonlinear, θ₀ = {THETA0:g}, h = {h}")
ax_E.legend(loc="lower left", fontsize=7, ncol=2)

# nonlinear effect on the decay rate: fitted mean rate of ln E over t <= 20, in units of 2 beta
for beta in (0.05, 0.15):
    rates = []
    for linear in (True, False):
        t, th, om = p.integrate(p.step_rk4, THETA0, 0.0, 0.01, 20.0, beta, linear)
        rates.append(-np.polyfit(t, np.log(p.energy(th, om, linear)), 1)[0] / (2 * beta))
    print(f"beta={beta}: mean decay rate over t<=20 / (2 beta): linear {rates[0]:.3f}, nonlinear {rates[1]:.3f}")

t, th, om = p.integrate(p.step_rk4, THETA0, 0.0, 1e-3, 4 * ellipk_period, 0.0, False)
print(f"<omega^2>/E at theta0={THETA0} (undamped, one period): {np.mean(om[:-1]**2) / E0:.3f}")

# --- (b) max budget residual / E0 vs h ---
# VV: W_diss by Simpson's rule (its quadrature error is far below VV's own error).
# RK4: W_diss integrated inside RK4 (dW/dt = 2 beta omega^2) gives RK4's own residual;
#      Simpson and trapezoid quadratures are shown too, to show how a quadrature can mask it.
beta = 0.15
T_b = 40.0
h_list = 10.0 / (10 * 2 ** np.arange(0, 7))     # h = 1 ... ~0.016, T_b/h integer
print(f"Max budget residual / E0 on 0 <= t <= {T_b}, theta0 = {THETA0}, beta = {beta}")
curves = {}
for hh in h_list:
    t, th, om = p.integrate(p.step_vv, THETA0, 0.0, hh, T_b, beta, False)
    E = p.energy(th, om, False)
    curves.setdefault("VV, Simpson", []).append(np.max(np.abs(E + p.dissipated(t, om, beta, "simpson") - E0)) / E0)
    t, th, om, W = p.integrate_rk4_with_work(THETA0, 0.0, hh, T_b, beta, False)
    curves.setdefault("RK4, W integrated", []).append(np.max(np.abs(p.energy(th, om, False) + W - E0)) / E0)
    t, th, om = p.integrate(p.step_rk4, THETA0, 0.0, hh, T_b, beta, False)
    E = p.energy(th, om, False)
    for rule in ("simpson", "trapezoid"):
        curves.setdefault(f"RK4, {rule.capitalize() if rule == 'simpson' else rule}", []).append(
            np.max(np.abs(E + p.dissipated(t, om, beta, rule) - E0)) / E0)
styles = {"VV, Simpson": dict(color=COLOR["VV"], marker="o", lw=1.1),
          "RK4, W integrated": dict(color=COLOR["RK4"], marker="o", lw=1.1),
          "RK4, Simpson": dict(color=COLOR["RK4"], marker="s", lw=0.6, mfc="none", ms=4.5),
          "RK4, trapezoid": dict(color=COLOR["RK4"], marker="^", lw=0.6, mfc="none", ms=5.5)}
m = h_list <= 0.26
for lab, res in curves.items():
    res = np.array(res)
    slope = np.polyfit(np.log(h_list[m]), np.log(res[m]), 1)[0]
    print(f"  {lab:18}: slope {slope:.2f}   " + "  ".join(f"{x:.1e}" for x in res))
    st = dict(ms=3.5); st.update(styles[lab])
    ax_h.loglog(h_list, res, ls=LS_BETA[beta], label=f"{lab} (slope {slope:.2f})", **st)
print("Equal cost (RK4 at 4h, W integrated) vs VV at h:")
for k in range(len(h_list) - 2):
    print(f"  h_RK4={h_list[k]:.3f}: RK4 {curves['RK4, W integrated'][k]:.1e}  VV {curves['VV, Simpson'][k + 2]:.1e}")
hh = np.array([1.5e-2, 1])
ax_h.loglog(hh, 1e-1 * hh ** 2, **GUIDE)
ax_h.loglog(hh, 3e-3 * hh ** 4, **GUIDE)
ax_h.annotate("∝ h²", xy=(0.04, 1e-1 * 0.04 ** 2), xytext=(0.02, 4e-3), color=INK,
             arrowprops=dict(arrowstyle="-", color=INK, lw=0.5))
ax_h.text(1.7e-2, 3e-3 * 1.7e-2 ** 4 / 15, "∝ h⁴", color=INK)
ax_h.set(xlabel="h  (units of 1/ω₀)", ylabel=rf"max $|E+W_\mathrm{{diss}}-E_0|\,/\,E_0$, $t\leq{T_b:g}$",
         title=f"(b) Energy-budget residual vs h, β = {beta}")
ax_h.legend(loc="lower right", fontsize=7)
ax_h.set_ylim(1e-13, 1)

fig.tight_layout()
save(fig, "fig_D_damping_budget")
