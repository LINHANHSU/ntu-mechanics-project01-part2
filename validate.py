"""Step 2: validate VV and RK4 against exact solutions before running experiments.

(a) Error |theta_num - theta_exact| vs t, linear model, undamped and underdamped.
(b) Convergence: error at t_f = 10 vs h, linear damped model against the exact
    solution, with reference slopes 2 and 4. (Nonlinear convergence is Fig. B.)
"""
import numpy as np
import matplotlib.pyplot as plt
import pendulum as p
from style import COLOR, STEPS, INK, LS_BETA, FULL, GUIDE, save

THETA0 = 0.5


def exact(t, theta0, beta):
    """Linear model from rest, underdamped (T&M Eq. 3.38 with w0 = 1): theta, omega."""
    w1 = np.sqrt(1 - beta**2)
    th = theta0 * np.exp(-beta * t) * (np.cos(w1 * t) + (beta / w1) * np.sin(w1 * t))
    om = -theta0 * np.exp(-beta * t) * np.sin(w1 * t) / w1
    return th, om


fig, (ax_t, ax_h) = plt.subplots(1, 2, figsize=(FULL, 2.7))

# --- (a) error vs analytic over time, h = 0.1 ---
h = 0.1
T_long = 40.0
print(f"Max |theta - exact| over t <= {T_long}, h = {h}, theta0 = {THETA0}")
for beta in (0.0, 0.15):
    for name, step in STEPS.items():
        t, th, _ = p.integrate(step, THETA0, 0.0, h, T_long, beta, True)
        err = np.abs(th - exact(t, THETA0, beta)[0])
        print(f"  beta={beta:4}  {name:3}  {np.max(err):.2e}")
        ax_t.semilogy(t[1:], err[1:], color=COLOR[name], ls=LS_BETA[beta], lw=0.9,
                      label=f"{name}, β={beta:g}")
ax_t.set(xlabel="t  (units of 1/ω₀)", ylabel=r"$|\theta-\theta_\mathrm{exact}|$  (rad)", ylim=(1e-9, 1),
         title=f"(a) Error vs exact solution, linear, h = {h}")
ax_t.legend(ncol=2, loc="upper center", fontsize=7, columnspacing=0.8)

# --- (b) convergence at t_f = 10 against the exact solution ---
t_f = 10.0
beta_c = 0.15
n_list = 20 * 2 ** np.arange(0, 10)          # h = 0.5 ... ~0.001, t_f/h is an integer
h_list = t_f / n_list
th_ex, om_ex = exact(t_f, THETA0, beta_c)
print(f"Convergence at t_f={t_f}, theta0={THETA0}, beta={beta_c}, linear vs exact")
for name, step in STEPS.items():
    err = []
    for hh in h_list:
        _, th, om = p.integrate(step, THETA0, 0.0, hh, t_f, beta_c, True)
        err.append(np.hypot(th[-1] - th_ex, om[-1] - om_ex))
    err = np.array(err)
    m = (h_list >= 0.004) & (h_list <= 0.26)          # asymptotic range, same for both methods
    s = np.polyfit(np.log(h_list[m]), np.log(err[m]), 1)[0]
    print(f"  {name:3} slope = {s:.2f}")
    ax_h.loglog(h_list, err, "o", ls=LS_BETA[beta_c], color=COLOR[name], ms=3.5, label=f"{name}: slope {s:.2f}")

hh = np.array([2e-3, 0.5])
ax_h.loglog(hh, 1e-1 * (hh / 0.5) ** 2, **GUIDE)
ax_h.loglog(hh, 1e-4 * (hh / 0.5) ** 4, **GUIDE)
ax_h.text(3e-3, 1e-1 * (3e-3 / 0.5) ** 2 * 3, "∝ h²", color=INK)
ax_h.text(3e-2, 1e-4 * (3e-2 / 0.5) ** 4 / 12, "∝ h⁴", color=INK)
ax_h.set(xlabel="h  (units of 1/ω₀)", ylabel=f"error in (θ, ω) at t = {t_f:g}",
         title=f"(b) Convergence, linear, β = {beta_c}")
ax_h.legend(loc="lower right")

fig.tight_layout()
save(fig, "fig_validation")

# Reference accuracy: RK4 at the reference step h = 1e-4 against the exact linear solution
t, th, _ = p.integrate(p.step_rk4, THETA0, 0.0, 1e-4, t_f, beta_c, True)
print(f"RK4 h=1e-4 vs exact linear solution at t={t_f:g}: {abs(th[-1] - exact(t_f, THETA0, beta_c)[0]):.1e}")
