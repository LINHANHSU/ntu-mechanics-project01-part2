"""Fig B: is each method converging at its stated order, and what does accuracy cost?

(a) Error at t_f = 10 vs h, log-log, with reference slopes 2 and 4.
(b) The same errors vs number of force evaluations F(theta):
    VV needs 1 new evaluation per step (F(theta_{n+1}) can be reused as the next
    step's alpha_n), RK4 needs 4.
Nonlinear model, theta0 = 1.0, beta = 0 and 0.15. Reference: RK4 with h = 1e-4.
"""
import numpy as np
import matplotlib.pyplot as plt
import pendulum as p
from style import COLOR, STEPS, INK, LS_BETA, FULL, GUIDE, save

THETA0 = 1.0
T_F = 10.0
EVALS_PER_STEP = {"VV": 1, "RK4": 4}
n_list = 10 * 2 ** np.arange(0, 11)          # h = 1 ... ~0.001
h_list = T_F / n_list

fig, (ax_h, ax_c) = plt.subplots(1, 2, figsize=(FULL, 2.9))

err = {}
for beta in (0.0, 0.15):
    _, th_ref, om_ref = p.reference(THETA0, 0.0, T_F, beta, False, dt=1e-4)
    for name, step in STEPS.items():
        e = []
        for h in h_list:
            _, th, om = p.integrate(step, THETA0, 0.0, h, T_F, beta, False)
            e.append(np.hypot(th[-1] - th_ref[-1], om[-1] - om_ref[-1]))
        e = np.array(e)
        err[name, beta] = e
        m = (h_list >= 0.004) & (h_list <= 0.26)          # asymptotic range, same for both methods
        slope = np.polyfit(np.log(h_list[m]), np.log(e[m]), 1)[0]
        print(f"{name:3} beta={beta:4}: slope {slope:.2f}")
        style = dict(color=COLOR[name], ls=LS_BETA[beta], marker="o", ms=3,
                     mfc="none" if beta == 0 else COLOR[name])
        ax_h.loglog(h_list, e, label=f"{name}, β={beta:g}  (slope {slope:.2f})", **style)
        ax_c.loglog(n_list * EVALS_PER_STEP[name], e, label=f"{name}, β={beta:g}", **style)

print("beta=0 errors by evaluation count:")
for name in STEPS:
    print(f"  {name:3}", "  ".join(f"{n * EVALS_PER_STEP[name]}:{x:.1e}"
                                  for n, x in zip(n_list[:5], err[name, 0.0][:5])))

hh = np.array([1e-3, 1])
ax_h.loglog(hh, 3e-1 * hh ** 2 * 4, **GUIDE)
ax_h.loglog(hh, 2e-3 * hh ** 4, **GUIDE)
ax_h.text(2e-3, 1.2 * (2e-3) ** 2 * 3, "∝ h²", color=INK)
ax_h.text(4e-2, 2e-3 * (4e-2) ** 4 / 12, "∝ h⁴", color=INK)
ax_h.set(xlabel="step size h  (units of 1/ω₀)", ylabel=f"error in (θ, ω) at t = {T_F:g}",
         title=f"(a) Error vs step size, nonlinear, θ₀ = {THETA0:g}")
ax_h.set_ylim(1e-15, 3)
ax_h.legend(loc="lower right", fontsize=7)

# equal-cost crossover: evaluation count where the VV and RK4 error curves meet (log-log interpolation)
for beta in (0.0, 0.15):
    N = np.geomspace(40, 160, 400)
    eV = np.exp(np.interp(np.log(N), np.log(n_list), np.log(err["VV", beta])))
    eR = np.exp(np.interp(np.log(N), np.log(4 * n_list), np.log(err["RK4", beta])))
    k = np.argmin(np.abs(np.log(eV / eR)))
    _, th_r, om_r = p.reference(THETA0, 0.0, T_F, beta, False, dt=1e-4)
    size = np.hypot(th_r[-1], om_r[-1])
    print(f"beta={beta}: curves cross at about {N[k]:.0f} evaluations (error {eV[k]:.1e}, "
          f"{eV[k] / size:.1e} relative to |(theta, omega)| = {size:.3f})")

# cost to reach a target error (beta = 0), by log-log interpolation
for target, lab in ((1e-3, "10⁻³"), (1e-6, "10⁻⁶")):          # beta = 0 runs
    cost = {}
    for name in STEPS:
        N = n_list * EVALS_PER_STEP[name]
        e = err[name, 0.0]
        cost[name] = np.exp(np.interp(np.log(target), np.log(e[::-1]), np.log(N[::-1])))
    print(f"evaluations for error {target:.0e} (beta=0): VV {cost['VV']:.0f}, RK4 {cost['RK4']:.0f}")
    ax_c.axhline(target, **GUIDE)                  # costs at these errors are quoted in the text

ax_c.text(0.97, 0.97, "β=0, evaluations needed (interpolated):\n10⁻³: VV ≈180, RK4 ≈100\n10⁻⁶: VV ≈5600, RK4 ≈590",
          transform=ax_c.transAxes, ha="right", va="top", fontsize=7, color=INK)
ax_c.set(xlabel=f"force evaluations to reach t = {T_F:g}", ylabel=f"error in (θ, ω) at t = {T_F:g}",
         title="(b) Error vs cost")
ax_c.legend(loc="lower left", fontsize=7)

fig.tight_layout()
save(fig, "fig_B_convergence_cost")
