"""Side check (tabulated in the report): does VV's long-time advantage survive damping?

Equal-cost pair at every beta: VV at h = 0.05 and RK4 at h = 0.2 (20 force evaluations
per unit time). Nonlinear model, theta0 = 1. Each run lasts t_f = min(3000, 3/beta),
i.e. until E has decayed to about e^-6 E0 (or t = 3000 without damping).
Reference: RK4 with h = 0.0125. Reported: the maximum energy-budget residual
|E + W_diss - E0| / E0 over the whole run (VV's error oscillates, so a single snapshot
would mislead) and the final
error in (theta, omega) relative to |(theta, omega)_ref(t_f)|. Also printed: the energy decay time 1/(2 beta), to compare with
the undamped energy crossover time t_cross ~ 180 for this pair (Fig. C).
"""
import numpy as np
import pendulum as p

THETA0, H_REF = 1.0, 0.0125
PAIR = (("VV", p.step_vv, 0.05), ("RK4", p.step_rk4, 0.2))

print("beta     t_f    1/(2beta)   max budget resid/E0 VV / RK4    final rel state err VV / RK4")
for beta in (0.0, 0.0005, 0.005, 0.05, 0.15):
    t_f = 3000.0 if beta == 0 else min(3000.0, round(3 / beta / 0.2) * 0.2)
    _, th_ref, om_ref = p.integrate(p.step_rk4, THETA0, 0.0, H_REF, t_f, beta, False)
    E_ref = p.energy(th_ref, om_ref, False)
    out, detail = [], []
    for name, step, h in PAIR:
        if name == "RK4":       # W_diss integrated inside RK4, so the residual is RK4's own error
            t, th, om, W = p.integrate_rk4_with_work(THETA0, 0.0, h, t_f, beta, False)
        else:                   # VV: Simpson quadrature, far more accurate than VV itself
            t, th, om = p.integrate(step, THETA0, 0.0, h, t_f, beta, False)
            W = p.dissipated(t, om, beta, rule="simpson")
        k = round(h / H_REF)
        # energy error = budget residual |E + W_diss - E0| / E0 (at beta = 0 simply |E - E0| / E0);
        # comparing E with the reference at the same t would mostly measure timing error
        E = p.energy(th, om, False)
        resid = np.abs(E + W - E[0]) / E[0]
        signed = (E + W - E[0]) / E[0]
        detail.append(f"{name} max at t={t[np.argmax(resid)]:.1f}, signed residual at t_f {signed[-1]:+.1e}")
        size = np.hypot(th_ref[-1], om_ref[-1])           # state error relative to the solution's size
        out.append((np.max(resid), np.hypot(th[-1] - th_ref[-1], om[-1] - om_ref[-1]) / size))
    tau = "inf" if beta == 0 else f"{1 / (2 * beta):.0f}"
    print(f"{beta:<7}  {t_f:<6g} {tau:>9}   {out[0][0]:.1e} / {out[1][0]:.1e}         "
          f"{out[0][1]:.1e} / {out[1][1]:.1e}")
    print("         " + ";  ".join(detail))
    if beta == 0:
        drift_rate = out[1][0] / t_f                 # RK4's undamped residual growth per unit time
    else:
        pred = drift_rate / (2 * beta) * (1 - np.exp(-2 * beta * t_f))   # drift accumulated while E lasts
        print(f"         RK4 drift-model prediction rate/(2 beta)*(1 - e^(-2 beta t_f)) = {pred:.1e}")

# Why weak damping removes VV's phase advantage (beta = 0.0005): timing error of upward zero
# crossings against the reference, in phase units, at t ~ 1500 and t ~ 3000.
# VV: amplitude-independent period error (-h^2/24 relative) -> grows ~ t.
# RK4: driven by its energy leak, which shrinks with E ~ e^{-2 beta t} -> saturates.
beta = 0.0005
t_r, th_r, om_r = p.integrate(p.step_rk4, THETA0, 0.0, H_REF, 3000.0, beta, False)
tc_ref = p.crossing_times(t_r, th_r, om_r)
T_mean = np.mean(np.diff(tc_ref))
print(f"\nbeta={beta}: phase (timing) error 2*pi*(t_n - t_n_ref)/T at t~1500 and t~3000")
for name, step, h in PAIR:
    t, th, om = p.integrate(step, THETA0, 0.0, h, 3000.0, beta, False)
    tc = p.crossing_times(t, th, om)
    m = min(len(tc), len(tc_ref))
    d = 2 * np.pi * (tc[:m] - tc_ref[:m]) / T_mean
    i_half = np.argmin(np.abs(tc_ref[:m] - 1500))
    extra = ""
    if name == "VV":
        extra = f"  (prediction -2*pi*(h^2/24)*N_periods = {-2 * np.pi * h**2 / 24 * (m - 1):+.3f})"
    print(f"  {name:3}: {d[i_half]:+.3f} at t={tc_ref[i_half]:.0f}, {d[m - 1]:+.3f} at t={tc_ref[m - 1]:.0f}, "
          f"ratio {d[m - 1] / d[i_half]:.2f}{extra}")

# RK4's damped budget residual at t_f for three step sizes: for beta >= 0.05 it is not a
# clean drift (two error terms of opposite sign compete), so the tabulated values are small
# partly by cancellation.
print("\nRK4 signed budget residual / E0 at t_f for h = 0.4, 0.2, 0.1:")
for beta in (0.005, 0.05, 0.15):
    t_f = min(3000.0, round(3 / beta / 0.4) * 0.4)
    vals = []
    for h in (0.4, 0.2, 0.1):
        t, th, om, W = p.integrate_rk4_with_work(THETA0, 0.0, h, t_f, beta, False)
        E = p.energy(th, om, False)
        vals.append((E[-1] + W[-1] - E[0]) / E[0])
    print(f"  beta={beta}: " + "  ".join(f"{v:+.1e}" for v in vals))
