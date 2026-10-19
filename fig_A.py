"""Fig A: when does the small-angle approximation fail?

(a) Phase portraits (theta, omega), linear vs nonlinear, plus one damped spiral.
(b) Period T/T0 vs theta0: VV and RK4 on top of the exact elliptic-integral curve
    (T&M Eq. 4.28), the series to O(theta0^4) (Eq. 4.29) and the linear value 1.
    Lower strip: physical deviation T/T0 - 1 vs each method's relative period error,
    and the floor of the period measurement itself.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk
import pendulum as p
from style import COLOR, STEPS, INK, THIRD, FULL, save

T0 = 2 * np.pi


def period_exact(theta0):
    # T/T0 = (2/pi) K(k), k = sin(theta0/2); scipy's ellipk takes m = k^2
    return (2 / np.pi) * ellipk(np.sin(theta0 / 2) ** 2)


def period_series(theta0):
    # T&M Eq. 4.29, to theta0^4
    return 1 + theta0**2 / 16 + 11 * theta0**4 / 3072


fig = plt.figure(figsize=(FULL, 3.6))
gs = fig.add_gridspec(2, 2, height_ratios=[2.6, 1.4], width_ratios=[1, 1.1], hspace=0.08)
ax_ph = fig.add_subplot(gs[:, 0])
ax_T = fig.add_subplot(gs[0, 1])
ax_dT = fig.add_subplot(gs[1, 1], sharex=ax_T)

# --- (a) phase portraits, RK4 with h = 0.01 (error ~1e-9, invisible here) ---
h = 0.01
for theta0 in (0.5, 1.0, 2.0, 3.0):
    _, th, om = p.integrate(p.step_rk4, theta0, 0.0, h, 1.02 * T0 * period_exact(theta0), 0.0, False)
    ax_ph.plot(th, om, color="black", lw=1.1)
    _, th, om = p.integrate(p.step_rk4, theta0, 0.0, h, 1.02 * T0, 0.0, True)
    ax_ph.plot(th, om, color="#9a9994", lw=0.8, ls=":")
    th_lab = 0.7 * theta0                                      # label on the nonlinear orbit
    w_lab = np.sqrt(2 * (np.cos(th_lab) - np.cos(theta0)))
    if theta0 < 1.5:                                           # small orbits: just outside, clear of the spiral
        th_lab, w_lab = 0.6 * theta0, 1.35 * w_lab
    if theta0 == 0.5:                                          # on its orbit, lower right
        th_lab, w_lab = 0.35, -0.36
    if theta0 == 3.0:                                          # upper left, away from the "2" label
        th_lab = -th_lab
    ax_ph.annotate(f"{theta0:g}", (th_lab, w_lab), xytext=(0, 0), textcoords="offset points",
                   fontsize=7, color=INK, ha="center", va="center",
                   bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))


_, th, om = p.integrate(p.step_rk4, 2.0, 0.0, h, 40, 0.15, False)
ax_ph.plot(th, om, color=THIRD, lw=0.9, zorder=1)

ax_ph.plot([], [], color="black", label="nonlinear, β=0")
ax_ph.plot([], [], color=INK, ls=":", label="linear, β=0")
ax_ph.plot([], [], color=THIRD, label="nonlinear, θ₀=2, β=0.15")
ax_ph.set(xlabel="θ (rad)", ylabel="ω = dθ/dt  (units of ω₀)", xlim=(-3.6, 3.6), ylim=(-3.6, 3.6),
          title="(a) Phase portraits (RK4, h = 0.01)")
ax_ph.set_aspect("equal")
ax_ph.legend(loc="upper center", ncol=2, fontsize=7, frameon=False,
             bbox_to_anchor=(0.5, -0.13), columnspacing=0.8)

# --- (b) period vs amplitude ---
th_grid = np.linspace(0.01, 3.1, 400)
ax_T.plot(th_grid, period_exact(th_grid), color=INK, lw=1.4, label="exact (2/π)K(sin(θ₀/2))")
ax_T.plot(th_grid, period_series(th_grid), color=INK, lw=0.9, ls=(0, (6, 2)), label="series to O(θ₀⁴)")
ax_T.axhline(1, color=INK, lw=0.9, ls=":", label="linear model")

theta0_list = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 2.8, 3.0, 3.1])
h = 0.05
T_num = {name: [] for name in STEPS}
rel = {name: [] for name in STEPS}
floor = []
print(f"Period T/T0, h = {h}, zero crossings by cubic Hermite interpolation")
print("theta0   exact       VV          RK4        relerr VV  relerr RK4  floor")
for theta0 in theta0_list:
    T_exact = period_exact(theta0)
    t_run = 6.2 * T0 * T_exact                    # ~6 periods, averaged
    for name, step in STEPS.items():
        t, th, om = p.integrate(step, theta0, 0.0, h, t_run, 0.0, False)
        T_num[name].append(p.period(t, th, om) / T0)
        rel[name].append(abs(T_num[name][-1] - T_exact) / T_exact)
    # measurement floor: the same period extraction applied to the reference solution
    t, th, om = p.reference(theta0, 0.0, t_run, 0.0, False, dt=1e-3)
    floor.append(abs(p.period(t[::50], th[::50], om[::50]) / T0 - T_exact) / T_exact)
    print(f"{theta0:5.2f}  {T_exact:.7f}  {T_num['VV'][-1]:.7f}  {T_num['RK4'][-1]:.7f}"
          f"  {rel['VV'][-1]:.1e}    {rel['RK4'][-1]:.1e}    {floor[-1]:.1e}")
# h-scaling: repeat at h = 0.1 and report error ratios err(0.1)/err(0.05) (expect 4 and 16)
print("Error ratio err(h=0.1)/err(h=0.05):")
for i, theta0 in enumerate(theta0_list):
    T_exact = period_exact(theta0)
    ratios = []
    for name, step in STEPS.items():
        t, th, om = p.integrate(step, theta0, 0.0, 0.1, 6.2 * T0 * T_exact, 0.0, False)
        ratios.append(abs(p.period(t, th, om) / T0 - T_exact) / T_exact / rel[name][i])
    print(f"  theta0={theta0:4}: VV {ratios[0]:.2f}  RK4 {ratios[1]:.2f}")
from scipy.optimize import brentq
for frac in (1e-3, 1e-2, 5e-2):
    a = brentq(lambda x: period_exact(x) - 1 - frac, 1e-3, 3.0)
    print(f"T/T0 - 1 = {frac:g} at theta0 = {a:.3f} rad = {np.degrees(a):.0f} deg")
print(f"Series error at theta0=2: {1 - period_series(2.0) / period_exact(2.0):.2%} of T")
a = brentq(lambda x: period_exact(x) - 1 / np.sqrt(1 - 0.15**2), 1e-3, 3.0)
print(f"Nonlinear period shift equals the damping shift 1/omega1 - 1 (beta=0.15) at theta0 = {a:.2f}")
print(f"Small-amplitude theory: VV h^2/24 = {h**2 / 24:.2e}, RK4 h^4/120 = {h**4 / 120:.2e}")

# Near the separatrix: RK4's period error predicted from its own mean energy loss via dlnT/dE
def T_of_E(E):
    return 4 * ellipk(np.sin(np.arccos(1 - E) / 2) ** 2)
for theta0 in (3.0, 3.1):
    t, th, om = p.integrate(p.step_rk4, theta0, 0.0, h, 6.2 * T0 * period_exact(theta0), 0.0, False)
    E = p.energy(th, om, False)
    dlnT_dE = (np.log(T_of_E(E[0] + 1e-7)) - np.log(T_of_E(E[0] - 1e-7))) / 2e-7
    measured = p.period(t, th, om) / T0 / period_exact(theta0) - 1
    print(f"RK4 theta0={theta0}: predicted (dlnT/dE)*<dE> = {dlnT_dE * np.mean(E - E[0]):+.2e}, measured {measured:+.2e}")

ax_T.plot(theta0_list, T_num["VV"], "o", color=COLOR["VV"], mfc="none", ms=6, mew=1.1,
          label=f"VV, h={h}")
ax_T.plot(theta0_list, T_num["RK4"], "x", color=COLOR["RK4"], ms=4.5, mew=1.2,
          label=f"RK4, h={h}")
ax_T.set(ylabel="T / T₀", ylim=(0.9, 3.5), title="(b) Period vs amplitude, β = 0")
ax_T.legend(loc="upper left", fontsize=7)
plt.setp(ax_T.get_xticklabels(), visible=False)

ax_dT.semilogy(th_grid, period_exact(th_grid) - 1, color=INK, lw=1.4)
for name, mk in (("VV", "o"), ("RK4", "x")):
    ax_dT.semilogy(theta0_list, rel[name], mk, color=COLOR[name], ms=3.5,
                   mfc="none" if name == "VV" else COLOR[name])        # markers only: errors change sign
ax_dT.semilogy(theta0_list, floor, "s", color=INK, ms=2.5)
ax_dT.text(0.05, 0.2, "physical T/T₀ − 1", color=INK, fontsize=7)
ax_dT.text(1.45, 3e-4, "VV error", color=COLOR["VV"], fontsize=7)
ax_dT.text(1.45, 2e-7, "RK4 error", color=COLOR["RK4"], fontsize=7)
ax_dT.text(0.05, 4e-11, "measurement floor", color=INK, fontsize=7)
ax_dT.set(xlabel="θ₀ (rad)", ylabel="relative\ndeviation", ylim=(1e-14, 10))
ax_dT.set_yticks([1e-12, 1e-8, 1e-4, 1])
ax_dT.yaxis.set_label_coords(-0.15, 0.5)
ax_T.yaxis.set_label_coords(-0.15, 0.5)

fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.17, wspace=0.28)
save(fig, "fig_A_small_angle")

# Damped periods (quoted in the text): successive upward-crossing intervals, beta = 0.15
beta = 0.15
print(f"\nDamped periods / T0 (RK4, h = 0.01, beta = {beta}); exact linear 1/omega1 = {1 / np.sqrt(1 - beta**2):.6f}")
for linear, theta0 in ((True, 0.5), (False, 2.0)):
    t, th, om = p.integrate(p.step_rk4, theta0, 0.0, 0.01, 60, beta, linear)
    print(f"  {'linear' if linear else 'nonlinear'}, theta0 = {theta0}:",
          " ".join(f"{x:.5f}" for x in np.diff(p.crossing_times(t, th, om)) / T0))
