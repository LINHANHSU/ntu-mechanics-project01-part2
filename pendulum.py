
import numpy as np
from scipy.integrate import solve_ivp, cumulative_simpson

def F(theta, linear):
    if linear:
        return theta
    else:
        return np.sin(theta)

def accel(theta, omega, beta, linear):
    return -2 * beta * omega - F(theta, linear)

def step_vv(theta, omega, dt, beta, linear):
    alpha = accel(theta, omega, beta, linear)
    theta_new = theta + omega * dt + 0.5 * alpha * dt**2
    omega_new = (omega + 0.5 * dt * (alpha - F(theta_new, linear))) / (1 + beta * dt)
    return theta_new, omega_new

def step_rk4(theta, omega, dt, beta, linear):
    k1_theta = omega
    k1_omega = accel(theta, omega, beta, linear)

    k2_theta = omega + 0.5 * dt * k1_omega
    k2_omega = accel(theta + 0.5 * dt * k1_theta, k2_theta, beta, linear)

    k3_theta = omega + 0.5 * dt * k2_omega
    k3_omega = accel(theta + 0.5 * dt * k2_theta, k3_theta, beta, linear)

    k4_theta = omega + dt * k3_omega
    k4_omega = accel(theta + dt * k3_theta, k4_theta, beta, linear)

    theta_new = theta + (dt / 6) * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta)
    omega_new = omega + (dt / 6) * (k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega)

    return theta_new, omega_new

def integrate_rk4_with_work(theta0, omega0, dt, t_max, beta, linear):
    """RK4 on (theta, omega, W) with dW/dt = 2 beta omega^2, so the dissipated work has
    RK4's own accuracy (a separate quadrature of omega^2 would add its own error)."""
    def f(y):
        return np.array([y[1], accel(y[0], y[1], beta, linear), 2 * beta * y[1] ** 2])
    n_steps = round(t_max / dt)
    t = np.arange(n_steps + 1) * dt
    Y = np.zeros((n_steps + 1, 3))
    Y[0] = theta0, omega0, 0.0
    for i in range(n_steps):
        y = Y[i]
        k1 = f(y); k2 = f(y + 0.5 * dt * k1); k3 = f(y + 0.5 * dt * k2); k4 = f(y + dt * k3)
        Y[i + 1] = y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, Y[:, 0], Y[:, 1], Y[:, 2]

def integrate(step, theta0, omega0, dt, t_max, beta, linear):
    # round(), not int(): int(0.3 / 0.1) == 2 because of floating point
    n_steps = round(t_max / dt)
    t = np.arange(n_steps + 1) * dt          # includes t = 0 and t = t_max
    theta = np.zeros(n_steps + 1)
    omega = np.zeros(n_steps + 1)
    theta[0] = theta0
    omega[0] = omega0

    for i in range(1, n_steps + 1):
        theta[i], omega[i] = step(theta[i-1], omega[i-1], dt, beta, linear)

    return t, theta, omega

def energy(theta, omega, linear):
    if linear:
        return 0.5 * omega**2 + 0.5 * theta**2
    else:
        return 0.5 * omega**2 - np.cos(theta) + 1

def dissipated(t, omega, beta, rule="trapezoid"):
    # trapezoid is O(h^2); use rule="simpson" (O(h^4)) when checking RK4's budget
    p = 2 * beta * omega**2
    if rule == "simpson":
        return cumulative_simpson(p, x=t, initial=0)
    W = np.zeros_like(t)
    W[1:] = np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(t))
    return W

def crossing_times(t, theta, omega=None):
    """Times of upward zero crossings of theta.
    Without omega: linear interpolation (error O(h^3) per crossing).
    With omega: root of the cubic Hermite interpolant through (theta, omega) at
    both ends of the step, found by Newton from the linear guess (error O(h^5))."""
    i = np.where((theta[:-1] < 0) & (theta[1:] >= 0))[0]
    h = t[i+1] - t[i]
    s = -theta[i] / (theta[i+1] - theta[i])          # linear guess, 0 <= s <= 1
    if omega is not None:
        a0, a1 = theta[i], theta[i+1]
        b0, b1 = h * omega[i], h * omega[i+1]
        for _ in range(6):
            q = (2*s**3 - 3*s**2 + 1) * a0 + (s**3 - 2*s**2 + s) * b0 \
                + (-2*s**3 + 3*s**2) * a1 + (s**3 - s**2) * b1
            dq = (6*s**2 - 6*s) * a0 + (3*s**2 - 4*s + 1) * b0 \
                + (-6*s**2 + 6*s) * a1 + (3*s**2 - 2*s) * b1
            s = s - q / dq
    return t[i] + s * h

def period(t, theta, omega=None):
    t_cross = crossing_times(t, theta, omega)
    if len(t_cross) < 2:
        return None
    return np.mean(np.diff(t_cross))

def reference(theta0, omega0, t_max, beta, linear, dt=1e-4):
    """RK4 with a tiny step. Keep dt much finer than the smallest test dt."""
    return integrate(step_rk4, theta0, omega0, dt, t_max, beta, linear)

def reference_scipy(theta0, omega0, t_eval, beta, linear):
    """Independent check: scipy DOP853 at tight tolerances, sampled at t_eval."""
    def rhs(t, y):
        return [y[1], accel(y[0], y[1], beta, linear)]
    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), [theta0, omega0], method="DOP853",
                    t_eval=t_eval, rtol=1e-12, atol=1e-12)
    return sol.y[0], sol.y[1]

def check_reference(theta0=2.0, omega0=0.0, t_max=20.0, beta=0.05, linear=False, dt=1e-4):
    """Run once: max difference between the RK4 reference and DOP853."""
    t, theta, omega = reference(theta0, omega0, t_max, beta, linear, dt)
    theta_s, omega_s = reference_scipy(theta0, omega0, t, beta, linear)
    return np.max(np.abs(theta - theta_s)), np.max(np.abs(omega - omega_s))

if __name__ == "__main__":
    for theta0, t_max in ((2.0, 20.0), (1.0, 10.0)):        # (1.0, 10.0) is the Fig. B case
        for linear in (True, False):
            for beta in (0.0, 0.15):
                d_theta, d_omega = check_reference(theta0=theta0, t_max=t_max, beta=beta, linear=linear)
                print(f"theta0={theta0} t<={t_max:g} linear={linear!s:5} beta={beta:4}: "
                      f"max|dtheta|={d_theta:.1e}  max|domega|={d_omega:.1e}")
