import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from scipy.optimize import root_scalar

# ============================================================
# Output directory
# ============================================================
OUTDIR = "results_sle_stressspace"
os.makedirs(OUTDIR, exist_ok=True)

# ============================================================
# 1) Treloar Experimental Data (engineering/nominal stress MPa)
# ============================================================

uni_lambda = np.array([1.00, 1.10, 1.62, 2.15, 2.85, 3.20, 3.82,
                       4.65, 5.15, 5.85, 6.60, 7.10, 7.60, 7.75])
uni_stress = np.array([0.00, 0.20, 0.84, 1.25, 1.60, 1.78, 2.10,
                       2.65, 3.25, 4.35, 6.00, 7.60, 10.30, 11.20])

biax_lambda = np.array([1.000, 1.045, 1.085, 1.300, 1.600, 1.950,
                        2.350, 2.700, 3.050, 3.350, 3.650])
biax_stress = np.array([0.00, 0.17, 0.35, 0.95, 1.55, 2.15,
                        2.80, 3.45, 4.25, 5.25, 6.45])

shear_lambda = np.array([1.00, 1.20, 1.60, 2.10, 2.70, 3.30,
                         3.90, 4.50, 4.90, 5.20])
shear_stress = np.array([0.00, 0.28, 0.70, 1.20, 1.68, 2.15,
                         2.70, 3.50, 4.40, 5.25])

# ============================================================
# 2) Strain measure
# ============================================================

def eps_from_lambda(lmbda: np.ndarray) -> np.ndarray:
    return np.log(lmbda)

# ============================================================
# 3) SLE model (your base law + scale)
#    eps(tau) = (tau/E) / (1 + (beta|tau|)^alpha)^(1/alpha)
#    Use gamma = E*beta => eps_max = 1/gamma (easy constraint)
# ============================================================

def eps_sle(tau, alpha, E, gamma):
    """
    tau: stress (MPa)
    alpha > 0, E > 0, gamma > 0
    beta = gamma/E
    """
    beta = gamma / E
    tau = np.asarray(tau, dtype=float)
    return (tau / E) / (1.0 + (beta * np.abs(tau))**alpha)**(1.0/alpha)

# ============================================================
# 4) Inversion tau(eps): solve eps_sle(tau)=eps_target
#    Monotone increasing; stable with bisection.
# ============================================================

def tau_from_eps(eps_target, alpha, E, gamma, tau_hi_init=50.0):
    # eps_target must be strictly below eps_max = 1/gamma
    eps_max = 1.0 / gamma
    if eps_target <= 0.0:
        return 0.0
    if eps_target >= eps_max:
        # No finite tau exists (vertical blow-up). Return large value.
        return np.inf

    def f(tau):
        return eps_sle(tau, alpha, E, gamma) - eps_target

    lo = 0.0
    hi = tau_hi_init
    # Expand bracket until f(hi) > 0
    # (i.e., eps_sle(hi) > eps_target)
    for _ in range(60):
        if f(hi) > 0.0:
            break
        hi *= 2.0
        if hi > 1e7:
            return np.inf

    sol = root_scalar(f, bracket=[lo, hi], method="bisect", xtol=1e-10)
    return sol.root

def predict_tau_curve(lmbda_sorted, alpha, E, gamma):
    eps = np.log(lmbda_sorted)
    tau_pred = np.zeros_like(eps)
    # warm-start bracket based on increasing tau
    tau_hi = 50.0
    for i, e in enumerate(eps):
        t = tau_from_eps(float(e), alpha, E, gamma, tau_hi_init=tau_hi)
        tau_pred[i] = t
        if np.isfinite(t) and t > 0:
            tau_hi = max(tau_hi, 1.2 * t)
    return tau_pred

# ============================================================
# 5) Plot helpers
# ============================================================

def save_scatter(lam, tau, title, path):
    plt.figure(figsize=(7, 5))
    plt.plot(lam, tau, "o")
    plt.xlabel(r"Stretch $\lambda$")
    plt.ylabel("Engineering Stress (MPa)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def save_fit_plot(lam, tau_exp, alpha, E, gamma, title, path):
    # sort for monotone inversion / clean curve
    idx = np.argsort(lam)
    lam_s = lam[idx]
    tau_s = tau_exp[idx]

    lamg = np.linspace(lam_s.min(), lam_s.max(), 250)
    tau_pred = predict_tau_curve(lamg, alpha, E, gamma)

    plt.figure(figsize=(7, 5))
    plt.plot(lam_s, tau_s, "o", label="Exp")
    plt.plot(lamg, tau_pred, "-", label="SLE (stress-space fit)")
    plt.xlabel(r"Stretch $\lambda$")
    plt.ylabel("Engineering Stress (MPa)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def save_loss_curve(loss_hist, title, path):
    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(1, len(loss_hist)+1), loss_hist, "-")
    plt.xlabel("Function evaluations")
    plt.ylabel("RMSE stress (MPa)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

# ============================================================
# 6) Stress-space fitting 
#    Minimize: tau_pred(lambda; params) - tau_exp
#    Subject to eps_max = 1/gamma > max(log(lambda))+delta
# ============================================================

def fit_sle_stress_space(lam, tau_exp, fit_name, delta=0.10):
    lam = np.asarray(lam, dtype=float)
    tau_exp = np.asarray(tau_exp, dtype=float)

    eps = np.log(lam)
    eps_need = float(np.max(eps) + delta)

    # gamma must satisfy: 1/gamma > eps_need  => gamma < 1/eps_need
    gamma_ub = 0.999 / eps_need  # strict margin
    gamma_lb = 1e-6

    # Bounds (broad but safe)
    alpha_lb, alpha_ub = 1.0, 50.0
    E_lb, E_ub = 1e-3, 1e3       # MPa scale
    # gamma bounds enforced via loggamma bounds below

    loss_hist = []

    # Optimize in transformed variables for positivity:
    # x = [alpha, logE, loggamma]
    def unpack(x):
        alpha = float(x[0])
        E = float(np.exp(x[1]))
        gamma = float(np.exp(x[2]))
        return alpha, E, gamma

    # Initial guesses (reasonable)
    x0 = np.array([2.0, np.log(1.0), np.log(min(0.3, gamma_ub*0.8))], dtype=float)

    # Parameter bounds
    lb = np.array([alpha_lb, np.log(E_lb), np.log(gamma_lb)], dtype=float)
    ub = np.array([alpha_ub, np.log(E_ub), np.log(gamma_ub)], dtype=float)

    # For speed, sort by lambda once (monotone)
    idx = np.argsort(lam)
    lam_s = lam[idx]
    tau_s = tau_exp[idx]
    eps_s = np.log(lam_s)

    def residuals(x):
        alpha, E, gamma = unpack(x)

        # Predict tau at each eps by inversion
        tau_pred = np.zeros_like(tau_s)
        tau_hi = max(50.0, float(np.max(tau_s))*2.0 + 10.0)

        for i, e in enumerate(eps_s):
            t = tau_from_eps(float(e), alpha, E, gamma, tau_hi_init=tau_hi)
            tau_pred[i] = t
            if np.isfinite(t) and t > 0:
                tau_hi = max(tau_hi, 1.2*t)

        r = tau_pred - tau_s
        rmse = float(np.sqrt(np.mean(r**2)))
        loss_hist.append(rmse)
        return r

    res = least_squares(
        residuals,
        x0=x0,
        bounds=(lb, ub),
        loss="soft_l1",          # robust to small digitization noise
        f_scale=0.5
    )

    alpha, E, gamma = unpack(res.x)

   