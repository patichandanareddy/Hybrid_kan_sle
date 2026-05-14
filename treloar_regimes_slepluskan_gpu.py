#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kan import KAN
from scipy.optimize import root_scalar

# ── device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

OUT_ROOT = "treloar_regimes_sle_kan"
os.makedirs(OUT_ROOT, exist_ok=True)

# ── Treloar experimental data ────────────────────────────────────────────────
DATA = {
    "uniaxial": {
        "lambda": np.array([1.00, 1.10, 1.62, 2.15, 2.85, 3.20, 3.82,
                             4.65, 5.15, 5.85, 6.60, 7.10, 7.60, 7.75]),
        "stress": np.array([0.00, 0.20, 0.84, 1.25, 1.60, 1.78, 2.10,
                             2.65, 3.25, 4.35, 6.00, 7.60, 10.30, 11.20]),
    },
    "biaxial": {
        "lambda": np.array([1.000, 1.045, 1.085, 1.300, 1.600, 1.950,
                             2.350, 2.700, 3.050, 3.350, 3.650]),
        "stress": np.array([0.00, 0.17, 0.35, 0.95, 1.55, 2.15,
                             2.80, 3.45, 4.25, 5.25, 6.45]),
    },
    "planar": {
        "lambda": np.array([1.00, 1.20, 1.60, 2.10, 2.70, 3.30,
                             3.90, 4.50, 4.90, 5.20]),
        "stress": np.array([0.00, 0.28, 0.70, 1.20, 1.68, 2.15,
                             2.70, 3.50, 4.40, 5.25]),
    },
}

# Calibrated SLE parameters (from sle_treloar_fit_stressspace.py, ln(lambda) basis)
LEARNED = {
    "uniaxial": {"alpha": 1.53836245, "E": 1.05854390},
    "biaxial":  {"alpha": 3.16771406, "E": 3.12024076},
    "planar":   {"alpha": 2.47956314, "E": 1.49049065},
}

REGIMES = {
    "moderate": {"gamma": 0.50},   # eps_max=2.00, lambda_limit=7.39
    "strong":   {"gamma": 0.80},   # eps_max=1.25, lambda_limit=3.49
}


# ── SLE constitutive law: eps(tau) ───────────────────────────────────────────
def eps_sle(tau, alpha, beta, E):
    return (tau / E) / (1.0 + (beta * abs(tau)) ** alpha) ** (1.0 / alpha)


# ── Robust inversion: eps -> tau ─────────────────────────────────────────────
def tau_from_eps(eps_target, alpha, beta, E, tau_max=80.0):
    eps_max = 1.0 / (E * beta)
    eps_eff = np.sign(eps_target) * min(abs(eps_target), 0.999 * eps_max)
    if abs(eps_eff) < 1e-10:
        return 0.0
    def f(tau):
        return eps_sle(tau, alpha, beta, E) - eps_eff
    fa, fb = f(0.0), f(tau_max)
    if fa * fb > 0:
        return tau_max
    return root_scalar(f, bracket=[0.0, tau_max], method="bisect").root


# ── Forward pass: lambda -> stress using ln(lambda) ─────────────────────────
def stress_from_lambda(lam, alpha, beta, E):
    eps = np.log(lam)   # logarithmic strain -- correct for SLE
    return np.array([tau_from_eps(e, alpha, beta, E) for e in eps])


# ── Admissibility mask ───────────────────────────────────────────────────────
def admissible_mask(lam, gamma, safety=0.999):
    return np.log(lam) < safety / gamma


# ── Stress-space metrics ─────────────────────────────────────────────────────
def stress_metrics(y, yhat):
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    mae  = float(np.mean(np.abs(y - yhat)))
    sst  = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    r2   = float(1.0 - np.sum((y - yhat) ** 2) / sst)
    return {"RMSE_MPa": rmse, "MAE_MPa": mae, "R2": r2}


# ── Plot style ───────────────────────────────────────────────────────────────
def _style():
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "axes.labelsize":    13,
        "xtick.labelsize":   11,
        "ytick.labelsize":   11,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "legend.fontsize":   10,
        "legend.frameon":    True,
        "legend.edgecolor":  "black",
        "legend.fancybox":   False,
        "legend.framealpha": 1.0,
        "figure.facecolor":  "white",
        "savefig.facecolor": "white",
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
    })

C_SLE    = "#1a6fad"
C_HYBRID = "#b22222"
C_EXP    = "#000000"


# ── Train one mode + regime ──────────────────────────────────────────────────
def train_case(mode, regime_name, gamma):
    lam     = DATA[mode]["lambda"]
    tau_exp = DATA[mode]["stress"]
    alpha   = LEARNED[mode]["alpha"]
    E       = LEARNED[mode]["E"]
    beta    = gamma / E
    eps_max = 1.0 / gamma
    lam_limit = np.exp(eps_max)

    tau_sle = stress_from_lambda(lam, alpha, beta, E)

    mask  = admissible_mask(lam, gamma)
    n_adm = mask.sum()
    n_excl = (~mask).sum()

    lam_train = lam[mask]
    tau_train = tau_exp[mask]
    resid     = tau_train - tau_sle[mask]

    print(f"\n[{mode} | {regime_name}]  gamma={gamma}  "
          f"eps_max={eps_max:.2f}  lambda_limit={lam_limit:.3f}")
    print(f"  admissible={n_adm}/{len(lam)}  excluded={n_excl}")
    if n_excl > 0:
        print(f"  excluded lambda values: {lam[~mask]}")

    X = torch.tensor(np.log(lam_train)[:, None],
                     dtype=torch.float32, device=DEVICE)
    Y = torch.tensor(resid[:, None],
                     dtype=torch.float32, device=DEVICE)

    model = KAN(width=[1, 8, 1], grid=10, k=3, seed=0).to(DEVICE)
    for layer in model.act_fun:
        if hasattr(layer, "grid"):
            layer.grid = layer.grid.to(DEVICE)

    opt     = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()
    losses  = []
    best    = {"val": 1e20, "state": None}

    for ep in range(1, 4001):
        opt.zero_grad()
        loss = loss_fn(model(X), Y)
        loss.backward()
        opt.step()
        lv = loss.item()
        losses.append(lv)
        if lv < best["val"]:
            best["val"]   = lv
            best["state"] = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}
        if ep % 500 == 0:
            print(f"  epoch {ep:4d}  loss={lv:.3e}")

    model.load_state_dict(best["state"])
    model.to(DEVICE)
    for layer in model.act_fun:
        if hasattr(layer, "grid"):
            layer.grid = layer.grid.to(DEVICE)

    with torch.no_grad():
        kan_corr = model(X).cpu().numpy().flatten()
    tau_pred = tau_sle[mask] + kan_corr

    metrics_out = {
        "mode": mode, "regime": regime_name,
        "alpha": alpha, "E": E, "gamma": gamma, "beta": beta,
        "eps_max": eps_max, "lambda_limit": lam_limit,
        "n_admissible": int(n_adm), "n_excluded": int(n_excl),
        "SLE":          stress_metrics(tau_train, tau_sle[mask]),
        "SLE_plus_KAN": stress_metrics(tau_train, tau_pred),
    }

    out_dir = f"{OUT_ROOT}/{mode}/{regime_name}"
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    np.savetxt(f"{out_dir}/loss.txt", np.array(losses))
    np.savetxt(
        f"{out_dir}/pred.txt",
        np.column_stack([lam_train, tau_train, tau_sle[mask], tau_pred]),
        header="lambda  tau_exp  tau_sle  tau_sle_plus_kan",
        comments="",
    )

    # Filenames matching LaTeX draft exactly
    ss_fname   = (f"stress_stretch__{mode}_{regime_name}.png"
                  if mode == "uniaxial"
                  else f"stress_stretch_{mode}_{regime_name}.png")
    loss_fname = f"loss_{mode}_{regime_name}.png"

    # ── Stress-stretch plot ──────────────────────────────────────────────────
    _style()
    fig, ax = plt.subplots(figsize=(6.0, 5.0))

    ax.scatter(lam[mask], tau_exp[mask],
               s=55, color=C_EXP, zorder=5, label="Experiment")
    if n_excl > 0:
        ax.scatter(lam[~mask], tau_exp[~mask],
                   s=40, color="#888888", marker="x", zorder=4,
                   label=r"Excluded ($\ln\lambda\geq\varepsilon_{\max}$)")

    ax.plot(lam_train, tau_sle[mask],
            color=C_SLE, ls="--", dashes=(8, 5), lw=2.2, zorder=3,
            label=r"SLE backbone ($\gamma=" + f"{gamma}$)")

    ax.plot(lam_train, tau_pred,
            color=C_HYBRID, ls="-", lw=2.5, alpha=1.0, zorder=4,
            label="SLE + KAN")

    if n_excl > 0:
        ax.axvline(lam_limit, color="#888888", ls=":", lw=1.2, zorder=2,
                   label=r"$\varepsilon_{\max}$ boundary"
                   + f"  ($\\lambda={lam_limit:.2f}$)")

    ax.set_xlabel(r"Stretch ratio $\lambda$")
    ax.set_ylabel(r"Engineering stress $P$ (MPa)")

    lam_pad  = 0.04 * (lam_train.max() - lam_train.min())
    stre_pad = 0.06 * tau_train.max()
    ax.set_xlim(lam_train.min() - lam_pad, lam_train.max() + lam_pad)
    ax.set_ylim(-stre_pad, tau_train.max() + stre_pad)

    ax.grid(True, ls=":", lw=0.7, color="#cccccc")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    leg = ax.legend(loc="lower right", fontsize=9,
                    title=f"{mode.capitalize()}  |  {regime_name} regime",
                    title_fontsize=8)
    leg.get_frame().set_linewidth(1.0)
    plt.tight_layout()
    fig.savefig(ss_fname)
    fig.savefig(f"{out_dir}/stress_stretch.png")
    fig.savefig(f"{out_dir}/stress_stretch.pdf")
    plt.close(fig)

    # ── Loss plot ────────────────────────────────────────────────────────────
    _style()
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.plot(np.arange(1, len(losses) + 1), losses, color=C_EXP, lw=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"KAN training loss  --  "
                 f"{mode.capitalize()}, {regime_name} regime",
                 fontsize=11, pad=8)
    ax.grid(True, which="both", ls=":", lw=0.7, color="#cccccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(loss_fname)
    fig.savefig(f"{out_dir}/loss.png")
    fig.savefig(f"{out_dir}/loss.pdf")
    plt.close(fig)

    m = metrics_out
    print(f"  SLE     RMSE={m['SLE']['RMSE_MPa']:.4f} MPa  "
          f"R2={m['SLE']['R2']:.4f}")
    print(f"  SLE+KAN RMSE={m['SLE_plus_KAN']['RMSE_MPa']:.4f} MPa  "
          f"R2={m['SLE_plus_KAN']['R2']:.4f}")
    print(f"  Saved -> {out_dir}/")
    return metrics_out


    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"{'Mode':8} {'Regime':8} {'n_adm':6} "
          f"{'SLE RMSE':10} {'KAN RMSE':10}")
    print("-"*50)
    for k, m in all_metrics.items():
        print(f"{m['mode']:8} {m['regime']:8} "
              f"{m['n_admissible']:6} "
              f"{m['SLE']['RMSE_MPa']:10.4f} "
              f"{m['SLE_plus_KAN']['RMSE_MPa']:10.4f}")

    with open(f"{OUT_ROOT}/all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("\nDONE.")
