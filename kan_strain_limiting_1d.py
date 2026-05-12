# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------------------------------
# Ground truth strain-limiting law
# -------------------------------------------------
def strain_limiting_eps(tau, beta, alpha):
    return tau / np.power(
        1.0 + np.power(beta * np.abs(tau), alpha),
        1.0 / alpha
    )


# -------------------------------------------------
# Piecewise-linear spline (KAN edge)
# -------------------------------------------------
class PiecewiseLinearSpline1D(nn.Module):
    def __init__(self, n_knots, xmin, xmax):
        super().__init__()
        self.n_knots = n_knots
        knots = torch.linspace(xmin, xmax, n_knots)
        self.register_buffer("knots", knots)
        self.coeffs = nn.Parameter(torch.zeros(n_knots))

    def forward(self, x):
        x = x.view(-1)
        x = torch.clamp(x, self.knots[0], self.knots[-1])
        idx = torch.bucketize(x, self.knots) - 1
        idx = torch.clamp(idx, 0, self.n_knots - 2)
        x0 = self.knots[idx]
        x1 = self.knots[idx + 1]
        t = (x - x0) / (x1 - x0 + 1e-12)
        c0 = self.coeffs[idx]
        c1 = self.coeffs[idx + 1]
        return (1.0 - t) * c0 + t * c1

    def slopes(self):
        return (self.coeffs[1:] - self.coeffs[:-1]) / (
            self.knots[1:] - self.knots[:-1]
        )


# -------------------------------------------------
# Odd-symmetric KAN
# -------------------------------------------------
class OddSymmetricKAN(nn.Module):
    def __init__(self, n_knots, tau_max):
        super().__init__()
        self.g = PiecewiseLinearSpline1D(n_knots, 0.0, tau_max)

    def forward(self, tau):
        tau = tau.view(-1)
        sign = torch.sign(tau)
        sign = torch.where(tau == 0, torch.zeros_like(sign), sign)
        return sign * self.g(torch.abs(tau))


# -------------------------------------------------
# R^2 metric
# -------------------------------------------------
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    return 1.0 - ss_res / ss_tot


# -------------------------------------------------
# Main training routine
# -------------------------------------------------
def main():

    # -------- Parameters --------
    beta = float(os.environ.get("BETA", 1.0))
    alpha = 2.0
    tau_max = 12.0
    eps_max = 1.0 / beta

    # -------- Force GPU --------
    assert torch.cuda.is_available(), "CUDA not available — GPU required"
    device = torch.device("cuda")
    print(f"✅ Training on GPU: {torch.cuda.get_device_name(0)}")
    print(f"β = {beta}, strain limit = {eps_max}")

    # -------- Paths --------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")
    RESULTS_DIR = os.path.join(BASE_DIR, "results", f"beta_{beta}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # -------- Load dataset --------
    train_data = np.loadtxt(
        os.path.join(DATASET_DIR, "train.csv"),
        delimiter=",", skiprows=1
    )
    test_data = np.loadtxt(
        os.path.join(DATASET_DIR, "test.csv"),
        delimiter=",", skiprows=1
    )

    tau_train = torch.tensor(train_data[:, 0], dtype=torch.float32, device=device)
    eps_train = torch.tensor(train_data[:, 1], dtype=torch.float32, device=device)

    tau_test = test_data[:, 0]
    eps_test = test_data[:, 1]
    tau_test_t = torch.tensor(tau_test, dtype=torch.float32, device=device)

    # -------- Model --------
    model = OddSymmetricKAN(n_knots=121, tau_max=tau_max).to(device)

    # Physics-informed initialization
    with torch.no_grad():
        knots = model.g.knots.cpu().numpy()
        init_vals = np.minimum(knots, eps_max)
        model.g.coeffs.copy_(
            torch.tensor(init_vals, dtype=torch.float32, device=device)
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    # -------- Flattening region --------
    flat_start = 0.7 * tau_max
    centers = 0.5 * (model.g.knots[:-1] + model.g.knots[1:])
    flat_mask = (centers >= flat_start).float().to(device)

    # -------- Training --------
    for step in range(1, 4001):
        optimizer.zero_grad()
        pred = model(tau_train)
        mse = torch.mean((pred - eps_train) ** 2)

        slopes = model.g.slopes()
        mono = torch.mean(torch.relu(-slopes) ** 2)
        limit = torch.mean(torch.relu(model.g.coeffs - eps_max) ** 2)
        flat = torch.sum((slopes ** 2) * flat_mask) / (
            torch.sum(flat_mask) + 1e-12
        )

        loss = mse + 80 * mono + 80 * limit + 8 * flat
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"Step {step:4d} | MSE = {mse.item():.3e}")

    # -------- Evaluation --------
    with torch.no_grad():
        pred_test = model(tau_test_t).cpu().numpy()
        g_vals = model.g.coeffs.cpu().numpy()
        g_knots = model.g.knots.cpu().numpy()
        slopes = model.g.slopes().cpu().numpy()

    mae = np.mean(np.abs(eps_test - pred_test))
    rmse = np.sqrt(np.mean((eps_test - pred_test) ** 2))
    r2 = r2_score(eps_test, pred_test)

    # -------- Save metrics --------
    with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
        f.write(f"beta = {beta}\n")
        f.write(f"alpha = {alpha}\n")
        f.write(f"strain_limit = {eps_max}\n")
        f.write(f"MAE = {mae}\n")
        f.write(f"RMSE = {rmse}\n")
        f.write(f"R2 = {r2}\n")

    plt.style.use('seaborn-v0_8-ticks')
    plt.rcParams['pdf.fonttype'] = 42  
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Liberation Serif', 'serif'], 
        'font.size': 14,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.edgecolor': 'black',
        'axes.linewidth': 1.5,
        'lines.linewidth': 2.8,        
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
    })



    fig1, ax1 = plt.subplots(figsize=(8, 7))
    

    ax1.plot(tau_test, eps_test, label="Analytical Ground Truth", 
             color='#FF0000', linestyle='-', linewidth=4, 
             alpha=0.3, zorder=1) 
    

    ax1.plot(tau_test, pred_test, label="KAN Model Prediction", 
             color='black', linestyle='--', dashes=(5, 5), 
             linewidth=2, alpha=1.0, zorder=2)
    
    ax1.set_xlabel(r"Applied Stress, $\tau$ (MPa)")
    ax1.set_ylabel(r"Resultant Strain, $\varepsilon(\tau)$")
    ax1.tick_params(direction='in', length=6, top=True, right=True)
    ax1.legend(loc='upper left', framealpha=1)
    ax1.grid(True, linestyle=':', alpha=0.4)
    
    fig1.savefig(os.path.join(RESULTS_DIR, "stress_strain_transparent.png"))
    plt.close(fig1)


    fig2, ax2 = plt.subplots(figsize=(8, 7))
    ax2.plot(g_knots, g_vals, color='#800000', linewidth=3.5) # Deep Navy Blue
    ax2.axhline(eps_max, linestyle=":", color='#4B0082', label=f"Limit ({eps_max:.2f})") # Indigo Limit
    
    ax2.set_xlabel(r"Effective Stress $|\tau|$ (MPa)")
    ax2.set_ylabel(r"Spline Response $g(|\tau|)$")
    ax2.tick_params(direction='in', length=6, top=True, right=True)
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle=':', alpha=0.4)
    
    fig2.savefig(os.path.join(RESULTS_DIR, "g_spline_professional.png"))
    plt.close(fig2)

    
    fig3, ax3 = plt.subplots(figsize=(8, 7))
    centers_np = 0.5 * (g_knots[:-1] + g_knots[1:])
    ax3.plot(centers_np, slopes, color='#006400', linewidth=2.8) # Dark Forest Green
    ax3.axhline(0.0, linestyle="-", color='black', linewidth=1.2)
    
    ax3.set_xlabel(r"Effective Stress $|\tau|$ (MPa)")
    ax3.set_ylabel("Tangent Modulus")
    ax3.tick_params(direction='in', length=6, top=True, right=True)
    ax3.grid(True, linestyle=':', alpha=0.4)
    
    fig3.savefig(os.path.join(RESULTS_DIR, "slope_profile_professional.png"))
    plt.close(fig3)

    print(f"\n📁 Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
