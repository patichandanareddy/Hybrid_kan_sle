# -*- coding: utf-8 -*-
"""
Dataset generation for strain-limiting elasticity (1D)
Beginner-friendly, HPC-safe
"""

import numpy as np
import csv
import os


# -------------------------------------------------
# Ground-truth strain-limiting law (Eq. 1)
# -------------------------------------------------
def strain_limiting_eps(tau, beta, alpha):
    """
    tau   : stress (numpy array)
    beta  : strain-limiting parameter
    alpha : smoothness parameter
    """
    return tau / np.power(1.0 + np.power(beta * np.abs(tau), alpha), 1.0 / alpha)


# -------------------------------------------------
# Main dataset generator
# -------------------------------------------------
def main():

    # ===== USER PARAMETERS (easy to change) =====
    beta = 1.0
    alpha = 2.0
    tau_max = 12.0

    n_train = 2500
    n_test = 1200

    random_seed = 42
    # ============================================

    np.random.seed(random_seed)

    # ---- Training data (random sampling) ----
    tau_train = np.random.uniform(-tau_max, tau_max, n_train)
    eps_train = strain_limiting_eps(tau_train, beta, alpha)

    # ---- Test data (uniform grid) ----
    tau_test = np.linspace(-tau_max, tau_max, n_test)
    eps_test = strain_limiting_eps(tau_test, beta, alpha)

    # ---- Save CSV files ----
    os.makedirs("dataset", exist_ok=True)

    with open("dataset/train.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tau", "epsilon"])
        for t, e in zip(tau_train, eps_train):
            writer.writerow([t, e])

    with open("dataset/test.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tau", "epsilon"])
        for t, e in zip(tau_test, eps_test):
            writer.writerow([t, e])

    # ---- Save metadata (important for paper) ----
    with open("dataset/metadata.txt", "w") as f:
        f.write("Strain-limiting elasticity dataset (1D)\n")
        f.write("-------------------------------------\n")
        f.write(f"beta      = {beta}\n")
        f.write(f"alpha     = {alpha}\n")
        f.write(f"tau_max   = {tau_max}\n")
        f.write(f"n_train   = {n_train}\n")
        f.write(f"n_test    = {n_test}\n")
        f.write(f"seed      = {random_seed}\n")
        f.write(f"epsilon_max = {1.0/beta}\n")

    print("Dataset generated successfully!")
    print("Files created:")
    print("  dataset/train.csv")
    print("  dataset/test.csv")
    print("  dataset/metadata.txt")


if __name__ == "__main__":
    main()
