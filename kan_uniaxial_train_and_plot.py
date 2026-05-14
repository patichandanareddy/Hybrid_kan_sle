import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kan import KAN
from scipy.optimize import root_scalar

# =====================================================
# 1. Treloar uniaxial experimental data
# =====================================================
lambda_exp = np.array([
    1.00, 1.10, 1.62, 2.15, 2.85, 3.20, 3.82,
    4.65, 5.15, 5.85, 6.60, 7.10, 7.60, 7.75
])
stress_exp = np.array([
    0.00, 0.20, 0.84, 1.25, 1.60, 1.78, 2.10,
    2.65, 3.25, 4.35, 6.00, 7.60, 10.30, 11.20
])

# =====================================================
# 2. SLE parameters (FIXED from calibration)
# =====================================================
alpha = 1.53836245
E     = 1.05854390     # MPa
beta  = 0.43750850

# =====================================================
# 3. SLE model (strain-limiting elasticity)
# =====================================================
def eps_sle(tau):
    return (tau / E) / (1.0 + (beta * abs(tau))**alpha)**(1.0/alpha)

def tau_from_lambda(lmbda):
    eps_target = np.log(lmbda)
    if eps_target == 0.0:
        return 0.0

    def f(tau):
        return eps_sle(tau) - eps_target

    sol = root_scalar(f, bracket=[0, 50], method="bisect")
    return sol.root

# Compute SLE stress
stress_sle = np.array([tau_from_lambda(l) for l in lambda_exp])

# Save dataset used (important for paper reproducibility)
np.savetxt(
    "pred_uniaxial.txt",
    np.column_stack([lambda_exp, stress_exp, stress_sle]),
    header="lambda stress_exp(MPa) stress_sle(MPa)"
)

# =====================================================
# 4. Prepare KAN training data (residual learning)
# =====================================================
x = np.log(lambda_exp).reshape(-1, 1)
y = (stress_exp - stress_sle).reshape(-1, 1)

X = torch.tensor(x, dtype=torch.float32)
Y = torch.tensor(y, dtype=torch.float32)

# =====================================================
# 5. Define and train KAN
# =====================================================
model = KAN(width=[1, 6, 1], grid=5, k=3, seed=0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = torch.nn.MSELoss()

loss_hist = []

for step in range(3000):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, Y)
    loss.backward()
    optimizer.step()
    loss_hist.append(loss.item())

    if step % 500 == 0:
        print(f"[Uniaxial] step {step}, loss = {loss.item():.3e}")

torch.save(model.state_dict(), "kan_uniaxial.pt")
np.savetxt("kan_uniaxial_loss.txt", np.array(loss_hist))

# =====================================================
# 6. Final prediction (SLE + KAN)
# =====================================================
with torch.no_grad():
    stress_kan = model(X).numpy().flatten()
    stress_total = stress_sle + stress_kan


plt.figure(figsize=(7,5))
plt.scatter(lambda_exp, stress_exp, c="black", s=45, label="Experiment", zorder=3)
plt.plot(lambda_exp, stress_sle, "--", linewidth=2, label="SLE")
plt.plot(lambda_exp, stress_total, "-", linewidth=2.5, label="SLE + KAN")

plt.xlabel(r"Stretch $\lambda$", fontsize=12)
plt.ylabel("Engineering Stress (MPa)", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("uniaxial_sle_kan.png", dpi=300)
plt.close()

print("✅ Uniaxial SLE + KAN completed (dataset saved internally).")
