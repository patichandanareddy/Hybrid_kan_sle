import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kan import KAN
from scipy.optimize import root_scalar

# =====================================================
# 1. Treloar planar experimental data
# =====================================================
lambda_exp = np.array([
    1.00, 1.20, 1.60, 2.10, 2.70,
    3.30, 3.90, 4.50, 4.90, 5.20
])
stress_exp = np.array([
    0.00, 0.28, 0.70, 1.20, 1.68,
    2.15, 2.70, 3.50, 4.40, 5.25
])

# =====================================================
# 2. SLE parameters (from calibration)
# =====================================================
alpha = 2.47956314
E     = 1.49049065     # MPa
beta  = 0.38102918

# =====================================================
# 3. SLE model
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

stress_sle = np.array([tau_from_lambda(l) for l in lambda_exp])

np.savetxt(
    "pred_planar.txt",
    np.column_stack([lambda_exp, stress_exp, stress_sle]),
    header="lambda stress_exp(MPa) stress_sle(MPa)"
)

# =====================================================
# 4. KAN residual learning
# =====================================================
x = np.log(lambda_exp).reshape(-1,1)
y = (stress_exp - stress_sle).reshape(-1,1)

X = torch.tensor(x, dtype=torch.float32)
Y = torch.tensor(y, dtype=torch.float32)

model = KAN(width=[1,6,1], grid=5, k=3, seed=0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = torch.nn.MSELoss()

loss_hist = []

for step in range(3000):
    optimizer.zero_grad()
    loss = loss_fn(model(X), Y)
    loss.backward()
    optimizer.step()
    loss_hist.append(loss.item())

    if step % 500 == 0:
        print(f"[Planar] step {step}, loss = {loss.item():.3e}")

torch.save(model.state_dict(), "kan_planar.pt")
np.savetxt("kan_planar_loss.txt", np.array(loss_hist))

# =====================================================
# 5. Final prediction
# =====================================================
with torch.no_grad():
    stress_kan = model(X).numpy().flatten()
    stress_total = stress_sle + stress_kan

# =====================================================
# 6. Paper-quality plot
# =====================================================
plt.figure(figsize=(7,5))
plt.scatter(lambda_exp, stress_exp, c="black", s=45, label="Experiment", zorder=3)
plt.plot(lambda_exp, stress_sle, "--", linewidth=2, label="SLE")
plt.plot(lambda_exp, stress_total, "-", linewidth=2.5, label="SLE + KAN")

plt.xlabel(r"Stretch $\lambda$", fontsize=12)
plt.ylabel("Engineering Stress (MPa)", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("planar_sle_kan.png", dpi=300)
plt.close()

print("✅ Planar SLE + KAN completed.")
