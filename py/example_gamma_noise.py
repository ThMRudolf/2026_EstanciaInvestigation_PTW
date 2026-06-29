import numpy as np
import matplotlib.pyplot as plt
from src.kienzle_model import KienzleModel

# -------------------------------------------------------------------
# Build a model with Gamma-distributed noise level (from the fit to
# std_iq_no_load): sigma_run ~ Gamma(shape=161.76, scale=0.00124)
# -------------------------------------------------------------------
model = KienzleModel(
    mc=0.25,
    kc11=1800.0,
    phi_in=0.0,
    phi_out=np.pi,
    fz=0.1,
    ap=2.0,
    omega=600.0,
    time=np.linspace(0, 0.5, 2500),   # 2500 samples
    z=4,
    r_tool=10.0,
    kappa_deg=90.0,
    km=1.3,
    gamma_shape=161.763,
    gamma_scale=0.001242,
    gamma_loc=0.0,
    seed=42,
)

# -------------------------------------------------------------------
# Clean vs. noisy spindle current
# -------------------------------------------------------------------
Iq_clean = model.current_time_series(add_noise=False)
Iq_noisy = model.current_time_series(add_noise=True)

print(f"sigma_no_load drawn for this run: {model.last_sigma_no_load:.5f}")

# -------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(model.time, Iq_clean, label="Iq clean", lw=2)
plt.plot(model.time, Iq_noisy, label="Iq + Gamma noise level", alpha=0.7)
plt.xlabel("time (s)")
plt.ylabel("Iq (A)")
plt.title(f"Spindle current with Gamma noise (σ_run = {model.last_sigma_no_load:.4f})")
plt.legend()
plt.tight_layout()
plt.savefig("results/gamma_noise_example.png", dpi=150)
print("Saved plot to gamma_noise_example.png")

# -------------------------------------------------------------------
# Generate several runs to see noise-level variability
# -------------------------------------------------------------------
sigmas = []
for run in range(5):
    Iq_run = model.current_time_series(add_noise=True)
    sigmas.append(model.last_sigma_no_load)

print("sigma_no_load across 5 runs:", [f"{s:.4f}" for s in sigmas])
