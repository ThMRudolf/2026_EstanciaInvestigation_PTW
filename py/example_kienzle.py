"""
example_kienzle.py
------------------
Minimal worked example showing KienzleModel and MillingSignalUtils
working together, including Stan data-dict preparation.

Run with:
    python example_kienzle.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kienzle_utils import MillingSignalUtils as MSU
from kienzle_model import KienzleModel

# ── 1.  Define process parameters ────────────────────────────────────────────
fs      = 500           # sampling frequency, Hz
t_total = 0.5           # simulation window, s
time    = np.linspace(0, t_total, int(fs * t_total))

model = KienzleModel(
    mc        = 0.25,
    kc11      = 1800.0,         # N/mm²
    phi_in    = 0.0,            # entry angle  (0 rad = 0°)
    phi_out   = np.pi,          # exit  angle  (π rad = 180° → half immersion)
    fz        = 0.1,            # mm/tooth
    ap        = 2.0,            # mm
    omega     = 600.0,          # rpm
    time      = time,
    z         = 4,
    r_tool    = 10.0,           # mm
    kappa_deg = 90.0,           # °
    km        = 1.3,            # N·mm / A
    noise_std = 5.0,            # N·mm  (add realistic measurement noise)
)

print(model)

# ── 2.  Simulate torque and current time series ───────────────────────────────
df = model.to_dataframe()
print(df.head())

# ── 3.  Apply moving-average filter ──────────────────────────────────────────
df["Mc_maf7"]  = MSU.maf(df["Mc"].values, n=7)
df["Mc_maf30"] = MSU.maf(df["Mc"].values, n=30)

# ── 4.  Wrap cumulative angle and find per-revolution peaks ──────────────────
# Simulate a cumulative (unwrapped) spindle angle the way the CNC controller
# would provide it, then wrap it to [0, 2π).
omega_rad   = model.omega * 2 * np.pi / 60
phi_cum     = omega_rad * time
phi_wrapped, id_reduce = MSU.lim_2pi(phi_cum)

df_peaks = MSU.max_Mc(id_reduce, df["Mc"].values, phi_wrapped)
print(f"\nPer-revolution torque peaks (first 5):\n{df_peaks.head()}")

# ── 5.  Beta-distribution priors for m_c ─────────────────────────────────────
alpha_mc, beta_mc = MSU.beta_dist_param(mean=0.25, var=0.02**2)
print(f"\nBeta prior on m_c:  α = {alpha_mc:.4f},  β = {beta_mc:.4f}")

# ── 6.  Build Stan data dictionary ───────────────────────────────────────────
stan_data = model.to_stan_data(
    Mc_observed  = df["Mc"].values,
    phi_observed = phi_wrapped,
    m_kc         = 2306.0,
    sd_kc        = 977.0,
)
print("\nStan data keys and shapes / values:")
for k, v in stan_data.items():
    if isinstance(v, list):
        print(f"  {k:12s}: list of {len(v)} floats")
    else:
        print(f"  {k:12s}: {v}")

# ── 7.  Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

# Time-domain torque
ax1 = axes[0]
ax1.plot(df["time"], df["Mc"],      color="steelblue", alpha=0.4, label="Mc (noisy)")
ax1.plot(df["time"], df["Mc_maf7"], color="crimson",   lw=1.5,    label="MAF n=7")
ax1.plot(df["time"], df["Mc_maf30"],color="forestgreen",lw=2,     label="MAF n=30")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Cutting torque (N·mm)")
ax1.set_title("Kienzle torque — time domain")
ax1.legend()

# Per-revolution peaks vs. angular position
ax2 = axes[1]
ax2.plot(np.degrees(df_peaks["phiMcMax"]), df_peaks["McMax"],
         marker="o", ms=4, color="darkorange", label="Peak M_c per revolution")
ax2.set_xlabel("Angular position of peak (deg)")
ax2.set_ylabel("Peak cutting torque (N·mm)")
ax2.set_title("Per-revolution torque maxima")
ax2.legend()

plt.tight_layout()
plt.savefig("kienzle_example.png", dpi=150)
plt.show()
print("\nFigure saved to kienzle_example.png")
