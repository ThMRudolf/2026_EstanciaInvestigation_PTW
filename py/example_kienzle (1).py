"""
example_kienzle.py
------------------
Demonstrates KienzleModel with the dual noise model (white + Gaussian)
and MillingSignalUtils working together.

Run with:
    python example_kienzle.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.kienzle_utils import MillingSignalUtils as MSU
from src.kienzle_model import KienzleModel

# ── 1.  Process parameters ────────────────────────────────────────────────────
fs, t_total = 500, 0.5
time = np.linspace(0, t_total, int(fs * t_total))

model = KienzleModel(
    mc=0.25, kc11=1800.0,
    phi_in=0.0, phi_out=np.pi,      # half-immersion
    fz=0.1, ap=2.0, omega=600.0,
    time=time, z=4,
    r_tool=10.0, kappa_deg=90.0, km=1.3,
    # ── noise ──────────────────────────────────────
    white_noise_amplitude=30.0,     # N·mm  uniform  ±30
    normal_noise_mean=5.0,          # N·mm  systematic offset
    normal_noise_std=20.0,          # N·mm  Gaussian spread
    seed=42,
)
print(model)
print()

# ── 2.  Signals ───────────────────────────────────────────────────────────────
df = model.to_dataframe()           # Mc_clean, Mc, noise, Iq_clean, Iq
print(f"Estimated SNR: {model.snr_db():.1f} dB\n")

# Moving-average filter on noisy Mc
df["Mc_maf7"]  = MSU.maf(df["Mc"].values, n=7)
df["Mc_maf30"] = MSU.maf(df["Mc"].values, n=30)

# ── 3.  Per-revolution peaks ──────────────────────────────────────────────────
omega_rad   = model.omega * 2 * np.pi / 60
phi_cum     = omega_rad * time
phi_wrapped, id_reduce = MSU.lim_2pi(phi_cum)
df_peaks = MSU.max_Mc(id_reduce, df["Mc"].values, phi_wrapped)

# ── 4.  Noise statistics ──────────────────────────────────────────────────────
noise_vec = df["noise"].values
print(f"Noise  mean={noise_vec.mean():.2f}  std={noise_vec.std():.2f}")

# ── 5.  Three-panel figure ────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 9))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# Panel A — clean vs noisy torque (time domain)
ax_a = fig.add_subplot(gs[0, :])
ax_a.plot(df["time"], df["Mc_clean"], color="steelblue", lw=1.5, label="Mc clean")
ax_a.plot(df["time"], df["Mc"],       color="grey",      alpha=0.4, lw=0.8, label="Mc noisy")
ax_a.plot(df["time"], df["Mc_maf30"], color="crimson",   lw=1.8, label="MAF n=30")
ax_a.set_xlabel("Time (s)")
ax_a.set_ylabel("Torque (N·mm)")
ax_a.set_title("A — Kienzle torque: clean vs noisy vs filtered")
ax_a.legend(fontsize=8)

# Panel B — noise time series
ax_b = fig.add_subplot(gs[1, 0])
ax_b.plot(df["time"], noise_vec, color="darkorange", lw=0.7, alpha=0.8)
ax_b.axhline(0, color="k", lw=0.8, ls="--")
ax_b.set_xlabel("Time (s)")
ax_b.set_ylabel("Noise (N·mm)")
ax_b.set_title("B — Noise signal  (white + Gaussian)")

# Panel C — noise histogram
ax_c = fig.add_subplot(gs[1, 1])
ax_c.hist(noise_vec, bins=40, color="darkorange", edgecolor="white", alpha=0.85)
ax_c.set_xlabel("Noise amplitude (N·mm)")
ax_c.set_ylabel("Count")
ax_c.set_title("C — Noise distribution")

# Panel D — per-revolution torque peaks
ax_d = fig.add_subplot(gs[2, 0])
ax_d.plot(np.degrees(df_peaks["phiMcMax"]), df_peaks["McMax"],
          marker="o", ms=5, color="forestgreen", lw=1.2, label="peak / rev")
ax_d.set_xlabel("Angular position of peak (°)")
ax_d.set_ylabel("Peak torque (N·mm)")
ax_d.set_title("D — Per-revolution torque maxima")
ax_d.legend(fontsize=8)

# Panel E — Iq (spindle current) clean vs noisy
ax_e = fig.add_subplot(gs[2, 1])
ax_e.plot(df["time"], df["Iq_clean"], color="steelblue", lw=1.5, label="Iq clean")
ax_e.plot(df["time"], df["Iq"],       color="grey",      alpha=0.4, lw=0.8, label="Iq noisy")
ax_e.set_xlabel("Time (s)")
ax_e.set_ylabel("Current (A)")
ax_e.set_title("E — Equivalent spindle current")
ax_e.legend(fontsize=8)

plt.savefig("kienzle_example.png", dpi=150, bbox_inches="tight")
print("Figure saved → kienzle_example.png")
plt.show()
