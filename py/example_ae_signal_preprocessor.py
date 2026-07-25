"""
example_ae_signal_preprocessor.py
----------------------------------
Demonstrates AeSignalPreprocessor end to end:

  1. Build a synthetic spindle-current measurement (no-load ramp-up +
     Kienzle-driven cutting + a slow static-bending drift), the same
     shape of signal AeSignalPreprocessor expects from a real
     Sinumerik trace.
  2. Run the preprocessing pipeline (ramp-up trim, static-bending
     removal, segmentation, feature extraction) to get a feature
     matrix X / target vector y.
  3. Load one of the trained checkpoints saved by
     `ae_estimation_step12_gridsearch_vs_tpe*.ipynb`
     (`ae_mlp_tpe_best.pt` + `ae_mlp_tpe_comparison_study.pkl`) and run
     it on the freshly preprocessed features.

Run with (from the py/ directory):
    python example_ae_signal_preprocessor.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler

from src.ae_signal_preprocessor import AeSignalPreprocessor
from src.kienzle_model import KienzleModel
from src.kienzle_utils import MillingSignalUtils as MSU

REPO_ROOT = Path(__file__).resolve().parent.parent
# Swap for "gpu_3Layers" / "cpu_5Layers" / "gpu_5Layers" to load another run.
RESULTS_DIR = REPO_ROOT / "nb_ae" / "results" / "cpu_3Layers"


# ── 1. Synthesize a measurement (stand-in for a real Sinumerik CSV trace) ────
def synthesize_measurement(
    ae: float,
    f: float = 1200.0,
    r_tool: float = 10.0,
    n_rpm: float = 6000.0,
    ap: float = 1.0,
    fz: float = 0.05,
    z: int = 2,
    fs: float = 500.0,
    cutting_duration: float = 4.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a no-load + cutting current trace for a given radial depth of cut ae."""
    s_offset = 5.0
    t_ramp_up = (2.0 * r_tool + s_offset) / f * 60.0
    rng = np.random.default_rng(seed)

    # no-load portion: tool still travelling toward the workpiece
    t_no_load = np.arange(0.0, t_ramp_up, 1.0 / fs)
    i_no_load = 1.8 + rng.normal(0, 0.05, len(t_no_load))

    # cutting portion: Kienzle-driven current over the ae-dependent engagement window
    D = 2.0 * r_tool
    phi_out = MSU.engagement_angle_from_ae(ae, D)
    t_cut = np.arange(0.0, cutting_duration, 1.0 / fs)
    model = KienzleModel(
        mc=0.25, kc11=2300.0, phi_in=0.0, phi_out=phi_out,
        fz=fz, ap=ap, omega=n_rpm, time=t_cut, z=z, r_tool=r_tool,
        kappa_deg=90.0, km=1300.0, I0=1.8,
        white_noise_amplitude=2.0, normal_noise_std=1.0, seed=seed,
    )
    i_cut = model.current_time_series(add_noise=True)

    time = np.concatenate([t_no_load, t_ramp_up + t_cut])
    current = np.concatenate([i_no_load, i_cut])

    # slow structural drift the MAF static-bending removal step should cancel out
    current = current + 0.4 * np.sin(2 * np.pi * 0.08 * time)

    return pd.DataFrame({"time": time, "current": current})


PROCESS_PARAMS = dict(f=1200.0, r_tool=10.0, n_rpm=6000.0, ap=1.0, fz=0.05, D=20.0, z=2)

pre = AeSignalPreprocessor(fs=500.0, n_segments=4)

samples = []
for ae in (3.0, 6.0, 9.0):
    df_meas = synthesize_measurement(ae=ae, seed=int(ae))
    samples += pre.process_measurement(
        df_meas, signal_col="current",
        f=PROCESS_PARAMS["f"], r_tool=PROCESS_PARAMS["r_tool"], n_rpm=PROCESS_PARAMS["n_rpm"],
        ap=PROCESS_PARAMS["ap"], fz=PROCESS_PARAMS["fz"], D=PROCESS_PARAMS["D"],
        z=PROCESS_PARAMS["z"], ae=ae,
    )

X, y = pre.build_dataset(samples)
print(f"Preprocessed {len(samples)} segments -> X {X.shape}, y {y}")
print("feature_names:", pre.feature_names)


# ── 2. Load a trained checkpoint + its Optuna study ──────────────────────────
class AeEstimatorMLP(nn.Module):
    """Mirrors the architecture trained in ae_estimation_step12_*.ipynb (Step 7)."""

    def __init__(self, n_features: int, hidden: list[int], dropout: float):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# Local, trusted checkpoint -> weights_only=False to also restore the
# hyperparameters (hidden_dims/dropout) saved alongside the weights.
ckpt = torch.load(RESULTS_DIR / "ae_mlp_tpe_best.pt", map_location="cpu", weights_only=False)
model = AeEstimatorMLP(n_features=len(pre.feature_names), hidden=ckpt["hidden_dims"], dropout=ckpt["dropout"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"\nLoaded '{ckpt['method']}' model, hidden={ckpt['hidden_dims']}, "
      f"trained with n_segments={ckpt['n_segments']}")

study = joblib.load(RESULTS_DIR / "ae_mlp_tpe_comparison_study.pkl")
print(f"Optuna study: best_value={study.best_value:.5f}, best_params={study.best_params}")


# ── 3. Run inference on the preprocessed features ────────────────────────────
# NOTE: the checkpoint only stores model weights, not the StandardScaler
# fitted on the original training features. To faithfully reproduce
# training-time inference, persist scaler_X/scaler_y next to the checkpoint
# when training (e.g. joblib.dump(scaler_X, "scaler_X.pkl")) and load them
# here instead of refitting. This example refits on its own demo batch
# purely to illustrate the call path, so the predicted values below are
# illustrative, not a faithful reproduction of the trained model's accuracy.
scaler_X = StandardScaler().fit(X)
scaler_y = StandardScaler().fit(y.reshape(-1, 1))

X_sc = scaler_X.transform(X)
with torch.no_grad():
    y_pred_sc = model(torch.tensor(X_sc, dtype=torch.float32)).numpy()
y_pred = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()

print("\ntrue ae [mm] | predicted ae [mm] | segment")
for sample, true_ae, pred_ae in zip(samples, y, y_pred):
    print(f"{true_ae:11.1f} | {pred_ae:17.2f} | seg_idx={sample['seg_idx']}")
