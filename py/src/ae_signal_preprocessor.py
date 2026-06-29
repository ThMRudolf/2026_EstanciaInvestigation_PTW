"""
ae_signal_preprocessor.py
--------------------------
Reusable signal-preprocessing pipeline for radial-depth-of-cut (ae)
estimation from milling spindle-current/torque traces.

The `nb_ae/ae_estimation_*.ipynb` notebooks all `%run` (or copy-paste)
the same sequence of steps before any model is trained:

    1. trim the no-load ramp-up trajectory (tool not yet engaged),
    2. estimate the no-load mean current over a friction window,
    3. remove the low-frequency static-bending trend with a causal
       moving-average filter (one-revolution window),
    4. split the remaining cutting signal into N equal-length segments,
    5. extract physics-informed features per segment (RMS, crest, peak,
       per-revolution statistics, tooth-mesh spectral energy),
    6. normalise the mean per-revolution torque by the Kienzle prediction,
    7. assemble the final feature vector used by the MLP / tree models.

This module collects those steps into :class:`AeSignalPreprocessor` so the
same logic can be applied to new measurement signals/files without
copy-pasting notebook cells.

Usage
-----
    from ae_signal_preprocessor import AeSignalPreprocessor

    pre = AeSignalPreprocessor(fs=500.0, n_segments=4)

    # single measurement -> list of per-segment sample dicts
    samples = pre.process_measurement(
        df, signal_col="+/Nck/!SD/nckServoDataActCurr32 [u1; 4]",
        f=1200.0, r_tool=10.0, n_rpm=6500.0,
        ap=1.0, fz=0.05, D=20.0, z=2, ae=5.0,
    )

    # whole experiment table (one row per CSV file) -> flat sample list
    samples = pre.process_experiment_table(df_info, data_dir=r"G:\...\all")

    X, y = pre.build_dataset(samples)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    # works when imported as part of the `src` package (e.g. `from src.ae_signal_preprocessor import ...`)
    from .kienzle_utils import MillingSignalUtils as MSU
except ImportError:
    # works when py/src is added to sys.path directly (notebook convention)
    from kienzle_utils import MillingSignalUtils as MSU


@dataclass
class AeSignalPreprocessor:
    """
    Preprocessing pipeline for spindle-current/torque cutting signals,
    used to estimate the radial depth of cut (ae).

    Parameters
    ----------
    fs : float
        Sampling frequency of the signal (Hz).
    s_offset : float
        NC-program trajectory offset (mm) before first tool contact,
        used to compute the ramp-up and no-load (friction) window
        durations from the feed rate and tool radius.
    n_segments : int
        Default number of equal-length segments each cutting signal is
        split into (overridable per call).
    time_col : str
        Name of the time column (seconds) in the raw measurement
        DataFrame.
    kc11_nominal, mc_nominal : float
        Default Kienzle material parameters used to normalise the
        per-revolution torque feature when not explicitly supplied.
    """

    fs: float = 500.0
    s_offset: float = 5.0
    n_segments: int = 4
    time_col: str = "time"
    kc11_nominal: float = 2300.0
    mc_nominal: float = 0.23

    feature_names: ClassVar[List[str]] = [
        "rms", "peak", "crest", "mean_rect", "std",
        "mean_rev_mean", "mean_rev_max", "std_rev_max", "tooth_energy",
        "Mc_norm", "ap", "fz", "n",
    ]

    # ------------------------------------------------------------------
    # 1. Ramp-up / no-load window timing
    # ------------------------------------------------------------------
    def ramp_up_time(self, f: float, r_tool: float) -> float:
        """Time (s) for the tool to travel the ramp-up trajectory to first ae contact."""
        s_ramp_up = 2.0 * r_tool + self.s_offset
        return s_ramp_up / f * 60.0

    def no_load_time(self, f: float, r_tool: float) -> float:
        """Time (s) of the no-load trajectory used to estimate the friction offset."""
        s_friction_trace = r_tool + self.s_offset
        return s_friction_trace / f * 60.0

    def ramp_up_index(self, df: pd.DataFrame, f: float, r_tool: float) -> int:
        """First sample index at which the tool has reached full ae engagement."""
        t_ramp_up = self.ramp_up_time(f, r_tool)
        return int(np.argmax(df[self.time_col].to_numpy() > t_ramp_up))

    def no_load_index(self, df: pd.DataFrame, f: float, r_tool: float) -> int:
        """Last sample index still within the no-load (friction) window."""
        t_friction = self.no_load_time(f, r_tool)
        return int(np.argmax(df[self.time_col].to_numpy() > t_friction))

    # ------------------------------------------------------------------
    # 2. No-load friction mean
    # ------------------------------------------------------------------
    def no_load_mean(self, df: pd.DataFrame, signal_col: str, f: float, r_tool: float) -> float:
        """Mean signal value over the no-load (friction) window."""
        idx_frict = self.no_load_index(df, f, r_tool)
        return float(np.mean(df[signal_col].to_numpy(dtype=float)[:idx_frict]))

    # ------------------------------------------------------------------
    # 3. Static-bending removal (causal moving-average filter)
    # ------------------------------------------------------------------
    def remove_static_bending(self, signal: np.ndarray, n_rpm: float) -> np.ndarray:
        """Subtract a one-revolution causal moving average to remove the low-frequency trend."""
        ns = round(self.fs / (n_rpm / 60.0))
        signal = np.asarray(signal, dtype=float)
        return signal - MSU.maf(signal, ns)

    # ------------------------------------------------------------------
    # 4. Extract the cutting-only signal (ramp-up trimmed, bending-corrected)
    # ------------------------------------------------------------------
    def extract_cutting_signal(
        self,
        df: pd.DataFrame,
        signal_col: str,
        f: float,
        r_tool: float,
        n_rpm: float,
        remove_bending: bool = True,
        subtract_no_load: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """
        Trim the no-load ramp-up trajectory and optionally correct for
        static bending and/or no-load friction.

        Returns
        -------
        iq_eval : np.ndarray
            Cutting-only signal, starting at :meth:`ramp_up_index`.
        mean_no_load : float
            Mean signal value over the no-load window (for bookkeeping,
            and used as the correction when ``subtract_no_load=True``).
        """
        idx_start = self.ramp_up_index(df, f, r_tool)
        mean_no_load = self.no_load_mean(df, signal_col, f, r_tool)

        signal = df[signal_col].to_numpy(dtype=float).copy()
        if remove_bending:
            signal = self.remove_static_bending(signal, n_rpm)
        if subtract_no_load:
            signal = signal - mean_no_load

        return signal[idx_start:], mean_no_load

    # ------------------------------------------------------------------
    # 5. Equal-length segmentation
    # ------------------------------------------------------------------
    def segment_signal(self, signal: np.ndarray, n_segments: Optional[int] = None) -> np.ndarray:
        """Split *signal* into ``n_segments`` equal-length chunks, dropping any remainder."""
        n_segments = self.n_segments if n_segments is None else n_segments
        signal = np.asarray(signal, dtype=float)
        seg_len = len(signal) // n_segments
        if seg_len == 0:
            raise ValueError(
                f"Signal of length {len(signal)} is too short to split into "
                f"{n_segments} segments."
            )
        n_used = seg_len * n_segments
        return signal[:n_used].reshape(n_segments, seg_len)

    # ------------------------------------------------------------------
    # 6. Per-measurement pipeline -> list of per-segment sample dicts
    # ------------------------------------------------------------------
    def process_measurement(
        self,
        df: pd.DataFrame,
        signal_col: str,
        f: float,
        r_tool: float,
        n_rpm: float,
        ap: float,
        fz: float,
        D: float,
        z: float,
        ae: float,
        n_segments: Optional[int] = None,
        remove_bending: bool = True,
        subtract_no_load: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Run the full preprocessing pipeline for one measurement (ramp-up
        removal, static-bending/friction compensation, segmentation) and
        return one sample dict per resulting segment, ready for
        :meth:`build_feature_vector`.
        """
        iq_eval, mean_no_load = self.extract_cutting_signal(
            df, signal_col, f, r_tool, n_rpm,
            remove_bending=remove_bending, subtract_no_load=subtract_no_load,
        )
        segments = self.segment_signal(iq_eval, n_segments)

        return [
            {
                "signal": segment,
                "ap": float(ap),
                "fz": float(fz),
                "n": float(n_rpm),
                "D": float(D),
                "z": float(z),
                "ae": float(ae),
                "mean_iq_frict": float(mean_no_load),
                "seg_idx": seg_idx,
            }
            for seg_idx, segment in enumerate(segments)
        ]

    # ------------------------------------------------------------------
    # 6b. Whole experiment table -> flat sample list
    # ------------------------------------------------------------------
    def process_experiment_table(
        self,
        info_df: pd.DataFrame,
        data_dir: str,
        n_segments: Optional[int] = None,
        file_col: str = "SinuTraceFile (*.csv)",
        signal_col: str = "+/Nck/!SD/nckServoDataActCurr32 [u1; 4]",
        f_col: str = "f (mm/min)",
        r_col: str = "R (mm)",
        n_col: str = "N (rpm)",
        ap_col: str = "Ap (mm)",
        fz_col: str = "fz",
        z_col: str = "Z",
        ae_col: str = "Ae (mm)",
        remove_bending: bool = True,
        subtract_no_load: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Apply :meth:`process_measurement` to every row of an experiment
        info table (one row per recorded CSV trace), as done in the
        `dataset_seg` build loop shared by all `ae_estimation_*.ipynb`
        notebooks. Each returned sample also carries ``cut_exper_idx``,
        the row index it came from.
        """
        samples: List[Dict[str, Any]] = []
        for exper_idx in range(len(info_df)):
            row = info_df.iloc[exper_idx]
            df_raw = pd.read_csv(os.path.join(data_dir, row[file_col]))
            r_tool = float(row[r_col])

            seg_samples = self.process_measurement(
                df_raw,
                signal_col=signal_col,
                f=float(row[f_col]),
                r_tool=r_tool,
                n_rpm=float(row[n_col]),
                ap=float(row[ap_col]),
                fz=float(row[fz_col]),
                D=2.0 * r_tool,
                z=float(row[z_col]),
                ae=float(row[ae_col]),
                n_segments=n_segments,
                remove_bending=remove_bending,
                subtract_no_load=subtract_no_load,
            )
            for s in seg_samples:
                s["cut_exper_idx"] = exper_idx
            samples.extend(seg_samples)

        return samples

    # ------------------------------------------------------------------
    # 7. Physics-informed feature extraction (per segment)
    # ------------------------------------------------------------------
    def extract_signal_features(self, signal: np.ndarray, n_rpm: float, z: float) -> Dict[str, float]:
        """Compute RMS/crest/peak, per-revolution, and tooth-mesh spectral-energy features."""
        signal = np.asarray(signal, dtype=float)

        samples_per_rev = int(self.fs * 60 / n_rpm)
        n_revs = len(signal) // samples_per_rev
        if n_revs == 0:
            revolutions = signal.reshape(1, -1)
        else:
            revolutions = signal[: n_revs * samples_per_rev].reshape(n_revs, samples_per_rev)

        rms = np.sqrt(np.mean(signal ** 2))
        peak = np.max(np.abs(signal))
        crest = peak / (rms + 1e-9)
        mean_rect = np.mean(np.abs(signal))
        std = np.std(signal)

        rev_means = revolutions.mean(axis=1)
        rev_maxes = revolutions.max(axis=1)

        spectrum = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), d=1 / self.fs)
        tooth_freq = n_rpm / 60 * z
        band_mask = (freqs >= tooth_freq - 50) & (freqs <= tooth_freq + 50)
        denom = np.sum(spectrum ** 2)
        tooth_energy = np.sum(spectrum[band_mask] ** 2) / (denom + 1e-9)

        return {
            "rms": rms, "peak": peak, "crest": crest,
            "mean_rect": mean_rect, "std": std,
            "mean_rev_mean": float(rev_means.mean()), "mean_rev_max": float(rev_maxes.mean()),
            "std_rev_max": float(rev_maxes.std()), "tooth_energy": tooth_energy,
        }

    # ------------------------------------------------------------------
    # 8. Kienzle-normalized torque residual
    # ------------------------------------------------------------------
    @staticmethod
    def kienzle_normalized_torque(
        Mc_mean: float, kc11: float, mc: float, ap: float, fz: float, D: float
    ) -> float:
        """
        Normalise measured mean torque by the Kienzle theoretical
        prediction; the residual carries information about ae.

            Mc_kienzle = kc11 * ap * fz^(1 - mc) * (D / 2)   [simplified]

        Returns 0.0 for physically meaningless samples (ap <= 0 or fz <= 0).
        """
        if ap <= 0 or fz <= 0:
            return 0.0
        Mc_kienzle = kc11 * ap * (fz ** (1.0 - mc)) * (D / 2.0)
        return Mc_mean / (Mc_kienzle + 1e-9)

    # ------------------------------------------------------------------
    # 9. Full feature vector for one sample/segment
    # ------------------------------------------------------------------
    def build_feature_vector(
        self, sample: Dict[str, Any], kc11: Optional[float] = None, mc: Optional[float] = None
    ) -> np.ndarray:
        """
        Assemble the 13-element feature vector (signal features +
        Kienzle-normalised torque + process parameters) used to train
        the ae estimators. Order matches :attr:`feature_names`.
        """
        kc11 = self.kc11_nominal if kc11 is None else kc11
        mc = self.mc_nominal if mc is None else mc

        sig_feats = self.extract_signal_features(sample["signal"], sample["n"], sample["z"])
        Mc_norm = self.kienzle_normalized_torque(
            sig_feats["mean_rev_max"], kc11, mc, sample["ap"], sample["fz"], sample["D"]
        )

        signal_feats = np.array([
            sig_feats["rms"], sig_feats["peak"], sig_feats["crest"],
            sig_feats["mean_rect"], sig_feats["std"],
            sig_feats["mean_rev_mean"], sig_feats["mean_rev_max"],
            sig_feats["std_rev_max"], sig_feats["tooth_energy"], Mc_norm,
        ])
        process_feats = np.array([sample["ap"], sample["fz"], sample["n"]])
        return np.concatenate([signal_feats, process_feats])

    # ------------------------------------------------------------------
    # 10. Batch dataset assembly
    # ------------------------------------------------------------------
    def build_dataset(
        self,
        samples: List[Dict[str, Any]],
        kc11: Optional[float] = None,
        mc: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build the feature matrix ``X`` and target vector ``y`` (ae) from a sample list."""
        X = np.array([self.build_feature_vector(s, kc11, mc) for s in samples])
        y = np.array([s["ae"] for s in samples])
        return X, y
