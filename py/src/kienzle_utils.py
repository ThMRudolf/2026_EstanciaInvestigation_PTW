"""
kienzle_utils.py
----------------
Signal-processing utilities for CNC spindle-torque analysis.

Converted from the R auxiliary functions in mc_kienzle_STAN.qmd
(Thomas M. Rudolf, 2024-03-26).

All functions are encapsulated in the MillingSignalUtils class so the
module can be imported cleanly into analysis notebooks or Stan pipelines.

Usage
-----
    from kienzle_utils import MillingSignalUtils as MSU
    phi_wrapped, idx = MSU.lim_2pi(phi_cumsum)
    smoothed        = MSU.maf(signal, n=14)
    df_max          = MSU.max_Mc(id_reduce, Mc, phi)
    alpha, beta     = MSU.beta_dist_param(mean=0.25, var=0.02**2)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List


class MillingSignalUtils:
    """
    Static utility methods for CNC milling spindle-torque signal processing.

    All methods are @staticmethod so no instantiation is required, but the
    class groups them logically and makes selective import straightforward.
    """

    # ------------------------------------------------------------------
    # 1.  Angle wrapping
    # ------------------------------------------------------------------
    @staticmethod
    def lim_2pi(
        phi: np.ndarray,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Limit a cumulative-angle vector to the [0, 2π) domain by subtracting
        2π whenever the signal crosses a full revolution boundary.

        Mirrors the R function ``lim22pi``.

        Parameters
        ----------
        phi : array-like
            Cumulative angular position in radians (monotonically increasing).

        Returns
        -------
        phi_wrapped : np.ndarray
            Angular position wrapped to [0, 2π).
        id_reduce : list of int
            Zero-based indices at which a 2π subtraction was applied.
            Equivalent to the revolution-boundary markers used by ``max_Mc``.
        """
        phi = np.array(phi, dtype=float).copy()
        N = len(phi)
        id_reduce: List[int] = []

        for k in range(N):
            if phi[k] > 2 * np.pi:
                phi[k:] -= 2 * np.pi
                id_reduce.append(k)

        return phi, id_reduce

    # ------------------------------------------------------------------
    # 2.  Moving-average filter
    # ------------------------------------------------------------------
    @staticmethod
    def maf(in_signal: np.ndarray, n: int) -> np.ndarray:
        """
        Causal moving-average filter of window length *n*.

        For the first *n* − 1 samples the window is truncated to the
        available data (expanding window), matching the R ``maf`` behaviour.

        Parameters
        ----------
        in_signal : array-like
            Input time-series.
        n : int
            Window length (number of samples).

        Returns
        -------
        maf_signal : np.ndarray
            Filtered signal, same length as *in_signal*.
        """
        in_signal = np.asarray(in_signal, dtype=float)
        N = len(in_signal)
        maf_signal = np.zeros(N)

        for k in range(n):                          # expanding window
            maf_signal[k] = np.mean(in_signal[: k + 1])
        for k in range(n, N):                       # full window
            maf_signal[k] = np.mean(in_signal[k - n + 1 : k + 1])

        return maf_signal

    # ------------------------------------------------------------------
    # 3.  Moving-average standard-scaler
    # ------------------------------------------------------------------
    @staticmethod
    def maf_std_scaler(in_signal: np.ndarray, n: int) -> np.ndarray:
        """
        Z-score normalisation using a causal moving-average mean and
        moving standard deviation of window length *n*.

        Mirrors the R function ``maf_std_scaler``.

        Parameters
        ----------
        in_signal : array-like
            Input time-series.
        n : int
            Window length.

        Returns
        -------
        out_signal : np.ndarray
            (in_signal − moving_mean) / moving_std, same length as input.
            Samples where the moving std is zero are returned as 0.
        """
        in_signal = np.asarray(in_signal, dtype=float)
        N = len(in_signal)
        maf_mean = np.zeros(N)
        maf_sigma = np.zeros(N)

        for k in range(n):
            window = in_signal[: k + 1]
            maf_mean[k] = np.mean(window)
            maf_sigma[k] = np.std(window, ddof=1) if len(window) > 1 else 0.0

        for k in range(n, N):
            window = in_signal[k - n + 1 : k + 1]
            maf_mean[k] = np.mean(window)
            maf_sigma[k] = np.std(window, ddof=1)

        with np.errstate(invalid="ignore", divide="ignore"):
            out_signal = np.where(
                maf_sigma != 0,
                (in_signal - maf_mean) / maf_sigma,
                0.0,
            )

        return out_signal

    # ------------------------------------------------------------------
    # 4.  Per-revolution torque maximum
    # ------------------------------------------------------------------
    @staticmethod
    def max_Mc(
        id_reduce: List[int],
        Mc: np.ndarray,
        phi: np.ndarray,
    ) -> pd.DataFrame:
        """
        Find the maximum cutting torque and corresponding angular position
        within each complete spindle revolution (0 → 2π interval).

        Revolution boundaries are defined by *id_reduce*, the list of
        zero-based indices returned by :meth:`lim_2pi`.

        Mirrors the R function ``max_Mc``.

        Parameters
        ----------
        id_reduce : list of int
            Revolution-boundary indices (zero-based).
        Mc : array-like
            Cutting-torque time series (N·m or N·mm).
        phi : array-like
            Wrapped angular-position vector in radians.

        Returns
        -------
        pd.DataFrame
            Columns ``McMax`` (peak torque per revolution) and
            ``phiMcMax`` (angular position of the peak).
        """
        Mc = np.asarray(Mc, dtype=float)
        phi = np.asarray(phi, dtype=float)

        McMax_list: List[float] = []
        phiMcMax_list: List[float] = []

        k_old = 0
        for k in id_reduce:
            Mc_seg = Mc[k_old:k]
            phi_seg = phi[k_old:k]
            if len(Mc_seg) == 0:
                k_old = k
                continue
            peak_val = float(np.max(Mc_seg))
            peak_idx = int(np.argmax(Mc_seg))
            McMax_list.append(peak_val)
            phiMcMax_list.append(float(phi_seg[peak_idx]))
            k_old = k

        return pd.DataFrame({"McMax": McMax_list, "phiMcMax": phiMcMax_list})

    # ------------------------------------------------------------------
    # 5.  Beta-distribution parameters from mean and variance
    # ------------------------------------------------------------------
    @staticmethod
    def beta_dist_param(mean: float, var: float) -> Tuple[float, float]:
        """
        Compute the shape parameters α and β of a Beta distribution from
        its mean and variance.

        Uses the moment-matching identities:

            α = mean² · (1 − mean) / var − mean
            β = α · (1 − mean) / mean

        Mirrors the R function ``beta_dist_param``.

        Parameters
        ----------
        mean : float
            Desired mean of the Beta distribution, in (0, 1).
        var : float
            Desired variance of the Beta distribution.

        Returns
        -------
        alpha : float
        beta  : float

        Raises
        ------
        ValueError
            If the requested (mean, var) combination is not achievable by
            a Beta distribution (var ≥ mean · (1 − mean)).
        """
        max_var = mean * (1.0 - mean)
        if var >= max_var:
            raise ValueError(
                f"Variance {var} is not achievable for mean {mean}. "
                f"Must be < {max_var:.6g}."
            )
        alpha = mean**2 * (1.0 - mean) / var - mean
        beta = alpha * (1.0 - mean) / mean
        return alpha, beta

    # ------------------------------------------------------------------
    # 6.  No-load torque correction
    # ------------------------------------------------------------------
    @staticmethod
    def correct_no_load(
        signal: np.ndarray,
        time: np.ndarray,
        t_start: float,
        t_end: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Subtract the mean no-load (friction + inertia) component from a
        spindle-current or torque signal.

        The no-load mean is estimated from the interval [t_start, t_end]
        (spindle running freely, no tool engagement).

        Parameters
        ----------
        signal : array-like
            Full time series (current in A, or torque in N·mm).
        time : array-like
            Corresponding time vector in seconds.
        t_start, t_end : float
            Start and end times (seconds) of the no-load window.

        Returns
        -------
        corrected : np.ndarray
            Signal with no-load mean subtracted.
        no_load_mean : float
            The mean value that was subtracted.
        """
        signal = np.asarray(signal, dtype=float)
        time = np.asarray(time, dtype=float)
        mask = (time >= t_start) & (time <= t_end)
        no_load_mean = float(np.mean(signal[mask]))
        return signal - no_load_mean, no_load_mean
