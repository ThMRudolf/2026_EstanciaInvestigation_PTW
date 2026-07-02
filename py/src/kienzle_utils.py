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
    phi_meas        = MSU.phi_from_encoder(encoder_deg, mach_time_s, force_time_s)
    Mc_meas         = MSU.spindle_torque(force_0, force_1, phi_meas, r_tool)
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

    # ------------------------------------------------------------------
    # 7.  convert ae into entry angle
    # ------------------------------------------------------------------
    @staticmethod
    def entry_angle_from_ae(ae: float, D: float, phi_out: float) -> float:
        """
        Compute the tool-engagement entry angle phi_in (rad) from the radial
        depth of cut ae, tool diameter D, and the (known) exit angle phi_out.

        Relation:  phi_out - phi_in = arccos(1 - 2*ae/D)

        Parameters
        ----------
        ae : float
            Radial depth of cut (same units as D, e.g. mm). Must satisfy 0 < ae <= D.
        D : float
            Tool diameter (e.g. mm). D = 2 * r_tool.
        phi_out : float
            Exit angle (rad).

        Returns
        -------
        phi_in : float (rad)
        """
        if not (0.0 < ae <= D):
            raise ValueError("ae must satisfy 0 < ae <= D")

        delta_phi = np.arccos(1.0 - 2.0 * ae / D)
        phi_in = phi_out - delta_phi

        # Optional: wrap into [0, 2*pi)
        phi_in = phi_in % (2.0 * np.pi)
        return phi_in

    # ------------------------------------------------------------------
    # 8.  radial engagement (immersion) angle from ae
    # ------------------------------------------------------------------
    @staticmethod
    def engagement_angle_from_ae(ae: float, D: float) -> float:
        """
        Radial engagement (immersion) angle phi_eng (rad), i.e. the
        angular width of the cutter-engagement window:

            phi_eng = arccos(1 - 2*ae/D) = phi_out - phi_in

        Parameters
        ----------
        ae : float
            Radial depth of cut (mm).
        D : float
            Tool diameter (mm), D = 2 * r_tool.

        Returns
        -------
        phi_eng : float (rad), in [0, pi].
            ae = D (slotting) -> phi_eng = pi.
        """
        ratio = np.clip(1.0 - 2.0 * ae / D, -1.0, 1.0)
        return float(np.arccos(ratio))

    # ------------------------------------------------------------------
    # 9.  Measured spindle angle from the machine-channel encoder
    # ------------------------------------------------------------------
    @staticmethod
    def phi_from_encoder(
        encoder_deg: np.ndarray,
        sample_time_s: np.ndarray,
        target_time_s: np.ndarray,
        phi0_offset_deg: float = 0.0,
    ) -> np.ndarray:
        """
        Resample the spindle's absolute encoder angle (e.g. the CNC
        controller's ``axis0_positionact``, logged at the machine-channel
        rate, hardware-wrapped to [0, 360) degrees) onto another time
        vector (typically a force channel logged at a much higher rate),
        returning the spindle angular position phi in radians.

        Unlike :meth:`lim_2pi`, this does not integrate a speed signal:
        the encoder angle is unwrapped with ``np.unwrap`` (vectorized,
        handles the periodic wrap directly) and then linearly
        interpolated onto *target_time_s*, which is far cheaper than
        ``lim_2pi``'s per-revolution rescan for long, high-rate recordings.

        Parameters
        ----------
        encoder_deg : array-like
            Wrapped absolute spindle angle in degrees, at *sample_time_s*.
        sample_time_s : array-like
            Timestamps of *encoder_deg*, in seconds (monotonically
            increasing).
        target_time_s : array-like
            Timestamps to resample onto (e.g. the force channel's sample
            times), in seconds.
        phi0_offset_deg : float, default 0.0
            Constant phase offset (degrees) between the encoder zero and
            whatever reference angle the caller needs (e.g. a sensor
            mounting offset). Not yet calibrated for the current setup.

        Returns
        -------
        phi : np.ndarray
            Spindle angular position in radians, wrapped to [0, 2*pi),
            same length as *target_time_s*.
        """
        unwrapped_deg = np.unwrap(np.asarray(encoder_deg, dtype=float), period=360.0)
        interp_deg = np.interp(
            np.asarray(target_time_s, dtype=float),
            np.asarray(sample_time_s, dtype=float),
            unwrapped_deg,
        )
        phi = np.deg2rad(interp_deg + phi0_offset_deg)
        return np.mod(phi, 2.0 * np.pi)

    # ------------------------------------------------------------------
    # 9b.  Angle interpolation onto an arbitrary N-sample grid (upsampling)
    # ------------------------------------------------------------------
    @staticmethod
    def phi_from_encoder_deg(
        encoder_deg: np.ndarray,
        sample_time_s: np.ndarray,
        target_time_s: np.ndarray,
        phi0_offset_deg: float = 0.0,
    ) -> np.ndarray:
        """
        Convenience wrapper around :meth:`phi_from_encoder` that returns the
        wrapped spindle angle in DEGREES [0, 360) instead of radians. Used
        to upsample the spindle-position channel (``axis0_positionact``,
        250 Hz, hardware-wrapped to [0, 360)) onto a higher-rate target grid
        (e.g. a fixed-size ``n_angle_samples`` grid per cut).
        """
        phi_rad = MillingSignalUtils.phi_from_encoder(
            encoder_deg, sample_time_s, target_time_s, np.deg2rad(phi0_offset_deg)
        )
        return np.rad2deg(phi_rad) % 360.0

    # ------------------------------------------------------------------
    # 9c.  Fixed-size resampling helpers (up/down-sampling to N samples)
    # ------------------------------------------------------------------
    @staticmethod
    def resample_linear_to_n(
        signal: np.ndarray,
        time_s: np.ndarray,
        n_target: int,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample *signal* (any sample rate) onto a uniform grid of exactly
        *n_target* samples spanning [t_start, t_end] via linear
        interpolation (``np.interp``). Works for both up- and down-sampling
        (e.g. 20 kHz force -> n_force_samples, or 250 Hz axis position ->
        n_angle_samples for non-angular signals).
        """
        signal = np.asarray(signal, dtype=float)
        time_s = np.asarray(time_s, dtype=float)
        if t_start is None:
            t_start = float(time_s[0])
        if t_end is None:
            t_end = float(time_s[-1])
        t_grid = np.linspace(t_start, t_end, int(n_target))
        resampled = np.interp(t_grid, time_s, signal)
        return t_grid, resampled

    # ------------------------------------------------------------------
    # 9d.  ADC counts -> physical voltage
    # ------------------------------------------------------------------
    @staticmethod
    def adc_counts_to_voltage(
        raw_counts: np.ndarray,
        bits: int = 16,
        v_range: float = 10.0,
        bipolar: bool = True,
    ) -> np.ndarray:
        """
        Convert raw ADC integer counts to physical voltage. Kept as its own
        function so bit depth / voltage range can be changed in one place
        (default: 16-bit ADC, +/-10 V bipolar input, typical Kistler
        charge-amplifier / DAQ front end).

        Bipolar:  voltage = raw_counts / 2**(bits-1) * v_range
        Unipolar: voltage = raw_counts / (2**bits - 1) * v_range
        """
        raw_counts = np.asarray(raw_counts, dtype=float)
        if bipolar:
            full_scale = 2 ** (bits - 1)
            voltage = raw_counts / full_scale * v_range
        else:
            full_scale = 2**bits - 1
            voltage = raw_counts / full_scale * v_range
        return voltage

    # ------------------------------------------------------------------
    # 10.  Spindle torque from measured Fx/Fy and spindle angle
    # ------------------------------------------------------------------
    @staticmethod
    def spindle_torque(
        force_x: np.ndarray,
        force_y: np.ndarray,
        phi: np.ndarray,
        r_tool: float,
        phi0_offset: float = 0.0,
    ) -> np.ndarray:
        """
        Cutting torque about the spindle axis from the two in-plane
        force components (e.g. ``force_0``/``force_1``) measured in the
        fixed dynamometer/workpiece frame, projected onto the tangential
        (torque-producing) direction at spindle angle *phi*.

            Ft(phi) = -Fx*sin(phi + phi0_offset) + Fy*cos(phi + phi0_offset)
            Mc(phi) = r_tool * Ft(phi)

        *phi0_offset* accounts for an unknown/uncalibrated phase
        difference between the encoder's zero and the force sensor's X
        axis; default 0.0 until calibrated (e.g. against the known
        tool-engagement window from the NC log via :meth:`max_Mc`).

        Parameters
        ----------
        force_x, force_y : array-like
            Fx/Fy force components (same units and length as *phi*; note
            raw ``force_0``/``force_1`` are ADC counts, not newtons, so a
            counts-to-newtons calibration must be applied by the caller
            first if physical units are required).
        phi : array-like
            Spindle angular position in radians, e.g. from
            :meth:`phi_from_encoder`.
        r_tool : float
            Tool radius (mm) — read from the machine channel's
            ``toolradius`` field for the true value rather than assumed.
        phi0_offset : float, default 0.0
            Phase offset (radians) applied to *phi* before projection.

        Returns
        -------
        Mc : np.ndarray
            Cutting torque time series, same length as *phi* (N·mm if
            forces are in N and r_tool in mm).
        """
        force_x = np.asarray(force_x, dtype=float)
        force_y = np.asarray(force_y, dtype=float)
        phi = np.asarray(phi, dtype=float)
        ft = -force_x * np.sin(phi + phi0_offset) + force_y * np.cos(phi + phi0_offset)
        return r_tool * ft

    # ------------------------------------------------------------------
    # 11.  Radial force at the cutting edge from measured Fx/Fy and phi
    # ------------------------------------------------------------------
    @staticmethod
    def radial_force(
        force_x: np.ndarray,
        force_y: np.ndarray,
        phi: np.ndarray,
        phi0_offset: float = 0.0,
    ) -> np.ndarray:
        """
        Radial (centre-outward) force component at the cutting edge,
        obtained by projecting the two in-plane dynamometer force
        components onto the radial direction defined by the spindle angle
        *phi*. Complementary to :meth:`spindle_torque`, which projects onto
        the tangential direction.

            Fr(phi) = Fx*cos(phi + phi0_offset) + Fy*sin(phi + phi0_offset)

        Parameters
        ----------
        force_x, force_y : array-like
            Fx/Fy force components (physical units, e.g. N after ADC +
            sensitivity calibration; same length as *phi*).
        phi : array-like
            Spindle angular position in radians, e.g. from
            :meth:`phi_from_encoder` (or degrees converted with
            ``np.deg2rad`` beforehand).
        phi0_offset : float, default 0.0
            Phase offset (radians) applied to *phi* before projection.

        Returns
        -------
        Fr : np.ndarray
            Radial force time series, same length as *phi*.
        """
        force_x = np.asarray(force_x, dtype=float)
        force_y = np.asarray(force_y, dtype=float)
        phi = np.asarray(phi, dtype=float)
        return force_x * np.cos(phi + phi0_offset) + force_y * np.sin(phi + phi0_offset)

    # ------------------------------------------------------------------
    # 12.  Sum of radial force per angular bin (across all revolutions)
    # ------------------------------------------------------------------
    @staticmethod
    def sum_radial_force_by_angle(
        Fr: np.ndarray,
        phi: np.ndarray,
        n_bins: int = 360,
        degrees: bool = True,
    ) -> pd.DataFrame:
        """
        Bin the radial force time series *Fr* by spindle angle *phi* and
        sum the contributions in each bin, across however many spindle
        revolutions the input covers. This is the radial-force analogue of
        :meth:`max_Mc` (which takes the per-revolution MAX of torque); here
        the aggregation is a SUM over all samples/revolutions falling into
        each of the *n_bins* angular bins spanning [0, 360) / [0, 2*pi).

        Used to build the aE-relevant radial-force-vs-angle profile per cut
        (tool-engagement window becomes visible as the angular range with
        non-negligible summed radial force).

        Parameters
        ----------
        Fr : array-like
            Radial force samples (output of :meth:`radial_force`).
        phi : array-like
            Spindle angular position, same length as *Fr*. In degrees if
            *degrees* is True (default), else radians.
        n_bins : int, default 360
            Number of angular bins across one full revolution.
        degrees : bool, default True
            Whether *phi* is given in degrees (True) or radians (False).

        Returns
        -------
        pd.DataFrame
            Columns ``angle_deg`` (bin centre, degrees), ``Fr_sum``
            (summed radial force in the bin), ``Fr_mean`` (mean radial
            force in the bin, i.e. ``Fr_sum / count``), and ``count``
            (number of samples that fell into the bin).
        """
        Fr = np.asarray(Fr, dtype=float)
        phi = np.asarray(phi, dtype=float)
        phi_deg = phi if degrees else np.rad2deg(phi)
        phi_deg = np.mod(phi_deg, 360.0)

        bin_edges = np.linspace(0.0, 360.0, n_bins + 1)
        bin_idx = np.clip(np.digitize(phi_deg, bin_edges) - 1, 0, n_bins - 1)

        Fr_sum = np.bincount(bin_idx, weights=Fr, minlength=n_bins)
        counts = np.bincount(bin_idx, minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            Fr_mean = np.where(counts > 0, Fr_sum / counts, 0.0)

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return pd.DataFrame(
            {
                "angle_deg": bin_centers,
                "Fr_sum": Fr_sum,
                "Fr_mean": Fr_mean,
                "count": counts,
            }
        )