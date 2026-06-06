"""
kienzle_model.py
----------------
Kienzle cutting-force / torque model for flat-end milling operations.

Implements the per-tooth cutting-force formula

    F_{c,i} = k_{c1.1} · a_p · f_z^{1−m_c} · sin(κ)^{m_c} · sin(φ_eff,i)^{1−m_c}

and the resulting spindle torque

    M_c = r_tool · Σ_i F_{c,i}

across a time vector, taking spindle rotation and multi-tooth geometry
into account.

Noise model
-----------
Two independent additive noise sources are supported and can be combined:

    white_noise_amplitude : float
        Peak amplitude of uniform white noise  U(−A, +A).
        Represents broad-band electrical / quantisation noise.

    normal_noise_mean : float
        Mean μ of Gaussian noise  N(μ, σ²).
        A non-zero mean models a systematic offset (e.g. sensor drift).

    normal_noise_std : float
        Standard deviation σ of the Gaussian noise component.

The total noise added to each sample is:

    η(t) = η_white(t) + η_normal(t)

where η_white ~ U(−A, +A) and η_normal ~ N(μ, σ).
Either component is suppressed when its amplitude / std is set to 0.

Usage
-----
    from kienzle_model import KienzleModel

    model = KienzleModel(
        mc                  = 0.25,
        kc11                = 1800.0,       # N/mm²
        phi_in              = 0.0,          # entry angle, rad
        phi_out             = np.pi,        # exit  angle (half immersion)
        fz                  = 0.1,          # mm/tooth
        ap                  = 2.0,          # mm
        omega               = 600.0,        # rpm
        time                = np.linspace(0, 0.5, 5000),
        z                   = 4,
        r_tool              = 10.0,         # mm
        kappa_deg           = 90.0,
        km                  = 1.3,
        white_noise_amplitude = 30.0,       # N·mm  — uniform white noise ±30
        normal_noise_mean   = 5.0,          # N·mm  — systematic offset
        normal_noise_std    = 20.0,         # N·mm  — Gaussian spread
        seed                = 42,
    )

    Mc_clean = model.torque_time_series(add_noise=False)
    Mc_noisy = model.torque_time_series(add_noise=True)
    stan_dat = model.to_stan_data()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# Noise model helper (standalone, importable separately if needed)
# ─────────────────────────────────────────────────────────────────────────────

class NoiseModel:
    """
    Additive signal noise generator combining white (uniform) and
    Gaussian (normal) noise components.

    White noise
    ~~~~~~~~~~~
    Samples drawn from a uniform distribution U(−A, +A), where *A* is the
    peak amplitude.  This models broad-band electrical interference and
    ADC quantisation error, both of which have approximately flat spectral
    density across the measurement bandwidth.

    Gaussian noise
    ~~~~~~~~~~~~~~
    Samples drawn from N(μ, σ²).  A non-zero mean *μ* represents a
    systematic sensor offset or drift component.  The standard deviation
    *σ* captures thermal and amplifier noise, which is well described by a
    normal distribution in practice.

    Parameters
    ----------
    white_amplitude : float
        Peak amplitude *A* of the uniform white-noise component (same
        units as the signal, e.g. N·mm).  Set to 0 to disable.
    normal_mean : float
        Mean *μ* of the Gaussian component.  Default 0 (zero-mean noise).
    normal_std : float
        Standard deviation *σ* of the Gaussian component.  Set to 0 to
        disable.
    seed : int or None
        Random seed for reproducibility.  Passed to
        ``numpy.random.default_rng``.
    """

    def __init__(
        self,
        white_amplitude: float = 0.0,
        normal_mean: float = 0.0,
        normal_std: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        if white_amplitude < 0:
            raise ValueError("white_amplitude must be ≥ 0.")
        if normal_std < 0:
            raise ValueError("normal_std must be ≥ 0.")

        self.white_amplitude = float(white_amplitude)
        self.normal_mean     = float(normal_mean)
        self.normal_std      = float(normal_std)
        self._rng            = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def sample(self, N: int) -> np.ndarray:
        """
        Draw *N* additive noise samples.

        Returns
        -------
        noise : np.ndarray, shape (N,)
            Sum of white and Gaussian components.  Zero array when both
            amplitudes are 0.
        """
        noise = np.zeros(N)

        if self.white_amplitude > 0.0:
            noise += self._rng.uniform(
                -self.white_amplitude, self.white_amplitude, size=N
            )

        if self.normal_std > 0.0 or self.normal_mean != 0.0:
            noise += self._rng.normal(self.normal_mean, self.normal_std, size=N)

        return noise

    # ------------------------------------------------------------------
    def snr_db(self, signal_std: float) -> float:
        """
        Estimate the signal-to-noise ratio in dB, given the standard
        deviation of the clean signal.

        The noise variance is computed analytically:

            Var(η) = Var(white) + Var(normal)
                   = A² / 3  +  σ²

        (The Gaussian mean *μ* shifts the signal baseline but does not
        contribute to noise power.)

        Parameters
        ----------
        signal_std : float
            Standard deviation of the noiseless signal (N·mm).

        Returns
        -------
        snr : float  (dB)
        """
        noise_var = (self.white_amplitude ** 2) / 3.0 + self.normal_std ** 2
        if noise_var == 0.0:
            return float("inf")
        return 10.0 * np.log10(signal_std ** 2 / noise_var)

    # ------------------------------------------------------------------
    @property
    def is_active(self) -> bool:
        """True when at least one noise component has non-zero amplitude."""
        return self.white_amplitude > 0.0 or self.normal_std > 0.0 or self.normal_mean != 0.0

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"NoiseModel("
            f"white_amplitude={self.white_amplitude}, "
            f"normal_mean={self.normal_mean}, "
            f"normal_std={self.normal_std})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# KienzleModel
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KienzleModel:
    """
    Time-domain Kienzle cutting-torque model for flat-end milling.

    Parameters
    ----------
    mc : float
        Kienzle cutting-force exponent (material-specific, dimensionless).
        Typical value for steel: 0.25.
    kc11 : float
        Kienzle specific cutting-force constant k_{c1.1} (N/mm²).
    phi_in : float
        Tool-engagement entry angle in **radians** (start of cut).
        0 rad for up-milling / full-immersion entry.
    phi_out : float
        Tool-engagement exit angle in **radians** (end of cut).
        π rad for half-immersion; 2π for full-immersion.
    fz : float
        Feed per tooth (mm/tooth).
    ap : float
        Axial depth of cut (mm).
    omega : float
        Spindle rotational speed (rpm).
    time : array-like
        Time vector (seconds).
    z : int
        Number of cutting edges (flutes).
    r_tool : float, optional
        Tool radius (mm).  Default 10 mm.
    kappa_deg : float, optional
        Tool cutting-edge angle κ (degrees).  Default 90°.
    km : float, optional
        Motor torque constant k_m (N·mm / A).  Default 1.3.
    phi0_deg : float, optional
        Initial angular offset of the first tooth at t = 0 (degrees).
        Default 0.
    white_noise_amplitude : float, optional
        Peak amplitude *A* of uniform white noise U(−A, +A) added to the
        torque signal.  Default 0 (disabled).
    normal_noise_mean : float, optional
        Mean *μ* of the additive Gaussian noise component.  Default 0.
    normal_noise_std : float, optional
        Standard deviation *σ* of the Gaussian noise component.
        Default 0 (disabled).
    seed : int or None, optional
        Random seed forwarded to the internal NoiseModel for
        reproducible simulations.  Default None (non-deterministic).
    """

    # ── required ──────────────────────────────────────────────────────
    mc: float
    kc11: float
    phi_in: float
    phi_out: float
    fz: float
    ap: float
    omega: float
    time: Any
    z: int

    # ── optional process parameters ───────────────────────────────────
    r_tool: float = 10.0
    kappa_deg: float = 90.0
    km: float = 1.3
    phi0_deg: float = 0.0

    # ── noise parameters ──────────────────────────────────────────────
    white_noise_amplitude: float = 0.0
    normal_noise_mean: float = 0.0
    normal_noise_std: float = 0.0
    seed: Optional[int] = None

    # ── internal fields (not part of constructor signature) ───────────
    _time: np.ndarray          = field(init=False, repr=False)
    _kappa_rad: float          = field(init=False, repr=False)
    _omega_rad: float          = field(init=False, repr=False)
    _tooth_offsets_rad: np.ndarray = field(init=False, repr=False)
    _noise: NoiseModel         = field(init=False, repr=False)

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self._time              = np.asarray(self.time, dtype=float)
        self._kappa_rad         = self.kappa_deg * np.pi / 180.0
        self._omega_rad         = self.omega * 2.0 * np.pi / 60.0
        self._tooth_offsets_rad = np.arange(self.z) * 2.0 * np.pi / self.z
        self._noise             = NoiseModel(
            white_amplitude = self.white_noise_amplitude,
            normal_mean     = self.normal_noise_mean,
            normal_std      = self.normal_noise_std,
            seed            = self.seed,
        )

    # ------------------------------------------------------------------
    # Noise configuration (post-construction)
    # ------------------------------------------------------------------
    def set_noise(
        self,
        white_amplitude: Optional[float] = None,
        normal_mean: Optional[float] = None,
        normal_std: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Update noise parameters without reconstructing the model.

        Only parameters that are explicitly passed (not None) are updated;
        the rest keep their current values.

        Parameters
        ----------
        white_amplitude : float, optional
            New peak amplitude for uniform white noise.
        normal_mean : float, optional
            New mean for the Gaussian noise component.
        normal_std : float, optional
            New standard deviation for the Gaussian noise component.
        seed : int or None, optional
            New random seed (resets the internal RNG).

        Example
        -------
        >>> model.set_noise(white_amplitude=0, normal_std=15.0)
        """
        wa  = white_amplitude if white_amplitude is not None else self._noise.white_amplitude
        nm  = normal_mean     if normal_mean     is not None else self._noise.normal_mean
        ns  = normal_std      if normal_std      is not None else self._noise.normal_std
        sd  = seed            if seed            is not None else self.seed

        # Update dataclass fields for consistency with __str__
        self.white_noise_amplitude = wa
        self.normal_noise_mean     = nm
        self.normal_noise_std      = ns
        self.seed                  = sd
        self._noise                = NoiseModel(wa, nm, ns, sd)

    # ------------------------------------------------------------------
    # Core single-instant calculation
    # ------------------------------------------------------------------
    def _Fci_at_phi(self, phi_spindle: float) -> np.ndarray:
        """
        Cutting force on each tooth at a single spindle angle (noiseless).

        Parameters
        ----------
        phi_spindle : float
            Current spindle angle in radians (reference tooth).

        Returns
        -------
        Fci : np.ndarray, shape (z,)
        """
        Fci = np.zeros(self.z)
        phi_in_deg  = np.degrees(self.phi_in)
        phi_out_deg = np.degrees(self.phi_out)

        for i in range(self.z):
            phi_eff     = (phi_spindle + self._tooth_offsets_rad[i]) % (2.0 * np.pi)
            phi_eff_deg = np.degrees(phi_eff)

            if phi_in_deg <= phi_eff_deg <= phi_out_deg:
                Fci[i] = (
                    self.kc11
                    * self.ap
                    * self.fz ** (1.0 - self.mc)
                    * np.sin(self._kappa_rad) ** self.mc
                    * np.sin(phi_eff) ** (1.0 - self.mc)
                )
        return Fci

    # ------------------------------------------------------------------
    # Per-sample torque — core public method
    # ------------------------------------------------------------------
    def cutting_moment_at_phi(self, phi_spindle: float) -> Dict[str, Any]:
        """
        Cutting forces and torque at a single spindle angle (noiseless).

        Returns
        -------
        dict : keys ``Fci``, ``Mc``, ``phi_eff_deg``
        """
        Fci = self._Fci_at_phi(phi_spindle)
        Mc  = float(np.sum(Fci) * self.r_tool)
        phi_eff_deg = np.degrees(
            (phi_spindle + self._tooth_offsets_rad) % (2.0 * np.pi)
        )
        return {"Fci": Fci, "Mc": Mc, "phi_eff_deg": phi_eff_deg}

    # ------------------------------------------------------------------
    # Time-series: noiseless clean signal
    # ------------------------------------------------------------------
    def _torque_clean(self, phi_vec: np.ndarray) -> np.ndarray:
        """Noiseless torque vector for a given angular-position array."""
        return np.array(
            [np.sum(self._Fci_at_phi(p)) * self.r_tool for p in phi_vec]
        )

    # ------------------------------------------------------------------
    # Time-series: with optional noise
    # ------------------------------------------------------------------
    def torque_time_series(
        self,
        phi_ext: Optional[np.ndarray] = None,
        add_noise: bool = True,
    ) -> np.ndarray:
        """
        Compute the spindle torque over the full time vector.

        Parameters
        ----------
        phi_ext : array-like, optional
            External angular-position vector (radians), same length as
            *self.time*.  Overrides the internally computed angle when
            supplied (useful when feeding measured φ from the CNC
            controller).
        add_noise : bool
            Whether to add the configured noise signal.  Default True.
            Pass ``False`` to retrieve the noiseless Kienzle torque.

        Returns
        -------
        Mc_vec : np.ndarray  (N·mm)
        """
        N = len(self._time)

        if phi_ext is not None:
            phi_vec = np.asarray(phi_ext, dtype=float)
            if len(phi_vec) != N:
                raise ValueError(
                    f"phi_ext length ({len(phi_vec)}) must match "
                    f"time length ({N})."
                )
        else:
            phi0_rad = self.phi0_deg * np.pi / 180.0
            phi_vec  = (phi0_rad + self._omega_rad * self._time) % (2.0 * np.pi)

        Mc_vec = self._torque_clean(phi_vec)

        if add_noise and self._noise.is_active:
            Mc_vec = Mc_vec + self._noise.sample(N)

        return Mc_vec

    # ------------------------------------------------------------------
    # Noise-only time series (diagnostic)
    # ------------------------------------------------------------------
    def noise_time_series(self, N: Optional[int] = None) -> np.ndarray:
        """
        Return a standalone noise sample of length *N* (or
        ``len(self.time)`` when *N* is None).

        Useful for inspecting noise character independently of the
        process signal, e.g. to verify amplitude and distribution
        before running a simulation.

        Parameters
        ----------
        N : int, optional
            Number of samples.  Defaults to the length of *self.time*.

        Returns
        -------
        noise : np.ndarray
        """
        if N is None:
            N = len(self._time)
        return self._noise.sample(N)

    # ------------------------------------------------------------------
    # SNR helper
    # ------------------------------------------------------------------
    def snr_db(self, phi_ext: Optional[np.ndarray] = None) -> float:
        """
        Signal-to-noise ratio (dB) of the simulated torque signal.

        The clean-signal standard deviation is estimated from
        :meth:`torque_time_series` called with ``add_noise=False``.

        Parameters
        ----------
        phi_ext : array-like, optional
            External angular-position vector forwarded to the torque
            computation.

        Returns
        -------
        snr : float  (dB).  Returns ``inf`` when noise is disabled.
        """
        Mc_clean = self.torque_time_series(phi_ext=phi_ext, add_noise=False)
        return self._noise.snr_db(float(np.std(Mc_clean)))

    # ------------------------------------------------------------------
    # Angular position from time
    # ------------------------------------------------------------------
    def phi_time_series(self) -> np.ndarray:
        """
        Spindle angular position [0, 2π) for each time sample.

        Returns
        -------
        phi_vec : np.ndarray  (radians)
        """
        phi0_rad = self.phi0_deg * np.pi / 180.0
        return (phi0_rad + self._omega_rad * self._time) % (2.0 * np.pi)

    # ------------------------------------------------------------------
    # Spindle current
    # ------------------------------------------------------------------
    def current_time_series(
        self,
        phi_ext: Optional[np.ndarray] = None,
        add_noise: bool = True,
    ) -> np.ndarray:
        """
        Equivalent spindle current  I_q(t) = M_c(t) / k_m  (A).

        Parameters
        ----------
        phi_ext : array-like, optional
            External angular-position vector.
        add_noise : bool
            Forwarded to :meth:`torque_time_series`.

        Returns
        -------
        Iq_vec : np.ndarray  (A)
        """
        return self.torque_time_series(phi_ext=phi_ext, add_noise=add_noise) / self.km

    # ------------------------------------------------------------------
    # Summary DataFrame
    # ------------------------------------------------------------------
    def to_dataframe(
        self,
        phi_ext: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Tidy DataFrame with clean and noisy torque columns side-by-side.

        Returns
        -------
        pd.DataFrame with columns:
            ``time``     – seconds
            ``phi``      – spindle angle (rad)
            ``Mc_clean`` – noiseless Kienzle torque (N·mm)
            ``Mc``       – torque with noise applied (N·mm)
            ``noise``    – noise component alone (N·mm)
            ``Iq_clean`` – noiseless spindle current (A)
            ``Iq``       – noisy spindle current (A)
        """
        phi_vec  = (
            np.asarray(phi_ext, dtype=float)
            if phi_ext is not None
            else self.phi_time_series()
        )
        Mc_clean = self.torque_time_series(phi_ext=phi_ext, add_noise=False)
        Mc_noisy = self.torque_time_series(phi_ext=phi_ext, add_noise=True)

        return pd.DataFrame(
            {
                "time":     self._time,
                "phi":      phi_vec,
                "Mc_clean": Mc_clean,
                "Mc":       Mc_noisy,
                "noise":    Mc_noisy - Mc_clean,
                "Iq_clean": Mc_clean / self.km,
                "Iq":       Mc_noisy / self.km,
            }
        )

    # ------------------------------------------------------------------
    # Stan data dictionary
    # ------------------------------------------------------------------
    def to_stan_data(
        self,
        Mc_observed: Optional[np.ndarray] = None,
        phi_observed: Optional[np.ndarray] = None,
        m_kc: float = 2306.0,
        sd_kc: float = 977.0,
        alpha_mc: Optional[float] = None,
        beta_mc: Optional[float] = None,
        mc_mean_prior: float = 0.25,
        mc_var_prior: float = 0.02 ** 2,
    ) -> Dict[str, Any]:
        """
        Build the data dictionary for CmdStanPy ``model.sample(data=...)``.

        Parameters
        ----------
        Mc_observed : array-like, optional
            Measured torque vector (N·mm).  When None, synthetic noisy
            data from :meth:`torque_time_series` are used.
        phi_observed : array-like, optional
            Measured angular-position vector (rad).
        m_kc, sd_kc : float
            Prior mean and standard deviation for k_{c1.1}.
        alpha_mc, beta_mc : float, optional
            Beta prior shapes for m_c.  Computed from *mc_mean_prior* and
            *mc_var_prior* when not supplied.
        mc_mean_prior, mc_var_prior : float
            Moments used to derive Beta prior shapes when alpha / beta are
            not passed explicitly.

        Returns
        -------
        dict
        """
        if alpha_mc is None or beta_mc is None:
            from kienzle_utils import MillingSignalUtils as MSU
            alpha_mc, beta_mc = MSU.beta_dist_param(mc_mean_prior, mc_var_prior)

        if Mc_observed is None:
            phi_obs = self.phi_time_series()
            Mc_obs  = self.torque_time_series(add_noise=True)
        else:
            Mc_obs  = np.asarray(Mc_observed, dtype=float)
            phi_obs = (
                np.asarray(phi_observed, dtype=float)
                if phi_observed is not None
                else self.phi_time_series()
            )

        return {
            "k":        int(len(Mc_obs)),
            "Mc":       Mc_obs.tolist(),
            "phi":      phi_obs.tolist(),
            "ap":       float(self.ap),
            "fz":       float(self.fz),
            "z":        int(self.z),
            "rtool":    float(self.r_tool),
            "kappa":    float(self._kappa_rad),
            "m_kc":     float(m_kc),
            "tau_kc":   float(1.0 / sd_kc ** 2),
            "alpha_mc": float(alpha_mc),
            "beta_mc":  float(beta_mc),
        }

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------
    def __str__(self) -> str:
        snr = self.snr_db()
        snr_str = f"{snr:.1f} dB" if snr != float("inf") else "∞ (no noise)"
        lines = [
            "KienzleModel",
            f"  mc                    = {self.mc}",
            f"  kc11                  = {self.kc11} N/mm²",
            f"  phi_in                = {np.degrees(self.phi_in):.1f}°",
            f"  phi_out               = {np.degrees(self.phi_out):.1f}°",
            f"  fz                    = {self.fz} mm/tooth",
            f"  ap                    = {self.ap} mm",
            f"  omega                 = {self.omega} rpm",
            f"  z                     = {self.z} teeth",
            f"  r_tool                = {self.r_tool} mm",
            f"  kappa                 = {self.kappa_deg}°",
            f"  km                    = {self.km} N·mm/A",
            f"  time pts              = {len(self._time)} samples",
            "  ── Noise ──────────────────────────────────────",
            f"  white_noise_amplitude = {self.white_noise_amplitude} N·mm  (±A uniform)",
            f"  normal_noise_mean     = {self.normal_noise_mean} N·mm",
            f"  normal_noise_std      = {self.normal_noise_std} N·mm",
            f"  seed                  = {self.seed}",
            f"  estimated SNR         = {snr_str}",
        ]
        return "\n".join(lines)
