# Kienzle Milling Model — Python Library Documentation

**Author:** Thomas M. Rudolf  
**Version:** 1.1  
**Modules:** `kienzle_model.py` · `kienzle_utils.py`

---

## Table of Contents

1. [Overview](#1-overview)
2. [Physical Model](#2-physical-model)
3. [Noise Model](#3-noise-model)
4. [Module: `kienzle_model`](#4-module-kienzle_model)
   - 4.1 [Class `NoiseModel`](#41-class-noisemodel)
   - 4.2 [Class `KienzleModel`](#42-class-kienzlemodel)
5. [Module: `kienzle_utils`](#5-module-kienzle_utils)
   - 5.1 [Class `MillingSignalUtils`](#51-class-millingsignalutils)
6. [Stan Integration](#6-stan-integration)
7. [Quick-Start Examples](#7-quick-start-examples)
8. [Parameter Reference Tables](#8-parameter-reference-tables)

---

## 1. Overview

This library provides a time-domain simulation of the **Kienzle cutting-force model** for flat-end milling operations, together with signal-processing utilities for analysing CNC spindle-torque data. The primary application is Bayesian parameter estimation of the Kienzle material constants *m*_c and *k*_c1.1 from CNC controller-internal spindle drive signals, using CmdStanPy as the inference back-end.

The library consists of two importable Python modules:

| Module | Main export | Purpose |
|---|---|---|
| `kienzle_model.py` | `KienzleModel`, `NoiseModel` | Process simulation, noise injection, Stan data preparation |
| `kienzle_utils.py` | `MillingSignalUtils` | Signal filtering, angle wrapping, peak extraction, prior computation |

Both modules depend only on `numpy` and `pandas`.

---

## 2. Physical Model

### 2.1 Per-tooth cutting force

The cutting force on tooth *i* at spindle angle φ is given by the Kienzle formula:

$$F_{c,i} = k_{c1.1} \cdot a_p \cdot f_z^{\,1 - m_c} \cdot \sin(\kappa)^{m_c} \cdot \sin(\varphi_{\text{eff},i})^{\,1 - m_c}$$

where the effective angle of tooth *i* is:

$$\varphi_{\text{eff},i} = \left(\varphi + (i-1)\,\frac{2\pi}{z}\right) \bmod 2\pi, \quad i = 1, \dots, z$$

The force is non-zero only within the engagement window:

$$\varphi_{\text{in}} \leq \varphi_{\text{eff},i} \leq \varphi_{\text{out}}$$

### 2.2 Spindle torque

The total cutting torque at spindle angle φ is:

$$M_c(\varphi) = r_{\text{tool}} \sum_{i=1}^{z} F_{c,i}(\varphi)$$

### 2.3 Spindle current

The torque-generating spindle current is recovered via the motor torque constant *k*_m:

$$I_q(t) = \frac{M_c(t)}{k_m}$$

### 2.4 Symbol definitions

| Symbol | Parameter | Unit |
|---|---|---|
| *k*_c1.1 | Kienzle specific cutting-force constant | N/mm² |
| *m*_c | Kienzle cutting-force exponent | — |
| *a*_p | Axial depth of cut | mm |
| *f*_z | Feed per tooth | mm/tooth |
| κ | Tool cutting-edge angle | rad |
| *z* | Number of cutting edges (flutes) | — |
| φ | Spindle angular position (reference tooth) | rad |
| φ_in | Engagement entry angle | rad |
| φ_out | Engagement exit angle | rad |
| *r*_tool | Tool radius | mm |
| *k*_m | Motor torque constant | N·mm / A |
| *I*_q | Torque-generating spindle current | A |

---

## 3. Noise Model

The library implements two independent additive noise sources that can be used individually or in combination. The total noise at each sample is:

$$\eta(t) = \eta_{\text{white}}(t) + \eta_{\text{normal}}(t)$$

### 3.1 White noise

$$\eta_{\text{white}} \sim \mathcal{U}(-A,\, +A)$$

White noise drawn from a uniform distribution with peak amplitude *A* models broad-band electrical interference and ADC quantisation error. Both phenomena have approximately flat power spectral density across the measurement bandwidth, making the uniform distribution an appropriate model.

### 3.2 Gaussian noise

$$\eta_{\text{normal}} \sim \mathcal{N}(\mu,\, \sigma^2)$$

Gaussian noise models thermal noise and amplifier noise, which — by the central-limit theorem — follows a normal distribution when many small independent sources are superimposed. A non-zero mean *μ* represents a systematic sensor offset or slow drift.

### 3.3 Noise variance and SNR

The total noise variance is:

$$\operatorname{Var}(\eta) = \frac{A^2}{3} + \sigma^2$$

The signal-to-noise ratio in dB is:

$$\mathrm{SNR} = 10 \log_{10}\!\left(\frac{\operatorname{Var}(M_c^{\text{clean}})}{\operatorname{Var}(\eta)}\right)$$

Note that the Gaussian mean *μ* shifts the signal baseline but does not contribute to noise power.

---

## 4. Module: `kienzle_model`

```python
from kienzle_model import KienzleModel, NoiseModel
```

---

### 4.1 Class `NoiseModel`

A standalone additive noise generator. It can be imported separately for diagnostic use or integrated automatically through `KienzleModel`.

#### Constructor

```python
NoiseModel(
    white_amplitude: float = 0.0,
    normal_mean:     float = 0.0,
    normal_std:      float = 0.0,
    seed:            int | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `white_amplitude` | float | 0.0 | Peak amplitude *A* of uniform white noise U(−A, +A). Set to 0 to disable. |
| `normal_mean` | float | 0.0 | Mean *μ* of the Gaussian noise component. Non-zero values model systematic sensor offset. |
| `normal_std` | float | 0.0 | Standard deviation *σ* of the Gaussian noise. Set to 0 to disable. |
| `seed` | int \| None | None | Random seed passed to `numpy.random.default_rng`. Use for reproducible simulations. |

#### Methods

##### `sample(N) → np.ndarray`

Draw *N* independent noise samples.

```python
noise = NoiseModel(white_amplitude=30.0, normal_std=15.0, seed=0).sample(1000)
```

Returns an array of shape `(N,)` containing the sum of white and Gaussian components. Returns a zero array when all amplitudes are zero.

---

##### `snr_db(signal_std) → float`

Compute the analytical signal-to-noise ratio in dB.

```python
snr = noise_model.snr_db(signal_std=956.3)   # → e.g. 31.2
```

| Parameter | Description |
|---|---|
| `signal_std` | Standard deviation of the noiseless signal (same units as noise). |

Returns `float('inf')` when noise is fully disabled.

---

##### `is_active` (property)

Returns `True` when at least one noise component has non-zero amplitude.

```python
if model._noise.is_active:
    print("Noise is enabled")
```

---

### 4.2 Class `KienzleModel`

The main simulation class. Implemented as a Python `dataclass`; all parameters are set in the constructor.

#### Constructor

```python
KienzleModel(
    # ── required ──────────────────────────
    mc:    float,
    kc11:  float,
    phi_in:  float,
    phi_out: float,
    fz:    float,
    ap:    float,
    omega: float,
    time:  array-like,
    z:     int,
    # ── optional process ──────────────────
    r_tool:    float = 10.0,
    kappa_deg: float = 90.0,
    km:        float = 1.3,
    phi0_deg:  float = 0.0,
    # ── noise ─────────────────────────────
    white_noise_amplitude: float = 0.0,
    normal_noise_mean:     float = 0.0,
    normal_noise_std:      float = 0.0,
    seed: int | None = None,
)
```

All nine required parameters mirror the Kienzle model variables defined in Section 2. The four noise parameters map directly to `NoiseModel` and are described in Section 3.

#### Methods

---

##### `torque_time_series(phi_ext=None, add_noise=True) → np.ndarray`

Compute the cutting torque M_c(t) over the full time vector.

```python
Mc_clean = model.torque_time_series(add_noise=False)
Mc_noisy = model.torque_time_series(add_noise=True)

# Feed measured angular position from the CNC controller:
Mc_measured = model.torque_time_series(phi_ext=phi_from_controller)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `phi_ext` | array-like \| None | None | External angular-position vector in radians, same length as `time`. Overrides the internally computed angle when supplied. Must be wrapped to [0, 2π). |
| `add_noise` | bool | True | Whether to add the configured noise. Pass `False` to retrieve the noiseless Kienzle torque. |

Returns `np.ndarray` of shape `(N,)` in N·mm.

---

##### `current_time_series(phi_ext=None, add_noise=True) → np.ndarray`

Equivalent spindle current I_q(t) = M_c(t) / k_m.

```python
Iq = model.current_time_series()   # shape (N,), units A
```

Parameters identical to `torque_time_series`. Returns `np.ndarray` in amperes.

---

##### `phi_time_series() → np.ndarray`

Spindle angular position [0, 2π) computed from `omega`, `time`, and `phi0_deg`.

```python
phi = model.phi_time_series()   # shape (N,), units rad
```

---

##### `noise_time_series(N=None) → np.ndarray`

Return a standalone noise sample for diagnostic inspection, independent of the process signal.

```python
noise = model.noise_time_series(N=2000)
```

Useful for verifying noise amplitude and distribution before a full simulation run. Defaults to `len(self.time)` when *N* is not supplied.

---

##### `snr_db(phi_ext=None) → float`

Signal-to-noise ratio in dB. The clean-signal standard deviation is estimated numerically from `torque_time_series(add_noise=False)`.

```python
print(f"SNR = {model.snr_db():.1f} dB")
```

Returns `float('inf')` when noise is disabled.

---

##### `cutting_moment_at_phi(phi_spindle) → dict`

Compute cutting forces and torque at a single spindle angle. Noiseless.

```python
result = model.cutting_moment_at_phi(phi_spindle=0.523)
# result["Fci"]         → np.ndarray, shape (z,)
# result["Mc"]          → float  (N·mm)
# result["phi_eff_deg"] → np.ndarray, shape (z,)
```

---

##### `set_noise(white_amplitude=None, normal_mean=None, normal_std=None, seed=None)`

Update noise parameters after construction without rebuilding the model. Only explicitly passed arguments are changed; all others keep their current values.

```python
model.set_noise(white_amplitude=0.0, normal_std=15.0)
model.set_noise(normal_mean=0.0)      # zero out the offset only
model.set_noise(seed=99)              # reset RNG seed
```

---

##### `to_dataframe(phi_ext=None) → pd.DataFrame`

Return a tidy DataFrame with clean and noisy signals side by side.

```python
df = model.to_dataframe()
```

| Column | Unit | Description |
|---|---|---|
| `time` | s | Time vector |
| `phi` | rad | Spindle angular position |
| `Mc_clean` | N·mm | Noiseless Kienzle torque |
| `Mc` | N·mm | Torque with noise applied |
| `noise` | N·mm | Noise component alone (`Mc − Mc_clean`) |
| `Iq_clean` | A | Noiseless spindle current |
| `Iq` | A | Noisy spindle current |

---

##### `to_stan_data(...) → dict`

Build the data dictionary for `cmdstanpy.CmdStanModel.sample(data=...)`.

```python
stan_data = model.to_stan_data(
    Mc_observed   = measured_torque,   # array or None → uses simulation
    phi_observed  = measured_phi,
    m_kc          = 2306.0,            # prior mean for kc11
    sd_kc         = 977.0,             # prior std  for kc11
    mc_mean_prior = 0.25,              # Beta prior mean for mc
    mc_var_prior  = 0.02**2,           # Beta prior variance for mc
)
```

| Key | Type | Description |
|---|---|---|
| `k` | int | Number of observations |
| `Mc` | list[float] | Observed torque vector (N·mm) |
| `phi` | list[float] | Observed angular position (rad) |
| `ap` | float | Axial depth of cut (mm) |
| `fz` | float | Feed per tooth (mm/tooth) |
| `z` | int | Number of cutting edges |
| `rtool` | float | Tool radius (mm) |
| `kappa` | float | Tool cutting-edge angle (rad) |
| `m_kc` | float | Prior mean for *k*_c1.1 |
| `tau_kc` | float | Prior precision = 1 / sd_kc² |
| `alpha_mc` | float | Beta prior α for *m*_c |
| `beta_mc` | float | Beta prior β for *m*_c |

When `Mc_observed` is `None`, the method calls `torque_time_series(add_noise=True)` to generate synthetic observations including the configured noise.

---

## 5. Module: `kienzle_utils`

```python
from kienzle_utils import MillingSignalUtils as MSU
```

All methods are `@staticmethod`; no instantiation is required.

---

### 5.1 Class `MillingSignalUtils`

---

#### `lim_2pi(phi) → (np.ndarray, list[int])`

Wrap a cumulative angular-position vector to [0, 2π) by subtracting 2π at each revolution boundary.

```python
phi_wrapped, id_reduce = MSU.lim_2pi(phi_cumulative)
```

| Parameter | Description |
|---|---|
| `phi` | Monotonically increasing cumulative angle in radians, as returned by `np.cumsum(omega_rad * dt)`. |

Returns the wrapped array and a list of zero-based boundary indices. These indices are passed directly to `max_Mc`.

---

#### `maf(in_signal, n) → np.ndarray`

Causal moving-average filter with window length *n*.

```python
Mc_smooth = MSU.maf(Mc_raw, n=14)
```

The first *n* − 1 samples use an expanding window (matching the original R implementation). The filter introduces a lag of approximately *n*/2 samples.

---

#### `maf_std_scaler(in_signal, n) → np.ndarray`

Z-score normalisation using a causal moving mean and moving standard deviation.

```python
Mc_scaled = MSU.maf_std_scaler(Mc_raw, n=14)
```

Samples where the moving standard deviation is zero are returned as 0. Uses Bessel's correction (ddof=1) for the standard deviation.

---

#### `max_Mc(id_reduce, Mc, phi) → pd.DataFrame`

Find the maximum cutting torque and its angular position within each complete spindle revolution.

```python
df_peaks = MSU.max_Mc(id_reduce, Mc_array, phi_wrapped)
# df_peaks.columns → ["McMax", "phiMcMax"]
```

| Parameter | Description |
|---|---|
| `id_reduce` | Revolution-boundary indices from `lim_2pi`. |
| `Mc` | Cutting-torque array (N·mm). |
| `phi` | Wrapped angular-position array (rad). |

Returns a `pd.DataFrame` with columns `McMax` (peak torque per revolution, N·mm) and `phiMcMax` (angular position of the peak, rad).

---

#### `beta_dist_param(mean, var) → (float, float)`

Compute Beta distribution shape parameters α and β from a desired mean and variance using moment matching.

$$\alpha = \frac{\mu^2 (1-\mu)}{\sigma^2} - \mu \qquad \beta = \frac{\alpha(1-\mu)}{\mu}$$

```python
alpha, beta = MSU.beta_dist_param(mean=0.25, var=0.02**2)
# → (116.94, 350.81)
```

Raises `ValueError` when the requested variance is not achievable for the given mean (i.e. when σ² ≥ μ(1 − μ)).

---

#### `correct_no_load(signal, time, t_start, t_end) → (np.ndarray, float)`

Subtract the mean no-load (friction + inertia) torque from a spindle current or torque signal.

```python
Mc_corrected, offset = MSU.correct_no_load(
    Mc_raw, time, t_start=0.0, t_end=2.0
)
```

The no-load mean is estimated from the interval [t_start, t_end], during which the spindle runs freely without tool engagement.

| Returns | Description |
|---|---|
| `corrected` | Signal with no-load mean subtracted. |
| `no_load_mean` | The mean value subtracted (for logging / reporting). |

---

## 6. Stan Integration

The `to_stan_data()` method produces a dictionary that maps directly onto the variables expected by `STAN_kienzle_cont.stan`. A minimal CmdStanPy workflow:

```python
import numpy as np
from cmdstanpy import CmdStanModel
from kienzle_model import KienzleModel
from kienzle_utils import MillingSignalUtils as MSU

# 1.  Build model and generate (or load) observations
time  = np.linspace(0, 0.5, 2500)
model = KienzleModel(
    mc=0.25, kc11=1800.0, phi_in=0.0, phi_out=np.pi,
    fz=0.1, ap=2.0, omega=600.0, time=time, z=4,
    normal_noise_std=20.0, seed=42,
)

# 2.  Wrap measured angular position (from CNC controller)
phi_cum      = model._omega_rad * time
phi_wrapped, _ = MSU.lim_2pi(phi_cum)

# 3.  Prepare Stan data dictionary
stan_data = model.to_stan_data(
    Mc_observed  = model.torque_time_series(),
    phi_observed = phi_wrapped,
    m_kc=2306.0, sd_kc=977.0,
    mc_mean_prior=0.25, mc_var_prior=0.02**2,
)

# 4.  Compile and sample
stan_model = CmdStanModel(stan_file="STAN_kienzle_cont.stan")
fit = stan_model.sample(
    data=stan_data,
    chains=4, parallel_chains=4,
    iter_warmup=1000, iter_sampling=2000,
)

print(fit.summary())
```

---

## 7. Quick-Start Examples

### Example A — Noiseless simulation

```python
import numpy as np
from kienzle_model import KienzleModel

time  = np.linspace(0, 0.5, 5000)
model = KienzleModel(
    mc=0.25, kc11=1800.0,
    phi_in=0.0, phi_out=np.pi,
    fz=0.1, ap=2.0, omega=600.0,
    time=time, z=4,
)
Mc = model.torque_time_series(add_noise=False)
```

### Example B — Combined white and Gaussian noise

```python
model = KienzleModel(
    mc=0.25, kc11=1800.0,
    phi_in=0.0, phi_out=np.pi,
    fz=0.1, ap=2.0, omega=600.0,
    time=time, z=4,
    white_noise_amplitude=30.0,   # ±30 N·mm uniform
    normal_noise_mean=5.0,        # +5 N·mm systematic offset
    normal_noise_std=20.0,        # σ = 20 N·mm
    seed=42,
)
df = model.to_dataframe()
print(f"SNR = {model.snr_db():.1f} dB")
```

### Example C — Update noise after construction

```python
model.set_noise(white_amplitude=0.0, normal_std=10.0, seed=99)
Mc_new = model.torque_time_series()
```

### Example D — Signal filtering and peak extraction

```python
from kienzle_utils import MillingSignalUtils as MSU

phi_cum       = model._omega_rad * time
phi_w, id_red = MSU.lim_2pi(phi_cum)
Mc_smooth     = MSU.maf(df["Mc"].values, n=14)
df_peaks      = MSU.max_Mc(id_red, df["Mc"].values, phi_w)
```

### Example E — Beta prior computation

```python
alpha, beta = MSU.beta_dist_param(mean=0.25, var=0.02**2)
print(f"Beta prior: α = {alpha:.2f}, β = {beta:.2f}")
# → Beta prior: α = 116.94, β = 350.81
```

---

## 8. Parameter Reference Tables

### `KienzleModel` — full parameter reference

| Parameter | Required | Default | Unit | Description |
|---|---|---|---|---|
| `mc` | ✓ | — | — | Kienzle exponent *m*_c |
| `kc11` | ✓ | — | N/mm² | Specific cutting-force constant *k*_c1.1 |
| `phi_in` | ✓ | — | rad | Engagement entry angle |
| `phi_out` | ✓ | — | rad | Engagement exit angle |
| `fz` | ✓ | — | mm/tooth | Feed per tooth |
| `ap` | ✓ | — | mm | Axial depth of cut |
| `omega` | ✓ | — | rpm | Spindle rotational speed |
| `time` | ✓ | — | s | Time vector |
| `z` | ✓ | — | — | Number of cutting edges |
| `r_tool` | — | 10.0 | mm | Tool radius |
| `kappa_deg` | — | 90.0 | ° | Tool cutting-edge angle κ |
| `km` | — | 1.3 | N·mm/A | Motor torque constant |
| `phi0_deg` | — | 0.0 | ° | Initial angular offset at *t* = 0 |
| `white_noise_amplitude` | — | 0.0 | N·mm | Peak amplitude of uniform white noise |
| `normal_noise_mean` | — | 0.0 | N·mm | Mean of Gaussian noise component |
| `normal_noise_std` | — | 0.0 | N·mm | Standard deviation of Gaussian noise |
| `seed` | — | None | — | Random seed for reproducibility |

### Engagement angle reference

| Milling condition | `phi_in` | `phi_out` |
|---|---|---|
| Full immersion | 0 rad (0°) | 2π rad (360°) |
| Half immersion (down-milling) | 0 rad (0°) | π rad (180°) |
| Quarter immersion | 0 rad (0°) | π/2 rad (90°) |
| Custom radial engagement *a*_e/*D* | 0 rad | arccos(1 − 2·*a*_e/*D*) |

### Typical Kienzle constants for common workpiece materials

| Material | *k*_c1.1 (N/mm²) | *m*_c |
|---|---|---|
| Steel (structural, 42CrMo4) | 1 800 – 2 500 | 0.22 – 0.30 |
| Aluminium alloys | 700 – 1 000 | 0.20 – 0.28 |
| Cast iron (GG25) | 900 – 1 400 | 0.20 – 0.26 |
| Titanium (Ti-6Al-4V) | 2 000 – 2 800 | 0.25 – 0.35 |

---

*End of documentation.*
