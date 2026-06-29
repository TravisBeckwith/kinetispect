# physio_noise.py — Physiological noise for fMRS simulation
#
# Extends FSL-MRS synthetic_spectra_from_model output with realistic
# per-acquisition noise sources not modeled by the FSL-MRS generator.
#
# Input/output: list of fsl_mrs.core.MRS objects (mrs_list), one per
# acquisition, as returned by synthetic_spectra_from_model().
#
# Author: Travis Beckwith / KinetiSpect project
# Reference: Just (2024) Front Neurosci; Clarke (2024) MRM

import numpy as np


def add_b0_drift(
    mrs_list: list,
    drift_rate_hz_per_min: float = 1.0,
    random_walk_std_hz: float = 0.3,
    tr_s: float = 2.0,
    seed: int | None = None,
) -> list:
    """Apply per-acquisition B0 frequency drift to a list of MRS objects.

    Models two drift components observed in vivo:
      - Linear drift: slow scanner field creep, typically 0.5–2 Hz/min at 3T.
      - Random walk: stochastic jump-to-jump variation, typically ±0.3 Hz SD.

    Both shift the FID phase as: FID(t) *= exp(i·2π·Δf·t), where t is the
    intra-acquisition time axis and Δf is the per-acquisition frequency offset.

    Args:
        mrs_list: List of MRS objects from synthetic_spectra_from_model().
        drift_rate_hz_per_min: Linear B0 drift rate in Hz per minute.
            Typical range: 0.5–2.0 Hz/min at 3T. Default 1.0 Hz/min.
        random_walk_std_hz: SD of per-acquisition random walk step in Hz.
            Each acquisition draws an independent step; cumulative sum gives
            the random walk trajectory. Default 0.3 Hz.
        tr_s: Repetition time in seconds. Used to convert acquisition index
            to elapsed time. Default 2.0 s.
        seed: Random seed for reproducibility. Default None (random).

    Returns:
        mrs_list with FIDs modified in place. Also returns freq_shifts_hz
        as a second return value for use in downstream nuisance correction.

    Example:
        mrs_list, vm, params = synthetic_spectra_from_model(...)
        mrs_list, freq_shifts = add_b0_drift(mrs_list, drift_rate_hz_per_min=1.5)
    """
    rng = np.random.default_rng(seed)
    n_acq = len(mrs_list)

    # Per-acquisition elapsed time in minutes
    t_min = np.arange(n_acq) * tr_s / 60.0

    # Linear component: grows monotonically across the scan
    linear_drift_hz = drift_rate_hz_per_min * t_min

    # Random walk component: cumulative sum of zero-mean Gaussian steps
    steps = rng.normal(loc=0.0, scale=random_walk_std_hz, size=n_acq)
    random_walk_hz = np.cumsum(steps)

    freq_shifts_hz = linear_drift_hz + random_walk_hz

    for n, (mrs, delta_f) in enumerate(zip(mrs_list, freq_shifts_hz)):
        n_pts = len(mrs.FID)
        dwell_time = 1.0 / mrs.bandwidth          # seconds per point
        t = np.arange(n_pts) * dwell_time         # intra-acquisition time axis

        # Phase ramp in time domain = frequency shift in spectrum
        phase_ramp = np.exp(1j * 2 * np.pi * delta_f * t)
        mrs.FID = mrs.FID * phase_ramp

    return mrs_list, freq_shifts_hz


def add_physiological_noise(
    mrs_list: list,
    cardiac_bpm: float = 72.0,
    cardiac_phase_amp_hz: float = 1.0,
    resp_bpm: float = 15.0,
    resp_phase_amp_hz: float = 2.5,
    tr_s: float = 2.0,
    seed: int | None = None,
) -> list:
    """Apply cardiac and respiratory phase modulation to a list of MRS objects.

    Models the two dominant physiological noise sources in fMRS:

    1. Cardiac (~1.2 Hz at rest): Pulsatile blood flow causes ~1 Hz peak-to-peak
       frequency modulation of the water and metabolite signals. At TR=2s the
       cardiac cycle is aliased, producing structured low-frequency noise.

    2. Respiratory (~0.25 Hz at rest): Chest movement shifts B0 by 2–5 Hz
       peak-to-peak, producing slow sinusoidal frequency modulation across
       the time series.

    Both are modeled as sinusoidal phase modulation of the FID with random
    initial phases (to avoid assuming phase coherence across subjects).

    Args:
        mrs_list: List of MRS objects from synthetic_spectra_from_model().
        cardiac_bpm: Heart rate in beats per minute. Default 72 bpm (1.2 Hz).
        cardiac_phase_amp_hz: Peak cardiac frequency modulation amplitude in Hz.
            Typical range: 0.5–2.0 Hz. Default 1.0 Hz.
        resp_bpm: Respiratory rate in breaths per minute. Default 15 bpm (0.25 Hz).
        resp_phase_amp_hz: Peak respiratory frequency modulation amplitude in Hz.
            Typical range: 1.0–5.0 Hz. Default 2.5 Hz.
        tr_s: Repetition time in seconds. Default 2.0 s.
        seed: Random seed for reproducibility. Default None.

    Returns:
        mrs_list with FIDs modified in place.

    Example:
        mrs_list, vm, params = synthetic_spectra_from_model(...)
        mrs_list = add_physiological_noise(mrs_list, cardiac_bpm=65, resp_bpm=12)
    """
    rng = np.random.default_rng(seed)
    n_acq = len(mrs_list)

    # Convert BPM to Hz
    cardiac_hz = cardiac_bpm / 60.0
    resp_hz = resp_bpm / 60.0

    # Random initial phases — each subject's cardiac/respiratory cycle is
    # at an arbitrary phase relative to scan onset
    cardiac_phase0 = rng.uniform(0, 2 * np.pi)
    resp_phase0 = rng.uniform(0, 2 * np.pi)

    # Per-acquisition timing (centre of each TR window)
    t_acq = np.arange(n_acq) * tr_s

    # Frequency modulation at each acquisition (Hz)
    cardiac_mod_hz = cardiac_phase_amp_hz * np.sin(
        2 * np.pi * cardiac_hz * t_acq + cardiac_phase0
    )
    resp_mod_hz = resp_phase_amp_hz * np.sin(
        2 * np.pi * resp_hz * t_acq + resp_phase0
    )
    total_mod_hz = cardiac_mod_hz + resp_mod_hz

    for mrs, delta_f in zip(mrs_list, total_mod_hz):
        n_pts = len(mrs.FID)
        dwell_time = 1.0 / mrs.bandwidth
        t = np.arange(n_pts) * dwell_time

        phase_ramp = np.exp(1j * 2 * np.pi * delta_f * t)
        mrs.FID = mrs.FID * phase_ramp

    return mrs_list
