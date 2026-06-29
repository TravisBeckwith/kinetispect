# waterbold.py — Per-acquisition waterBOLD referencing for fMRS simulation
#
# Models the BOLD T2*-induced linewidth change per acquisition and applies
# it as additional Lorentzian damping to the FID.
#
# Physics: Just (2024) Front Neurosci establishes that BOLD effects produce
# a linewidth narrowing of:
#
#   ΔFWHM(n) = -1 / (π·TE) · ln(1 + BOLD(n))
#
# where TE is in seconds and BOLD(n) is the fractional BOLD signal change
# (e.g. 0.02 for a 2% BOLD effect). Narrowing means the effective T2* is
# longer during activation, so the FID decays more slowly.
#
# WaterBOLD referencing (Just 2024 recommendation): using the per-acquisition
# unsuppressed water signal as the internal reference cancels the BOLDwater
# component, leaving only BOLDmetab as the remaining confound.
#
# Author: Travis Beckwith / KinetiSpect project
# Reference: Just (2024) Front Neurosci doi:10.3389/fnins.2024.1283825

import numpy as np


def apply_waterbold_referencing(
    mrs_list: list,
    bold_pct: np.ndarray | list,
    te_ms: float = 30.0,
    bold_metab_fraction: float = 0.06,
) -> tuple[list, np.ndarray]:
    """Apply BOLD-induced linewidth change to each acquisition's FID.

    Models the T2*-mediated linewidth narrowing that occurs during neural
    activation. In a real experiment this is applied after waterBOLD
    referencing has cancelled the BOLDwater component; here we model the
    residual BOLDmetab effect directly.

    The BOLD linewidth change (Just 2024, Eq. 1):
        ΔFWHM(n) = -1 / (π·TE_s) · ln(1 + BOLD(n))

    Applied to the FID as additional Lorentzian damping:
        FID(t,n) *= exp(-π · |ΔFWHM(n)| · t)   [narrowing → negative ΔFWHM
                                                    → remove damping]

    For narrowing (BOLD > 0, ΔFWHM < 0): the FID decays *more slowly*,
    implemented by multiplying by exp(+π · |ΔFWHM| · t).

    Args:
        mrs_list: List of MRS objects, one per acquisition.
        bold_pct: Array of BOLD signal values, one per acquisition.
            Expressed as fractional change (e.g. 0.02 for 2% BOLD).
            Length must equal len(mrs_list).
        te_ms: Echo time in milliseconds. Default 30 ms.
            Short TE (3 ms STEAM) → large BOLD effect.
            Long TE (26 ms semi-LASER) → smaller BOLD effect.
            This is a key design parameter — see Just (2024) Fig. 3.
        bold_metab_fraction: Fraction of the BOLD signal that affects
            metabolites after waterBOLD referencing cancels BOLDwater.
            Just (2024) reports ~6% for NAA (BOLDmetab = 0.37% for 6.2%
            total BOLD). Default 0.06.

    Returns:
        Tuple of:
          - mrs_list: FIDs modified in place.
          - linewidth_change_hz: Array [n_acq] of ΔFWHM per acquisition in Hz.
            Negative values = narrowing (activation). For use in nuisance
            tracking validation.

    Raises:
        ValueError: If bold_pct length does not match mrs_list length.

    Example:
        # Simulate a 2% BOLD response during a 30s task block
        import numpy as np
        n_acq, tr_s = 150, 2.0
        t_acq = np.arange(n_acq) * tr_s
        bold_pct = 0.02 * np.where((t_acq >= 60) & (t_acq < 90), 1.0, 0.0)
        mrs_list, lw_change = apply_waterbold_referencing(
            mrs_list, bold_pct, te_ms=30.0
        )
    """
    bold_pct = np.asarray(bold_pct, dtype=float)
    if len(bold_pct) != len(mrs_list):
        raise ValueError(
            f"bold_pct length ({len(bold_pct)}) must match "
            f"mrs_list length ({len(mrs_list)})"
        )

    te_s = te_ms / 1000.0

    # Residual metabolite BOLD fraction after waterBOLD referencing
    bold_metab = bold_pct * bold_metab_fraction

    # Just (2024) Eq. 1: linewidth change in Hz
    # Negative for BOLD > 0 (activation → narrowing)
    # Clamp argument to avoid log(0) for bold_metab = -1
    linewidth_change_hz = -1.0 / (np.pi * te_s) * np.log1p(
        np.clip(bold_metab, -0.999, None)
    )

    for mrs, delta_lw in zip(mrs_list, linewidth_change_hz):
        n_pts = len(mrs.FID)
        dwell_time = 1.0 / mrs.bandwidth
        t = np.arange(n_pts) * dwell_time

        # Lorentzian damping modification
        # Narrowing (delta_lw < 0): remove damping → multiply by exp(+π|Δlw|t)
        # Broadening (delta_lw > 0): add damping → multiply by exp(-πΔlw·t)
        damping = np.exp(-np.pi * delta_lw * t)
        mrs.FID = mrs.FID * damping

    return mrs_list, linewidth_change_hz


def simulate_bold_timeseries(
    n_acq: int,
    tr_s: float = 2.0,
    block_onsets_s: list | None = None,
    block_duration_s: float = 30.0,
    bold_amplitude: float = 0.02,
    hrf_peak_s: float = 6.0,
    hrf_undershoot_s: float = 16.0,
    noise_std: float = 0.002,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a realistic synthetic BOLD time series.

    Convolves a boxcar stimulus with a double-gamma HRF (canonical SPM HRF)
    and adds Gaussian noise at realistic fMRI SNR. Useful for testing
    apply_waterbold_referencing() without needing real fMRI data.

    Args:
        n_acq: Number of acquisitions.
        tr_s: Repetition time in seconds. Default 2.0 s.
        block_onsets_s: List of block onset times in seconds.
            Default: single block at 60 s.
        block_duration_s: Duration of each block in seconds. Default 30 s.
        bold_amplitude: Peak BOLD fractional change. Default 0.02 (2%).
        hrf_peak_s: Time to peak of HRF in seconds. Default 6 s.
        hrf_undershoot_s: Time to undershoot peak. Default 16 s.
        noise_std: SD of Gaussian noise added to BOLD (fractional).
            Default 0.002 (0.2% — realistic fMRI noise at 3T).
        seed: Random seed. Default None.

    Returns:
        bold_pct: Array [n_acq] of fractional BOLD signal changes.

    Example:
        bold = simulate_bold_timeseries(150, tr_s=2.0,
                                        block_onsets_s=[60, 150, 240])
    """
    rng = np.random.default_rng(seed)

    if block_onsets_s is None:
        block_onsets_s = [60.0]

    t_acq = np.arange(n_acq) * tr_s

    # Build boxcar stimulus at acquisition resolution
    stimulus = np.zeros(n_acq)
    for onset in block_onsets_s:
        onset_idx = int(onset / tr_s)
        dur_idx = int(block_duration_s / tr_s)
        end_idx = min(onset_idx + dur_idx, n_acq)
        stimulus[onset_idx:end_idx] = 1.0

    # Double-gamma HRF (simplified SPM canonical)
    # h(t) = t^(a1-1) * exp(-t/b1) / (b1^a1 * Gamma(a1))
    #       - c * t^(a2-1) * exp(-t/b2) / (b2^a2 * Gamma(a2))
    hrf_t = np.arange(0, 32, tr_s)
    a1, b1 = hrf_peak_s, 1.0
    a2, b2 = hrf_undershoot_s, 1.0
    c = 0.35  # undershoot fraction

    def gamma_pdf(t, a, b):
        from math import gamma
        t = np.maximum(t, 1e-10)
        return t**(a - 1) * np.exp(-t / b) / (b**a * gamma(a))

    hrf = gamma_pdf(hrf_t, a1, b1) - c * gamma_pdf(hrf_t, a2, b2)
    hrf = hrf / hrf.max()  # normalise to peak = 1

    # Convolve and trim
    bold_clean = np.convolve(stimulus, hrf)[:n_acq]
    bold_clean = bold_clean / max(bold_clean.max(), 1e-10) * bold_amplitude

    # Add realistic fMRI noise
    noise = rng.normal(0, noise_std, size=n_acq)
    bold_pct = bold_clean + noise

    return bold_pct
