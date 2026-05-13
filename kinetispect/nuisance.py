"""
kinetispect/nuisance.py

Per-transient frequency and phase correction via spectral registration.
Ported from Osprey's op_robustSpecReg.m
(Oeltzschner & Mikkelsen, Johns Hopkins University, 2019)

The core idea: for each transient FID, find the frequency shift (Hz) and
phase shift (degrees) that minimizes the residual against a reference FID.
The reference is iteratively updated using a weighted combination of the
corrected transients, improving robustness to outliers.

For KinetiSpect, this runs as preprocessing before the Kalman smoother.
Outputs (fs, phs, weights) feed into the nuisance observation equation.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import norm


def spec_reg_residual(x, fid, target, time):
    """
    Residual function for spectral registration.
    
    Applies frequency shift f (Hz) and phase shift phi (degrees) to fid,
    returns residual against target as real-valued vector [real; imag].
    
    Parameters
    ----------
    x : array [f_hz, phi_deg]
    fid : complex array [n_points]
    target : real array [2*n_points]  (stacked real/imag)
    time : array [n_points]  time axis in seconds
    """
    f, phi = x
    fid_shifted = fid * np.exp(1j * np.pi * (time * f * 2 + phi / 180))
    fid_stacked = np.concatenate([np.real(fid_shifted), np.imag(fid_shifted)])
    return target - fid_stacked


def spectral_registration(fids, dwell_time, t_max_fraction=0.2,
                           f0_hz=None, verbose=False):
    """
    Robust spectral registration: correct per-transient frequency and phase
    drifts in the time domain.

    Parameters
    ----------
    fids : complex array [n_points, n_averages]
        Raw FID data, one column per transient.
    dwell_time : float
        Dwell time in seconds (1 / bandwidth).
    t_max_fraction : float
        Use only the first t_max_fraction seconds of FID for registration.
        Default 0.2 s. Shorter = faster, uses high-SNR early part only.
    f0_hz : array [n_averages] or None
        Initial frequency shift guess per transient in Hz.
        If None, estimated from cross-correlation.
    verbose : bool
        Print progress.

    Returns
    -------
    fids_corrected : complex array [n_points, n_averages]
        Frequency- and phase-corrected FIDs (NOT yet averaged).
    fids_averaged : complex array [n_points]
        Weighted average of corrected FIDs.
    fs : array [n_averages]
        Frequency shifts applied (Hz).
    phs : array [n_averages]
        Phase shifts applied (degrees).
    weights : array [n_averages]
        Quality weights for weighted averaging (sum to 1).
    """
    n_points, n_avg = fids.shape
    time = np.arange(n_points) * dwell_time

    # Determine tMax: last point where SNR change > 0.5
    # Use only early high-SNR portion of FID for registration
    t_max_idx = np.searchsorted(time, t_max_fraction)
    signal = np.abs(fids)
    noise = 2 * np.std(signal[int(0.75 * n_points):], axis=0)
    noise = np.where(noise == 0, 1e-10, noise)
    snr = signal / noise[np.newaxis, :]
    snr_diff = np.abs(np.diff(np.mean(snr, axis=1)))
    snr_diff = snr_diff[:t_max_idx]
    last_high = np.where(snr_diff > 0.5)[0]
    t_max = last_high[-1] + 1 if len(last_high) > 0 else t_max_idx
    t_max = max(t_max, int(0.1 / dwell_time))  # at least 100 ms worth

    # --- Step 1: Rank transients by pairwise similarity ---
    data_trunc = fids[:t_max, :]
    D = np.full((n_avg, n_avg), np.nan)
    for jj in range(n_avg):
        for kk in range(n_avg):
            if jj != kk:
                d = np.sum((np.real(data_trunc[:, jj]) -
                            np.real(data_trunc[:, kk])) ** 2) / t_max
                D[jj, kk] = d if d > 0 else np.nan

    median_dist = np.nanmedian(D, axis=1)
    align_order = np.argsort(median_dist)

    # --- Step 2: Build flat real/imag representation for optimization ---
    flat = np.zeros((2 * t_max, n_avg))
    flat[:t_max, :] = np.real(data_trunc)
    flat[t_max:, :] = np.imag(data_trunc)

    # Initial reference: most central transient
    target = flat[:, align_order[0]].copy()
    a = np.max(np.abs(target))  # normalization scalar
    if a == 0:
        a = 1.0

    # Initial frequency guesses
    time_trunc = time[:t_max]
    if f0_hz is not None:
        f0_sorted = np.array(f0_hz)[align_order] - f0_hz[align_order[0]]
    else:
        f0_sorted = np.zeros(n_avg)

    # --- Step 3: Iterative spectral registration ---
    params = np.zeros((n_avg, 2))  # [f_hz, phi_deg] per transient
    mse = np.zeros(n_avg)
    corr_weights = np.zeros(n_avg)

    for iter_idx, jj in enumerate(align_order):
        if verbose:
            print(f'\r  SpecReg transient {iter_idx+1}/{n_avg}', end='')

        fid_j = complex(0) + data_trunc[:, jj]
        x0 = [f0_sorted[iter_idx], 0.0]

        result = least_squares(
            spec_reg_residual,
            x0,
            args=(fid_j / a, target / a, time_trunc),
            method='lm',
            max_nfev=200
        )
        params[jj] = result.x

        f, phi = params[jj]
        fid_corrected = fid_j * np.exp(
            1j * np.pi * (time_trunc * f * 2 + phi / 180))
        m_flat = np.concatenate([np.real(fid_corrected),
                                  np.imag(fid_corrected)])

        resid = target - m_flat
        mse[jj] = np.sum(resid ** 2) / (len(resid) - 2)

        # Update reference with weighted combination
        w_j = 0.5 * np.corrcoef(target, m_flat)[0, 1] ** 2
        w_j = max(0.0, min(1.0, w_j))  # clamp to [0, 1]
        corr_weights[jj] = w_j
        target = (1 - w_j) * target + w_j * m_flat

    if verbose:
        print(' done.')

    # --- Step 4: Apply corrections to full FIDs ---
    fids_corrected = np.zeros_like(fids)
    for jj in range(n_avg):
        f, phi = params[jj]
        fids_corrected[:, jj] = fids[:, jj] * np.exp(
            1j * np.pi * (time * f * 2 + phi / 180))

    # --- Step 5: Compute quality weights for weighted averaging ---
    # Weight = 1/d^2 where d is median pairwise distance after correction
    data_corr_trunc = fids_corrected[:t_max, :]
    D2 = np.full((n_avg, n_avg), np.nan)
    for jj in range(n_avg):
        for kk in range(n_avg):
            if jj != kk:
                d = np.sum((np.real(data_corr_trunc[:, jj]) -
                            np.real(data_corr_trunc[:, kk])) ** 2) / t_max
                D2[jj, kk] = d if d > 0 else np.nan

    median_dist2 = np.nanmedian(D2, axis=1)
    median_dist2 = np.where(np.isnan(median_dist2), 1.0, median_dist2)
    median_dist2 = np.where(median_dist2 == 0, 1e-10, median_dist2)
    weights = 1.0 / median_dist2 ** 2
    weights /= weights.sum()

    # --- Step 6: Weighted average ---
    fids_averaged = np.sum(fids_corrected * weights[np.newaxis, :], axis=1)

    return fids_corrected, fids_averaged, params[:, 0], params[:, 1], weights


def navigator_nuisance(fids_naa, dwell_time, naa_ppm=2.01,
                        cf_mhz=127.7):
    """
    Fast per-acquisition nuisance estimation using the NAA singlet
    as a navigator signal. Much faster than full spectral registration
    because NAA is a single well-resolved peak.

    For KinetiSpect: use this inside the Kalman smoother's nuisance step
    rather than full spectral registration.

    Parameters
    ----------
    fids_naa : complex array [n_points, n_acquisitions]
        Time-domain data (water-suppressed, one FID per dynamic acquisition).
    dwell_time : float
        Dwell time in seconds.
    naa_ppm : float
        Expected NAA chemical shift (2.01 ppm for in vivo brain).
    cf_mhz : float
        Center frequency in MHz (127.7 for 3T ¹H).

    Returns
    -------
    freq_shifts_hz : array [n_acquisitions]
        Per-acquisition frequency offset from nominal NAA position (Hz).
    phase_shifts_deg : array [n_acquisitions]
        Per-acquisition zero-order phase shift (degrees).
    linewidth_hz : array [n_acquisitions]
        Per-acquisition NAA linewidth FWHM (Hz). 
        Proxy for B0 inhomogeneity / BOLD linewidth change.
    """
    n_points, n_acq = fids_naa.shape
    bw = 1.0 / dwell_time
    freq_axis = np.linspace(-bw / 2, bw / 2, n_points)  # Hz
    ppm_axis = freq_axis / cf_mhz  # ppm
    time = np.arange(n_points) * dwell_time

    # Expected NAA frequency offset from carrier
    naa_hz_nominal = naa_ppm * cf_mhz

    freq_shifts = np.zeros(n_acq)
    phase_shifts = np.zeros(n_acq)
    linewidths = np.zeros(n_acq)

    # Reference: first acquisition
    ref_spec = np.fft.fftshift(np.fft.fft(fids_naa[:, 0]))

    # NAA region mask (±0.1 ppm around 2.01 ppm)
    naa_mask = np.abs(ppm_axis - naa_ppm) < 0.1

    for jj in range(n_acq):
        spec = np.fft.fftshift(np.fft.fft(fids_naa[:, jj]))
        spec_naa = spec[naa_mask]
        freq_naa = freq_axis[naa_mask]

        # Frequency shift: peak position relative to nominal
        peak_idx = np.argmax(np.abs(spec_naa))
        freq_shifts[jj] = freq_naa[peak_idx] - naa_hz_nominal

        # Phase shift: phase of peak relative to reference
        ref_naa = ref_spec[naa_mask]
        ref_phase = np.angle(ref_naa[np.argmax(np.abs(ref_naa))])
        cur_phase = np.angle(spec_naa[peak_idx])
        phase_shifts[jj] = np.degrees(cur_phase - ref_phase)

        # Linewidth: FWHM of NAA peak via half-maximum crossing
        mag_naa = np.abs(spec_naa)
        half_max = mag_naa[peak_idx] / 2
        above = mag_naa > half_max
        if above.any():
            transitions = np.diff(above.astype(int))
            rises = np.where(transitions == 1)[0]
            falls = np.where(transitions == -1)[0]
            if len(rises) > 0 and len(falls) > 0:
                f_left = freq_naa[rises[0]]
                f_right = freq_naa[falls[-1]]
                linewidths[jj] = abs(f_right - f_left)
            else:
                linewidths[jj] = linewidths[jj - 1] if jj > 0 else 10.0
        else:
            linewidths[jj] = linewidths[jj - 1] if jj > 0 else 10.0

    return freq_shifts, phase_shifts, linewidths


def estimate_bold_linewidth_coupling(linewidth_hz, bold_pct,
                                      te_ms, cf_mhz=127.7):
    """
    Estimate the BOLD-linewidth coupling coefficient.

    From Just (2024): ΔFWHM = -1/(π·TE) × ln(1 + BOLD)

    This function fits the empirical per-acquisition relationship between
    BOLD signal change and NAA linewidth change, allowing the Kalman
    observation equation to use the measured BOLD as a linewidth prior.

    Parameters
    ----------
    linewidth_hz : array [n_acquisitions]
        Per-acquisition linewidth from navigator_nuisance().
    bold_pct : array [n_acquisitions]
        BOLD signal as percentage change from baseline (from fMRI).
    te_ms : float
        Echo time in milliseconds.
    cf_mhz : float
        Center frequency (unused here, kept for consistency).

    Returns
    -------
    coupling_hz_per_pct : float
        Observed Δlinewidth per 1% BOLD change (Hz/%).
    theoretical_coupling : float
        Theoretical value from Just (2024) equation (Hz/%).
    residuals : array
        Deviation of observed from theoretical coupling per timepoint.
    """
    te_s = te_ms / 1000.0

    # Theoretical prediction from Just (2024)
    delta_fwhm_theory = -1.0 / (np.pi * te_s) * np.log(1 + bold_pct / 100)

    # Baseline linewidth (mean of first 10 acquisitions)
    lw_baseline = np.mean(linewidth_hz[:10])
    delta_lw_observed = linewidth_hz - lw_baseline

    # Linear fit: observed Δlinewidth vs theoretical Δlinewidth
    valid = np.isfinite(delta_fwhm_theory) & np.isfinite(delta_lw_observed)
    if valid.sum() < 3:
        return 1.0, -1.0 / (np.pi * te_s), np.zeros_like(linewidth_hz)

    A = delta_fwhm_theory[valid, np.newaxis]
    b = delta_lw_observed[valid]
    coupling, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    coupling = float(coupling[0])

    theoretical_coupling = -1.0 / (np.pi * te_s) * 0.01  # per 1% BOLD
    residuals = delta_lw_observed - coupling * delta_fwhm_theory

    return coupling, theoretical_coupling, residuals
