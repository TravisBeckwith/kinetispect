"""
kinetispect/crlb.py

Cramér-Rao Lower Bound computation for MRS fitting.
Ported from Osprey's fit_Osprey_CRLB.m
(Oeltzschner, Johns Hopkins University, 2019)

The CRLB is the theoretical minimum variance of any unbiased estimator.
For MRS, it provides the %SD uncertainty on each metabolite concentration.

The Fisher information matrix is:
    F = (1/σ²) × Re(J^H J)

where J is the Jacobian of the model with respect to all parameters,
and σ is estimated from the residual standard deviation.

CRLB = sqrt(diag(inv(F)))

%SD = 100 × CRLB(amplitude_i) / amplitude_i

This is the standard uncertainty metric reported by LCModel and Osprey.
Values < 20% are considered reliable; > 50% unreliable.

For KinetiSpect: this is called after the Kalman smoother converges
to report per-timepoint and per-metabolite uncertainty.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class CRLBResult:
    """Container for CRLB computation results."""
    # Absolute CRLB (same units as concentration)
    crlb_abs: np.ndarray          # [n_params] absolute uncertainty
    # Relative CRLB = %SD for metabolite amplitudes
    crlb_pct: np.ndarray          # [n_metabolites] %SD
    # Fisher information matrix
    fisher: np.ndarray             # [n_params, n_params]
    # Residual standard deviation
    sigma: float
    # Jacobian (for diagnostics)
    jacobian: np.ndarray           # [n_spectral_points, n_params]
    # Parameter indices for easy access
    param_indices: dict


def compute_jacobian(basis_fids, concentrations, gamma, sigma_gauss,
                     eps, phi0, phi1, beta_j, spline_basis,
                     time, freq_axis, lineshape=None,
                     n_mets=None, n_mm=0):
    """
    Compute the analytical Jacobian of the Osprey/FSL-MRS forward model.

    The model is:
        S(ν) = exp(-i(φ₀ + φ₁·ν)) × [Σₖ aₖ · (Mₖ * lineShape) + B·β]

    where Mₖ = FFT(mₖ(t) · exp(-i·εₖ·t) · exp(-γₖ·t) · exp(-σₖ²·t²))

    Parameters
    ----------
    basis_fids : complex array [n_points, n_basis]
        Basis FIDs (one per metabolite + MM), pre-normalized.
    concentrations : array [n_basis]
        Fitted amplitudes.
    gamma : array [n_groups]
        Lorentzian linewidth per metabolite group (Hz).
    sigma_gauss : float
        Gaussian linewidth (shared, Hz).
    eps : array [n_groups]
        Frequency shift per metabolite group (Hz, as phase rate).
    phi0 : float
        Zero-order phase (radians).
    phi1 : float
        First-order phase (radians/ppm). Applied as exp(i·φ₁·(ν - pivot)).
    beta_j : array [n_splines]
        Spline baseline amplitudes.
    spline_basis : complex array [n_spectral_points, n_splines]
        Spline basis functions in frequency domain.
    time : array [n_points]
        Time axis (seconds).
    freq_axis : array [n_spectral_points]
        Frequency axis after cropping to fit range (ppm).
    lineshape : array or None
        Lineshape vector for convolution. If None, uses delta (no convolution).
    n_mets : int or None
        Number of metabolite basis functions (remainder are MM/lipid).
    n_mm : int
        Number of MM basis functions.

    Returns
    -------
    J_freq : complex array [n_spectral_points, n_params]
        Jacobian in the frequency domain.
    param_indices : dict
        Maps parameter names to column indices in J_freq.
    """
    n_points, n_basis = basis_fids.shape
    n_spectral = len(freq_axis)
    if n_mets is None:
        n_mets = n_basis - n_mm
    n_splines = len(beta_j)
    n_ls = len(lineshape) if lineshape is not None else 0

    # Total parameters:
    # ph0(1) + ph1(1) + gaussLB(1) + lorentzLB(n_basis) + freqShift(n_basis)
    # + ampl(n_basis) + spline_ampl(n_splines) + lineshape(n_ls)
    n_params = 3 + n_basis + n_basis + n_basis + n_splines + n_ls

    # Parameter index map
    idx = {}
    idx['phi0']       = 0
    idx['phi1']       = 1
    idx['gaussLB']    = 2
    idx['lorentzLB']  = slice(3, 3 + n_basis)
    idx['freqShift']  = slice(3 + n_basis, 3 + 2 * n_basis)
    idx['ampl']       = slice(3 + 2 * n_basis, 3 + 3 * n_basis)
    idx['spline']     = slice(3 + 3 * n_basis, 3 + 3 * n_basis + n_splines)
    if n_ls > 0:
        idx['lineshape'] = slice(3 + 3 * n_basis + n_splines,
                                  3 + 3 * n_basis + n_splines + n_ls)

    # Apply damping to basis FIDs
    damped = np.zeros_like(basis_fids)
    for ii in range(n_basis):
        g = gamma[ii] if ii < len(gamma) else gamma[-1]
        e = eps[ii] if ii < len(eps) else eps[-1]
        damped[:, ii] = (basis_fids[:, ii]
                         * np.exp(-1j * e * time)
                         * np.exp(-g * time)
                         * np.exp(-sigma_gauss * time ** 2)
                         * np.exp(1j * phi0))

    # Frequency-domain basis spectra
    def fft_shift(x):
        return np.fft.fftshift(np.fft.fft(x, axis=0), axes=0)

    spec_basis = fft_shift(damped)  # [n_points, n_basis]

    # Pivot for first-order phase
    pivot_ppm = 4.68
    mult = freq_axis - pivot_ppm
    phase_term = np.exp(-1j * (phi0 + phi1 * freq_axis))

    # Apply first-order phase and crop to fit range
    # (assume spec_basis already cropped to n_spectral points)
    # Convolve metabolite functions with lineshape
    if lineshape is not None:
        ls_norm = lineshape / lineshape.sum()
        A = spec_basis[:n_spectral, :].copy()
        for kk in range(n_mets):
            A[:, kk] = np.convolve(A[:, kk], ls_norm, mode='same')
    else:
        A = spec_basis[:n_spectral, :]

    # Apply phase
    for ii in range(n_basis):
        A[:, ii] = A[:, ii] * np.exp(1j * phi1 * mult)

    # Full forward model for completeness
    B = spline_basis  # [n_spectral, n_splines]
    B_phased = B * np.exp(1j * phi0) * np.exp(1j * phi1 * mult[:, np.newaxis])

    complete_fit = A @ concentrations + B_phased @ beta_j

    # --- Jacobian columns ---
    J = np.zeros((n_spectral, n_params), dtype=complex)

    # dS/dphi0
    J[:, idx['phi0']] = 1j * complete_fit

    # dS/dphi1
    J[:, idx['phi1']] = 1j * complete_fit * mult

    # dS/dgaussLB (shared Gaussian, all basis functions)
    tmp = np.zeros(n_spectral, dtype=complex)
    time_basis = fft_shift(-damped * time[:, np.newaxis] ** 2)[:n_spectral, :]
    for ii in range(n_basis):
        tb = time_basis[:, ii] * np.exp(1j * phi1 * mult)
        if ii < n_mets and lineshape is not None:
            tmp += np.convolve(tb, ls_norm, mode='same') * concentrations[ii]
        else:
            tmp += tb * concentrations[ii]
    J[:, idx['gaussLB']] = tmp

    # dS/dlorentzLB[ii] (per basis function)
    time_lor = fft_shift(-damped * time[:, np.newaxis])[:n_spectral, :]
    lo_sl = idx['lorentzLB']
    for ii in range(n_basis):
        tb = time_lor[:, ii] * np.exp(1j * phi1 * mult)
        if ii < n_mets and lineshape is not None:
            J[:, lo_sl.start + ii] = (np.convolve(tb, ls_norm, mode='same')
                                       * concentrations[ii])
        else:
            J[:, lo_sl.start + ii] = tb * concentrations[ii]

    # dS/dfreqShift[ii] (per basis function)
    time_eps = fft_shift(-1j * damped * time[:, np.newaxis])[:n_spectral, :]
    fs_sl = idx['freqShift']
    for ii in range(n_basis):
        tb = time_eps[:, ii] * np.exp(1j * phi1 * mult)
        if ii < n_mets and lineshape is not None:
            J[:, fs_sl.start + ii] = (np.convolve(tb, ls_norm, mode='same')
                                       * concentrations[ii])
        else:
            J[:, fs_sl.start + ii] = tb * concentrations[ii]

    # dS/dampl[ii] (per basis function)
    amp_sl = idx['ampl']
    for ii in range(n_basis):
        J[:, amp_sl.start + ii] = A[:, ii]

    # dS/dbeta_j[ii] (spline amplitudes)
    sp_sl = idx['spline']
    for ii in range(n_splines):
        J[:, sp_sl.start + ii] = B_phased[:, ii]

    return J, idx


def compute_crlb(data_spec, forward_spec, jacobian, param_indices,
                  n_mets, concentrations,
                  use_pseudoinverse=True):
    """
    Compute Cramér-Rao Lower Bounds from the analytical Jacobian.

    Parameters
    ----------
    data_spec : complex array [n_spectral_points]
        Observed spectrum in the fit range.
    forward_spec : complex array [n_spectral_points]
        Model fit (forward model evaluated at converged parameters).
    jacobian : complex array [n_spectral_points, n_params]
        Analytical Jacobian from compute_jacobian().
    param_indices : dict
        Parameter index map from compute_jacobian().
    n_mets : int
        Number of metabolite basis functions.
    concentrations : array [n_basis]
        Fitted amplitudes.
    use_pseudoinverse : bool
        Use pseudoinverse for numerical stability. Default True.

    Returns
    -------
    CRLBResult
    """
    # Noise estimate from residual standard deviation
    residual = np.real(data_spec) - np.real(forward_spec)
    sigma = np.std(residual)
    if sigma == 0:
        sigma = 1e-10

    # Fisher information matrix: F = (1/σ²) × Re(J^H J)
    J_real = np.real(jacobian)
    J_imag = np.imag(jacobian)
    J_combined = np.vstack([J_real, J_imag])  # treat real and imag separately

    fisher = (1.0 / sigma ** 2) * (J_combined.T @ J_combined)

    # CRLB = sqrt(diag(inv(F)))
    if use_pseudoinverse:
        fisher_inv = np.linalg.pinv(fisher)
    else:
        try:
            fisher_inv = np.linalg.inv(fisher)
        except np.linalg.LinAlgError:
            fisher_inv = np.linalg.pinv(fisher)

    crlb_abs = np.sqrt(np.abs(np.diag(fisher_inv)))

    # Relative %SD for metabolite amplitudes only
    amp_sl = param_indices['ampl']
    amp_crlb = crlb_abs[amp_sl][:n_mets]
    amp_vals = concentrations[:n_mets]

    crlb_pct = np.zeros(n_mets)
    for ii in range(n_mets):
        if amp_vals[ii] > 0:
            crlb_pct[ii] = 100.0 / (amp_vals[ii] / amp_crlb[ii])
        else:
            crlb_pct[ii] = np.inf

    return CRLBResult(
        crlb_abs=crlb_abs,
        crlb_pct=crlb_pct,
        fisher=fisher,
        sigma=sigma,
        jacobian=jacobian,
        param_indices=param_indices
    )


def crlb_quality_flag(crlb_pct, threshold_reliable=20.0,
                       threshold_unreliable=50.0):
    """
    Classify metabolite CRLB values using standard thresholds.

    Parameters
    ----------
    crlb_pct : array [n_mets]
        %SD values from compute_crlb().
    threshold_reliable : float
        %SD below this → reliable. Default 20%.
    threshold_unreliable : float
        %SD above this → unreliable/excluded. Default 50%.

    Returns
    -------
    flags : array [n_mets]  of str: 'reliable', 'marginal', 'unreliable'
    """
    flags = []
    for pct in crlb_pct:
        if pct < threshold_reliable:
            flags.append('reliable')
        elif pct < threshold_unreliable:
            flags.append('marginal')
        else:
            flags.append('unreliable')
    return np.array(flags)
