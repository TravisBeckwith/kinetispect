# kalman.py — Extended Kalman Filter for fMRS dynamic concentration estimation
#
# Implements a predict/update loop over a list of FSL-MRS MRS objects.
# The observation function is nonlinear (FSL-MRS Voigt forward model), so
# we use a numerical Jacobian for the linearisation at each update step.
#
# Filter equations
# ----------------
# Predict:
#   x_{n|n-1} = F · x_{n-1|n-1}
#   P_{n|n-1} = F · P_{n-1|n-1} · F^T + Q
#
# Update:
#   y_n        = z_n - h(x_{n|n-1})          innovation
#   H_n        = ∂h/∂x |_{x_{n|n-1}}         numerical Jacobian
#   S_n        = H_n · P_{n|n-1} · H_n^T + R innovation covariance
#   K_n        = P_{n|n-1} · H_n^T · S_n^{-1} Kalman gain
#   x_{n|n}    = x_{n|n-1} + K_n · y_n
#   P_{n|n}    = (I - K_n · H_n) · P_{n|n-1}
#
# Observation model h(x)
# ----------------------
# h maps the Kalman state vector to the predicted FID (complex, flattened to
# real/imag concatenated). Internally calls state_to_fslmrs_params() then
# the FSL-MRS Voigt forward model.
#
# Numerical Jacobian
# ------------------
# Central differences with step size ε = 1e-4 per state dimension.
# Analytical Jacobian would be faster but requires differentiating the
# FSL-MRS forward model — deferred to Phase 3 optimisation.
#
# Author: Travis Beckwith / KinetiSpect project

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from kinetispect.state_space import (
    StateSpaceConfig,
    make_transition_matrix,
    make_process_noise,
    make_initial_state,
    state_to_fslmrs_params,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class KalmanResult:
    """Output of run_kalman_filter().

    Attributes:
        state_means: Array [n_acq, n_state], filtered state means x_{n|n}.
        state_covs: Array [n_acq, n_state, n_state], filtered state covariances.
        innovations: Array [n_acq, n_obs], innovation vectors y_n.
        innovation_covs: Array [n_acq, n_obs, n_obs], innovation covariances S_n.
        cfg: StateSpaceConfig used for the run.
        metabolite_names: Convenience alias for cfg.metabolite_names.
    """
    state_means: np.ndarray
    state_covs: np.ndarray
    innovations: list
    innovation_covs: list
    cfg: StateSpaceConfig

    @property
    def metabolite_names(self) -> list[str]:
        return self.cfg.metabolite_names

    def concentration(self, metabolite: str) -> np.ndarray:
        """Return concentration time series [n_acq] for one metabolite (mM)."""
        if metabolite not in self.cfg.metabolite_names:
            raise ValueError(f"'{metabolite}' not in {self.cfg.metabolite_names}")
        idx = self.cfg.metabolite_names.index(metabolite)
        return self.state_means[:, idx]

    def concentration_std(self, metabolite: str) -> np.ndarray:
        """Return per-timepoint posterior SD [n_acq] for one metabolite (mM)."""
        idx = self.cfg.metabolite_names.index(metabolite)
        return np.sqrt(self.state_covs[:, idx, idx])

    def nuisance(self) -> dict[str, np.ndarray]:
        """Return estimated nuisance time series as a dict."""
        return {
            'df_hz':     self.state_means[:, self.cfg.idx_df],
            'dphi_rad':  self.state_means[:, self.cfg.idx_dphi],
            'dgamma_hz': self.state_means[:, self.cfg.idx_dgamma],
        }


# ---------------------------------------------------------------------------
# Observation function and numerical Jacobian
# ---------------------------------------------------------------------------

def _build_observation_fn(
    mrs,
    cfg: StateSpaceConfig,
    baseline_x: np.ndarray,
    x2p: Callable,
    p2x: Callable,
    forward: Callable,
    n_pts_obs: int | None = None,
) -> Callable:
    """Build the observation function h(state) for one MRS acquisition.

    h maps the Kalman state vector to the predicted FID, returned as a
    real-valued vector [Re(FID), Im(FID)] of length 2*n_pts.

    We truncate the FID to n_pts_obs points if specified — using the full
    2048-point FID as the observation would make the Jacobian computation
    prohibitively slow during development. Recommended: 128–256 points
    (covers the early FID where SNR is highest).

    Args:
        mrs: FSL-MRS MRS object for this acquisition.
        cfg: StateSpaceConfig instance.
        baseline_x: FSL-MRS reference parameter vector.
        x2p, p2x: FSL-MRS unpack/repack functions.
        forward: FSL-MRS Voigt forward function with signature
                 (x, nu, t, m, B, G, g).
        n_pts_obs: Number of FID points to use as observations.
            Default None uses all points (slow; use 128 for development).

    Returns:
        h: Callable[state → np.ndarray[float, (2*n_pts_obs,)]]
    """
    n_pts = len(mrs.FID)
    if n_pts_obs is None:
        n_pts_obs = n_pts
    n_pts_obs = min(n_pts_obs, n_pts)

    # Pre-extract MRS geometry needed by forward model
    # FSL-MRS MRS object attributes:
    #   .frequencyAxis  — frequency axis (Hz)
    #   .timeAxis       — time axis (s)
    #   ._basis         — basis set object with .basis_fids (complex [n_pts, n_metab])
    #   .numBasis       — n_metab
    #   .metab_groups   — group index per basis spectrum

    nu = mrs.frequencyAxis[:n_pts_obs]          # Hz
    t  = mrs.timeAxis[:n_pts_obs]               # s
    m  = mrs.numBasis
    B  = mrs._basis.basis_fids[:n_pts_obs, :]   # complex [n_pts_obs, n_metab]
    G  = mrs.metab_groups
    g  = max(G) + 1

    n_metab_fsl = cfg.n_metab
    n_groups = cfg.n_groups

    def h(state: np.ndarray) -> np.ndarray:
        x_fsl = state_to_fslmrs_params(
            state, cfg, baseline_x, x2p, p2x
        )
        fid_pred = forward(x_fsl, nu, t, m, B, G, g)
        fid_pred = fid_pred[:n_pts_obs]
        return np.concatenate([fid_pred.real, fid_pred.imag])

    return h


def _numerical_jacobian(
    h: Callable,
    state: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """Compute numerical Jacobian of h at state using central differences.

    H[i, j] = (h(state + eps*e_j)[i] - h(state - eps*e_j)[i]) / (2*eps)

    Args:
        h: Observation function, state → observation vector.
        state: Array [n_state], point at which to evaluate Jacobian.
        eps: Finite difference step size. Default 1e-4.

    Returns:
        H: Array [n_obs, n_state], numerical Jacobian.
    """
    n_state = len(state)
    h0 = h(state)
    n_obs = len(h0)
    H = np.zeros((n_obs, n_state))

    for j in range(n_state):
        e = np.zeros(n_state)
        e[j] = eps
        H[:, j] = (h(state + e) - h(state - e)) / (2 * eps)

    return H


# ---------------------------------------------------------------------------
# Main filter
# ---------------------------------------------------------------------------

def run_kalman_filter(
    mrs_list: list,
    cfg: StateSpaceConfig,
    baseline_x: np.ndarray,
    x2p: Callable,
    p2x: Callable,
    forward: Callable,
    obs_noise_std: float = 0.01,
    n_pts_obs: int = 128,
    jacobian_eps: float = 1e-4,
    verbose: bool = True,
) -> KalmanResult:
    """Run the extended Kalman filter over an fMRS time series.

    Args:
        mrs_list: List of FSL-MRS MRS objects, one per acquisition, as
            returned by synthetic_spectra_from_model() (and optionally
            modified by physio_noise / waterbold functions).
        cfg: StateSpaceConfig defining the state space model.
        baseline_x: FSL-MRS packed parameter vector from a reference fit
            (e.g. time-averaged spectrum). Used as the baseline for
            state_to_fslmrs_params(). Obtain from FSL-MRS fitting the
            mean spectrum, or pass zeros to use prior values only.
        x2p: FSL-MRS unpack function from getModelFunctions('voigt').
        p2x: FSL-MRS repack function from getModelFunctions('voigt').
        forward: FSL-MRS forward function from getModelFunctions('voigt').
        obs_noise_std: SD of observation noise (FID amplitude units).
            Scaled as R = obs_noise_std² · I. Default 0.01. Should be
            matched to the noisecovariance used in synthetic_spectra_from_model.
        n_pts_obs: Number of FID points to use as observations per acquisition.
            Lower values → faster Jacobian; higher values → more information.
            Default 128. Increase to 256–512 for final runs.
        jacobian_eps: Central difference step size for numerical Jacobian.
            Default 1e-4.
        verbose: Print progress every 10 acquisitions. Default True.

    Returns:
        KalmanResult with filtered state means, covariances, and diagnostics.
    """
    n_acq = len(mrs_list)
    n_state = cfg.n_state
    n_obs = 2 * n_pts_obs  # real + imaginary FID points

    # Initialise
    F = make_transition_matrix(cfg)
    Q = make_process_noise(cfg)
    R = (obs_noise_std ** 2) * np.eye(n_obs)

    x, P = make_initial_state(cfg)

    # Storage
    state_means = np.zeros((n_acq, n_state))
    state_covs  = np.zeros((n_acq, n_state, n_state))
    innovations     = []
    innovation_covs = []

    for n, mrs in enumerate(mrs_list):
        if verbose and n % 10 == 0:
            print(f"  Kalman filter: acquisition {n+1}/{n_acq}")

        # --- Predict ---
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # --- Build observation function for this acquisition ---
        h = _build_observation_fn(
            mrs, cfg, baseline_x, x2p, p2x, forward, n_pts_obs
        )

        # --- Observed FID (real/imag concatenated, truncated to n_pts_obs) ---
        fid_obs = mrs.FID[:n_pts_obs]
        z = np.concatenate([fid_obs.real, fid_obs.imag])

        # --- Numerical Jacobian at predicted state ---
        H = _numerical_jacobian(h, x_pred, eps=jacobian_eps)

        # --- Innovation ---
        y = z - h(x_pred)

        # --- Innovation covariance ---
        S = H @ P_pred @ H.T + R

        # --- Kalman gain (solve S·K^T = P_pred·H^T for numerical stability) ---
        K = np.linalg.solve(S.T, (P_pred @ H.T).T).T

        # --- Update ---
        x = x_pred + K @ y
        P = (np.eye(n_state) - K @ H) @ P_pred

        # Enforce symmetry (prevents numerical drift)
        P = 0.5 * (P + P.T)

        # Physical constraint: concentrations ≥ 0
        x[cfg.idx_conc] = np.maximum(x[cfg.idx_conc], 0.0)

        # Store
        state_means[n] = x
        state_covs[n]  = P
        innovations.append(y)
        innovation_covs.append(S)

    return KalmanResult(
        state_means=state_means,
        state_covs=state_covs,
        innovations=innovations,
        innovation_covs=innovation_covs,
        cfg=cfg,
    )


# ---------------------------------------------------------------------------
# Convenience: build baseline_x from FSL-MRS standard values
# ---------------------------------------------------------------------------

def make_baseline_x(
    mrs,
    concentrations: np.ndarray,
    cfg: StateSpaceConfig,
    p2x: Callable,
    gamma: float = 10.0,
    sigma: float = 10.0,
) -> np.ndarray:
    """Build a baseline FSL-MRS parameter vector from prior concentrations.

    Useful when no reference fit is available. Sets gamma/sigma to typical
    3T values, phase and baseline to zero.

    Args:
        mrs: Any MRS object from mrs_list (used for n_groups/n_basis shape).
        concentrations: Array [n_metab], prior concentration values (mM).
        cfg: StateSpaceConfig instance.
        p2x: FSL-MRS repack function.
        gamma: Lorentzian linewidth prior (Hz). Default 10 Hz.
        sigma: Gaussian linewidth prior (Hz). Default 10 Hz.

    Returns:
        baseline_x: Array, FSL-MRS packed parameter vector.
    """
    n_basis  = mrs.numBasis
    n_groups = cfg.n_groups
    n_base   = cfg.baseline_order + 1

    con    = np.array(concentrations).reshape(n_basis, n_groups)
    gam    = np.full(n_groups, gamma)
    sig    = np.full(n_groups, sigma)
    eps    = np.zeros(n_groups)
    phi0   = 0.0
    phi1   = 0.0
    b      = np.zeros(n_base)

    return p2x(con, gam, sig, eps, phi0, phi1, b)
