# state_space.py — Linear Gaussian state space model for fMRS Kalman smoother
#
# Defines the state vector, transition model, and process noise for the
# KinetiSpect Kalman filter. The observation function (which calls the
# FSL-MRS Voigt forward model) lives in kalman.py.
#
# State vector layout
# -------------------
# The state vector x_n ∈ R^(n_metab + 3) has two blocks:
#
#   [0 : n_metab]   Metabolite concentrations (mM), one per fitted metabolite.
#                   Ordered to match fsl_mrs MRS.names.
#   [n_metab]       Δf   — per-acquisition frequency offset (Hz)
#   [n_metab + 1]   Δφ   — per-acquisition zero-order phase offset (rad)
#   [n_metab + 2]   Δγ   — per-acquisition Lorentzian linewidth offset (Hz)
#
# Transition model (random walk)
# ------------------------------
# x_n = F · x_{n-1} + w_n,   w_n ~ N(0, Q)
#
# F = I  (identity — each state component follows an independent random walk)
#
# Process noise Q is diagonal. Metabolite concentrations evolve slowly
# (physiological timescale ~minutes); nuisance parameters can jump faster
# (heartbeat, B0 drift). Separate noise variances are specified for each block.
#
# Mapping to FSL-MRS parameter vector
# ------------------------------------
# The FSL-MRS Voigt forward model takes a packed vector x with layout:
#   [conc (n_metab × n_groups), gamma (n_groups), sigma (n_groups),
#    eps (n_groups), Phi_0, Phi_1, baseline coefficients]
#
# At each timepoint we build this vector from the Kalman state by:
#   - substituting Kalman concentration estimates for the conc block
#   - adding Δf → eps, Δφ → Phi_0, Δγ → gamma offsets
#   - holding other parameters at their baseline (fitted or prior) values
#
# Author: Travis Beckwith / KinetiSpect project

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class StateSpaceConfig:
    """Configuration for the KinetiSpect linear Gaussian state space model.

    Args:
        metabolite_names: Ordered list of metabolite names matching MRS.names.
        conc_prior_mean: Prior mean concentrations (mM), one per metabolite.
            If None, uses FSL-MRS standard concentrations (~8 mM for Glu).
        conc_prior_std: Prior SD on concentrations (mM). Large value = weak
            prior. Default 2.0 mM (±2σ covers ~95% of physiological range).
        conc_rw_std: Random walk step SD for concentrations per acquisition.
            Controls how fast concentrations can change between timepoints.
            Default 0.3 mM/acq — allows ~1 mM change over 3–4 acquisitions,
            consistent with typical Glu activation responses.
        df_prior_std: Prior SD on frequency offset Δf (Hz). Default 5 Hz.
        df_rw_std: Random walk step SD for Δf per acquisition (Hz).
            Default 0.5 Hz — captures B0 drift between TR windows.
        dphi_prior_std: Prior SD on phase offset Δφ (rad). Default 0.3 rad.
        dphi_rw_std: Random walk step SD for Δφ per acquisition (rad).
            Default 0.1 rad.
        dgamma_prior_std: Prior SD on linewidth offset Δγ (Hz). Default 2 Hz.
        dgamma_rw_std: Random walk step SD for Δγ per acquisition (Hz).
            Default 0.3 Hz — captures slow linewidth variation.
        n_groups: Number of metabolite groups in the FSL-MRS model. Default 1.
        baseline_order: Polynomial baseline order. Default 0.
    """

    metabolite_names: list[str]
    conc_prior_mean: np.ndarray | None = None
    conc_prior_std: float = 2.0
    conc_rw_std: float = 0.3
    df_prior_std: float = 5.0
    df_rw_std: float = 0.5
    dphi_prior_std: float = 0.3
    dphi_rw_std: float = 0.1
    dgamma_prior_std: float = 2.0
    dgamma_rw_std: float = 0.3
    n_groups: int = 1
    baseline_order: int = 0

    def __post_init__(self):
        n = len(self.metabolite_names)
        if self.conc_prior_mean is None:
            # Rough physiological priors (mM) — overridden per-metabolite below
            defaults = {
                'Glu': 9.0, 'Gln': 4.0, 'NAA': 10.0, 'NAAG': 1.5,
                'Cr': 5.5, 'PCr': 4.5, 'GPC': 1.0, 'PCh': 0.5,
                'Ins': 7.0, 'Tau': 2.0, 'Asc': 1.5, 'GSH': 1.5,
                'GABA': 1.5, 'Asp': 2.0, 'Glc': 1.5, 'Lac': 0.5,
                'Scyllo': 0.3, 'PE': 1.0, 'Mac': 3.0,
            }
            self.conc_prior_mean = np.array(
                [defaults.get(name, 2.0) for name in self.metabolite_names]
            )
        else:
            self.conc_prior_mean = np.asarray(self.conc_prior_mean, dtype=float)

    @property
    def n_metab(self) -> int:
        return len(self.metabolite_names)

    @property
    def n_state(self) -> int:
        """Total state dimension: n_metab concentrations + 3 nuisance params."""
        return self.n_metab + 3

    # State vector index helpers
    @property
    def idx_conc(self) -> slice:
        return slice(0, self.n_metab)

    @property
    def idx_df(self) -> int:
        return self.n_metab

    @property
    def idx_dphi(self) -> int:
        return self.n_metab + 1

    @property
    def idx_dgamma(self) -> int:
        return self.n_metab + 2


# ---------------------------------------------------------------------------
# State space matrices
# ---------------------------------------------------------------------------

def make_transition_matrix(cfg: StateSpaceConfig) -> np.ndarray:
    """Return the state transition matrix F.

    For a random walk model F = I (identity). Included as an explicit function
    so it can be swapped for a more complex transition (e.g. gamma-variate
    for metabolite dynamics) without changing the Kalman filter loop.

    Args:
        cfg: StateSpaceConfig instance.

    Returns:
        F: Array [n_state, n_state], identity matrix.
    """
    return np.eye(cfg.n_state)


def make_process_noise(cfg: StateSpaceConfig) -> np.ndarray:
    """Return the process noise covariance matrix Q.

    Diagonal matrix with separate variance for metabolite concentrations
    and each nuisance parameter. Off-diagonal terms are zero — physiologically
    reasonable since B0 drift and concentration changes are driven by
    independent processes.

    Args:
        cfg: StateSpaceConfig instance.

    Returns:
        Q: Array [n_state, n_state], diagonal process noise covariance.
    """
    q_diag = np.zeros(cfg.n_state)

    # Concentration block: same random walk variance for all metabolites
    q_diag[cfg.idx_conc] = cfg.conc_rw_std ** 2

    # Nuisance parameter block
    q_diag[cfg.idx_df]     = cfg.df_rw_std ** 2
    q_diag[cfg.idx_dphi]   = cfg.dphi_rw_std ** 2
    q_diag[cfg.idx_dgamma] = cfg.dgamma_rw_std ** 2

    return np.diag(q_diag)


def make_initial_state(cfg: StateSpaceConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return the initial state mean and covariance for the Kalman filter.

    The initial state is set to the physiological prior means with wide
    uncertainty (diagonal covariance). Nuisance parameters initialise to
    zero (no drift, no phase offset) with moderate uncertainty.

    Args:
        cfg: StateSpaceConfig instance.

    Returns:
        x0: Array [n_state], initial state mean.
        P0: Array [n_state, n_state], initial state covariance.
    """
    x0 = np.zeros(cfg.n_state)
    x0[cfg.idx_conc] = cfg.conc_prior_mean
    # Nuisance params start at zero — no prior drift/phase/linewidth offset
    x0[cfg.idx_df]     = 0.0
    x0[cfg.idx_dphi]   = 0.0
    x0[cfg.idx_dgamma] = 0.0

    p0_diag = np.zeros(cfg.n_state)
    p0_diag[cfg.idx_conc] = cfg.conc_prior_std ** 2
    p0_diag[cfg.idx_df]     = cfg.df_prior_std ** 2
    p0_diag[cfg.idx_dphi]   = cfg.dphi_prior_std ** 2
    p0_diag[cfg.idx_dgamma] = cfg.dgamma_prior_std ** 2

    P0 = np.diag(p0_diag)

    return x0, P0


# ---------------------------------------------------------------------------
# FSL-MRS parameter vector construction
# ---------------------------------------------------------------------------

def state_to_fslmrs_params(
    state: np.ndarray,
    cfg: StateSpaceConfig,
    baseline_x: np.ndarray,
    x2p,
    p2x,
) -> np.ndarray:
    """Map a Kalman state vector to an FSL-MRS packed parameter vector.

    The FSL-MRS Voigt forward model takes a packed vector x with layout:
        [conc (n_metab × n_groups), gamma (n_groups), sigma (n_groups),
         eps (n_groups), Phi_0, Phi_1, baseline (baseline_order + 1)]

    This function unpacks `baseline_x` (the FSL-MRS parameter vector from a
    reference fit or prior), substitutes the Kalman concentration estimates,
    and adds the nuisance offsets Δf → eps, Δφ → Phi_0, Δγ → gamma.

    Args:
        state: Array [n_state], current Kalman state mean.
        cfg: StateSpaceConfig instance.
        baseline_x: Array, FSL-MRS packed parameter vector from a reference
            fit (e.g. time-averaged spectrum fit). Used for gamma, sigma,
            Phi_1, and baseline — parameters not tracked in the Kalman state.
        x2p: FSL-MRS unpacking function from getModelFunctions('voigt').
        p2x: FSL-MRS packing function from getModelFunctions('voigt').

    Returns:
        x_fsl: Array, FSL-MRS packed parameter vector with Kalman
            concentrations and nuisance offsets inserted.
    """
    n_groups = cfg.n_groups
    n_basis = cfg.n_metab

    # Unpack the baseline parameter vector
    con, gamma, sigma, eps, phi0, phi1, b = x2p(
        baseline_x, n_basis, n_groups
    )

    # Substitute Kalman concentration estimates
    con_kalman = state[cfg.idx_conc].copy()
    con_kalman = np.maximum(con_kalman, 0.0)  # physical constraint: conc ≥ 0

    # Add nuisance offsets to the corresponding FSL-MRS parameters
    # Δf [Hz] → eps (frequency shift parameter, Hz in FSL-MRS)
    eps_new = eps + state[cfg.idx_df]

    # Δφ [rad] → Phi_0 (zero-order phase, rad in FSL-MRS)
    phi0_new = phi0 + state[cfg.idx_dphi]

    # Δγ [Hz] → gamma (Lorentzian linewidth, Hz in FSL-MRS)
    gamma_new = np.maximum(gamma + state[cfg.idx_dgamma], 0.0)

    # Repack into FSL-MRS parameter vector
    x_fsl = p2x(
        con_kalman.reshape(con.shape),
        gamma_new,
        sigma,
        eps_new,
        phi0_new,
        phi1,
        b,
    )

    return x_fsl


# ---------------------------------------------------------------------------
# Utility: extract concentration time series from a filter output
# ---------------------------------------------------------------------------

def extract_concentrations(
    state_means: np.ndarray,
    cfg: StateSpaceConfig,
    metabolite: str,
) -> np.ndarray:
    """Extract a single metabolite's concentration time series.

    Args:
        state_means: Array [n_acq, n_state], Kalman filter state means.
        cfg: StateSpaceConfig instance.
        metabolite: Metabolite name, must be in cfg.metabolite_names.

    Returns:
        conc_t: Array [n_acq], concentration time series in mM.

    Raises:
        ValueError: If metabolite name is not found.
    """
    if metabolite not in cfg.metabolite_names:
        raise ValueError(
            f"'{metabolite}' not in metabolite_names: {cfg.metabolite_names}"
        )
    idx = cfg.metabolite_names.index(metabolite)
    return state_means[:, idx]
