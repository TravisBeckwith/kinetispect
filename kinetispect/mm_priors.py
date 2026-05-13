"""
kinetispect/mm_priors.py

Macromolecule baseline priors for KinetiSpect's VAE baseline model.
Ported from Osprey's fit_createSoftConstrOsprey.m
(Oeltzschner, Johns Hopkins University, 2019)

These amplitude ratios between macromolecule (MM) and lipid (Lip)
peaks are derived from the literature and constrain the baseline model
to be physiologically plausible.

They serve two roles in KinetiSpect:
1. As soft constraints during FSL-MRS-style NNLS fitting (direct port)
2. As biochemical consistency loss terms during VAE training

References for the amplitude ratios:
- Behar et al., J Magn Reson 1994
- Gottschalk et al., Magn Reson Med 1999
- Pfeuffer et al., J Magn Reson 1999
- Seeger et al., Magn Reson Med 2003
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Literature-derived MM/Lip amplitude ratios
# Source: Osprey fit_createSoftConstrOsprey.m (Oeltzschner 2019)
# Format: (metabolite_1, metabolite_2, ratio)
# Meaning: amplitude(met_2) ≈ ratio × amplitude(met_1)
# ─────────────────────────────────────────────────────────────────────────────

MM_AMPLITUDE_RATIOS: List[Tuple[str, str, float]] = [
    # Lipid constraints
    ('Lip13', 'Lip09',  0.267),   # Lip09 ≈ 0.267 × Lip13
    ('Lip13', 'Lip20',  0.150),   # Lip20 ≈ 0.150 × Lip13

    # MM constraints relative to MM09 (the 0.9 ppm peak, reference)
    ('MM09',  'MM20',   1.500),   # MM20 ≈ 1.5 × MM09
    ('MM09',  'MM12',   0.300),   # MM12 ≈ 0.3 × MM09
    ('MM09',  'MM14',   0.750),   # MM14 ≈ 0.75 × MM09
    ('MM09',  'MM17',   0.375),   # MM17 ≈ 0.375 × MM09
    ('MM09',  'MM37',   0.890),   # MM37 ≈ 0.89 × MM09
    ('MM09',  'MM38',   0.100),   # MM38 ≈ 0.1 × MM09
    ('MM09',  'MM40',   1.390),   # MM40 ≈ 1.39 × MM09
    ('MM09',  'MM42',   0.300),   # MM42 ≈ 0.3 × MM09

    # NAA/NAAG ratio
    ('NAA',   'NAAG',   0.150),   # NAAG ≈ 0.15 × NAA
]

# Looser constraints (used for validation, not hard priors)
MM_AMPLITUDE_RATIOS_LOOSE: List[Tuple[str, str, float]] = [
    ('MM09',  'MM12',   0.3),
    ('MM09',  'MM14',   0.75),
    ('MM09',  'MM17',   0.375),
    ('MM09',  'MM20',   1.5),
    ('MM09',  'MM22',   1.0),
    ('MM09',  'MM27',   1.0),
    ('MM09',  'MM30',   1.0),
    ('MM09',  'MM32',   1.0),
    ('MM09',  'MM37',   0.89),
    ('MM09',  'MM42',   0.3),
]

# Tolerance on ratios for soft constraint (±fraction of expected ratio)
DEFAULT_RATIO_TOLERANCE = 0.5  # 50% tolerance → soft constraint


def build_soft_constraints(basis_names: List[str],
                            A: np.ndarray,
                            b: np.ndarray,
                            current_amplitudes: np.ndarray,
                            lambda_reg: float = 0.05,
                            ratios: List[Tuple[str, str, float]] = None
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment an NNLS problem with soft amplitude ratio constraints.

    Adds penalty rows to the equation system:
        min ||[A; λ·C] x - [b; λ·d]||²  subject to x ≥ 0

    where C and d encode the ratio constraints:
        C[i] x = d[i]  →  a_j ≈ ratio × a_k

    Direct port of Osprey's fit_createSoftConstrOsprey.m.

    Parameters
    ----------
    basis_names : list of str
        Names of basis functions in order (matching columns of A).
    A : array [n_spectral, n_basis]
        Basis function matrix.
    b : array [n_spectral]
        Spectral data vector.
    current_amplitudes : array [n_basis]
        Current amplitude estimates (for scaling the constraints).
    lambda_reg : float
        Regularization strength. Default 0.05 (Osprey default).
    ratios : list of (str, str, float) or None
        List of (met1, met2, ratio) constraints. If None, uses
        MM_AMPLITUDE_RATIOS.

    Returns
    -------
    A_augmented : array [n_spectral + n_constraints, n_basis]
    b_augmented : array [n_spectral + n_constraints]
    """
    if ratios is None:
        ratios = MM_AMPLITUDE_RATIOS

    name_to_idx = {name: idx for idx, name in enumerate(basis_names)}
    n_basis = A.shape[1]
    n_spectral = A.shape[0]

    penalty_rows_A = []
    penalty_rows_b = []

    for met1, met2, ratio in ratios:
        idx1 = name_to_idx.get(met1)
        idx2 = name_to_idx.get(met2)

        if idx1 is None or idx2 is None:
            continue  # skip if metabolite not in basis

        # Constraint: amp[idx2] = ratio * amp[idx1]
        # → amp[idx2] - ratio * amp[idx1] = 0
        # As penalty row: lambda * (amp[idx2] - ratio * amp[idx1]) = 0
        row = np.zeros(n_basis)
        row[idx2] = lambda_reg
        row[idx1] = -lambda_reg * ratio

        # Scale by current amplitude for numerical stability
        scale = max(abs(current_amplitudes[idx1]), 1e-6)
        row /= scale

        penalty_rows_A.append(row)
        penalty_rows_b.append(0.0)

    if len(penalty_rows_A) == 0:
        return A, b

    C = np.array(penalty_rows_A)
    d = np.array(penalty_rows_b)

    A_augmented = np.vstack([A, C])
    b_augmented = np.concatenate([b, d])

    return A_augmented, b_augmented


def mm_consistency_loss(amplitudes: Dict[str, float],
                         ratios: List[Tuple[str, str, float]] = None,
                         tolerance: float = DEFAULT_RATIO_TOLERANCE
                         ) -> float:
    """
    Compute biochemical consistency loss for VAE training.

    For each amplitude ratio constraint (met1, met2, ratio):
        loss += max(0, |amp[met2]/amp[met1] - ratio| - tolerance*ratio)²

    This is a soft hinge loss — zero when the ratio is within tolerance,
    quadratic when violated.

    Parameters
    ----------
    amplitudes : dict {metabolite_name: amplitude}
        Current amplitude estimates from the VAE decoder.
    ratios : list or None
        Ratio constraints. Defaults to MM_AMPLITUDE_RATIOS.
    tolerance : float
        Fractional tolerance. Default 0.5 (50%).

    Returns
    -------
    loss : float
        Total consistency loss (minimize during VAE training).
    """
    if ratios is None:
        ratios = MM_AMPLITUDE_RATIOS

    loss = 0.0
    for met1, met2, ratio in ratios:
        amp1 = amplitudes.get(met1, 0.0)
        amp2 = amplitudes.get(met2, 0.0)

        if amp1 <= 0:
            continue

        actual_ratio = amp2 / amp1
        violation = abs(actual_ratio - ratio) - tolerance * ratio
        if violation > 0:
            loss += violation ** 2

    return loss


def mm_consistency_loss_torch(amplitudes_tensor, basis_names,
                               ratios=None, tolerance=DEFAULT_RATIO_TOLERANCE):
    """
    PyTorch-compatible version for VAE training with autograd.

    Parameters
    ----------
    amplitudes_tensor : torch.Tensor [n_basis]
        Decoded amplitudes from VAE.
    basis_names : list of str
        Names corresponding to amplitude dimensions.
    ratios : list or None
    tolerance : float

    Returns
    -------
    loss : torch.Tensor (scalar)
    """
    import torch

    if ratios is None:
        ratios = MM_AMPLITUDE_RATIOS

    name_to_idx = {name: idx for idx, name in enumerate(basis_names)}
    loss = torch.tensor(0.0, requires_grad=False)

    for met1, met2, ratio in ratios:
        idx1 = name_to_idx.get(met1)
        idx2 = name_to_idx.get(met2)
        if idx1 is None or idx2 is None:
            continue

        amp1 = amplitudes_tensor[idx1]
        amp2 = amplitudes_tensor[idx2]

        # Soft hinge: penalize when ratio is violated beyond tolerance
        # Use smooth approximation for gradient stability
        eps = 1e-6
        actual_ratio = amp2 / (amp1 + eps)
        deviation = (actual_ratio - ratio).abs() - tolerance * ratio
        loss = loss + torch.relu(deviation) ** 2

    return loss


@dataclass
class MMPriorConfig:
    """
    Configuration for the macromolecule prior in KinetiSpect's VAE.

    Passed to the VAE trainer to configure the biochemical consistency loss.
    """
    # Amplitude ratio constraints
    ratios: List[Tuple[str, str, float]] = field(
        default_factory=lambda: MM_AMPLITUDE_RATIOS)

    # Loss weight in the combined VAE objective
    # Total loss = reconstruction_loss + lambda_mm * mm_consistency_loss
    lambda_mm: float = 0.1

    # Tolerance on ratio constraints (fraction of expected ratio)
    tolerance: float = DEFAULT_RATIO_TOLERANCE

    # NNLS soft constraint regularization strength
    lambda_nnls: float = 0.05

    # Minimum amplitude for a MM peak to be considered in ratio constraints
    # (prevents division by near-zero amplitudes)
    min_amplitude: float = 1e-6

    def validate(self, basis_names: List[str]) -> List[str]:
        """Check which ratio constraints can be applied given the basis set."""
        name_set = set(basis_names)
        applicable = []
        skipped = []
        for met1, met2, ratio in self.ratios:
            if met1 in name_set and met2 in name_set:
                applicable.append((met1, met2, ratio))
            else:
                skipped.append(f'{met1}/{met2}')
        if skipped:
            print(f'MM prior: skipping constraints (not in basis): '
                  f'{", ".join(skipped)}')
        return applicable


# Standard MM basis names used in Osprey/FSL-MRS
STANDARD_MM_NAMES = [
    'MM09', 'MM12', 'MM14', 'MM17', 'MM20',
    'MM22', 'MM27', 'MM30', 'MM32', 'MM37',
    'MM38', 'MM40', 'MM42',
    'Lip09', 'Lip13', 'Lip20',
]

# Expected MM concentration ratios relative to tCr (from Seeger et al. 2003)
# Used as VAE latent space anchor points during training
MM_ABSOLUTE_RATIOS_TO_TCR = {
    'MM09': 2.0,   # mM equivalent (approximate)
    'MM12': 0.6,
    'MM14': 1.5,
    'MM17': 0.75,
    'MM20': 3.0,
    'MM37': 1.78,
    'MM40': 2.78,
}
