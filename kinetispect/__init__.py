"""
kinetispect/

KinetiSpect: Kinetic spectral estimation for functional MRS.
Three novel components on top of FSL-MRS infrastructure.
"""

from .nuisance import spectral_registration, navigator_nuisance, estimate_bold_linewidth_coupling
from .crlb import compute_crlb, compute_jacobian, CRLBResult, crlb_quality_flag
from .mm_priors import (
    MM_AMPLITUDE_RATIOS, build_soft_constraints,
    mm_consistency_loss, mm_consistency_loss_torch,
    MMPriorConfig, STANDARD_MM_NAMES
)

__version__ = '0.1.0'
__all__ = [
    'spectral_registration',
    'navigator_nuisance',
    'estimate_bold_linewidth_coupling',
    'compute_crlb',
    'compute_jacobian',
    'CRLBResult',
    'crlb_quality_flag',
    'MM_AMPLITUDE_RATIOS',
    'build_soft_constraints',
    'mm_consistency_loss',
    'mm_consistency_loss_torch',
    'MMPriorConfig',
    'STANDARD_MM_NAMES',
]
