# kinetispect (Beta v0.1)

Kinetic spectral estimation for functional MRS.

## Status

Pre-alpha. Phase 0–1 development.

## Installation (development)

From the repository root:

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[torch]"   # for VAE baseline (Phase 2)
pip install -e ".[dev]"     # for testing
```

## Modules

- `nuisance` — Per-acquisition frequency, phase, and linewidth correction
- `crlb` — Cramér-Rao lower bound computation
- `mm_priors` — Macromolecule baseline amplitude constraints
- `kalman` — Coming in Phase 2
- `vae_baseline` — Coming in Phase 2

## Quick test

```python
import kinetispect as ks
print(ks.__version__)
print(ks.MM_AMPLITUDE_RATIOS[0])
```
