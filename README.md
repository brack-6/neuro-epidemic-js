# Neuro-Epidemic Engine (NEE)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20174053.svg)](https://doi.org/10.5281/zenodo.20174053)

**Dan Baker** — Independent Research, Bogotá, Colombia

Gradient descent on differentiable SIR epidemic models cannot recover transmission rate (beta) and recovery rate (gamma) independently from aggregate prevalence data. The loss landscape is degenerate along the R0 = beta/gamma manifold.

Fixing gamma via a clinical prior resolves the degeneracy. Beta then converges to within 4% of the true value.

**A low training loss is not evidence of correct parameter recovery.**

## Result

| Condition | Loss | Learned R0 | True R0 |
|-----------|------|-----------|---------|
| Free optimization | ~2e-6 | 2.93 (diverging) | 2.20 |
| Gamma fixed (clinical prior) | ~1e-6 | 2.12 | 2.20 |

## Run

```bash
node neuro_epidemic.js
```

## Paper

Baker, D. (2026). Gradient Descent Cannot Identify SIR Parameters from Prevalence Data Alone. Zenodo. https://doi.org/10.5281/zenodo.20174053

## Donate

[PayPal](https://paypal.me/bakermoto)
