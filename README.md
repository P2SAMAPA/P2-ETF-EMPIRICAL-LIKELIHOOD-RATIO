# Empirical Likelihood Ratio Engine for ETFs

Applies empirical likelihood (Owen 2001) to test for positive mean return conditional on macro regimes (e.g., VIX high/low). The per‑ETF score is the likelihood ratio statistic for H₀: μ ≤ 0 vs H₁: μ > 0, computed on the subsample defined by the current macro state.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Uses a primary macro variable (e.g., VIX) to define regimes (median split)
- Empirical likelihood ratio test for positive mean
- Non‑parametric, valid under any distribution
- Score = likelihood ratio statistic (higher = stronger evidence of positive mean)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-empirical-likelihood-ratio-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py`
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High likelihood ratio → strong evidence that the ETF's conditional mean return is positive given today's macro regime.
- Low ratio → evidence for zero or negative mean.

## Requirements

See `requirements.txt`.
