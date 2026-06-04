import numpy as np
from scipy.optimize import minimize_scalar
import config  # <-- added import

def empirical_likelihood_ratio(y, mu0=0):
    """
    Compute empirical likelihood ratio for H0: E[y] = mu0.
    Returns the log-likelihood ratio statistic (2 * log(sup L / L0)).
    """
    n = len(y)
    if n == 0:
        return 0.0
    # Solve for lambda such that sum (y_i - mu0) / (1 + lambda (y_i - mu0)) = 0
    def obj(lambda_, y, mu0):
        denom = 1 + lambda_ * (y - mu0)
        if np.any(denom <= 0):
            return 1e10
        return np.sum(np.log(denom))
    # Find lambda (maximizer of empirical likelihood)
    res = minimize_scalar(lambda lam: obj(lam, y, mu0), bounds=(-0.99, 0.99), method='bounded')
    lam_opt = res.x
    # Compute log-likelihood ratio
    llr = -2 * np.sum(np.log(1 + lam_opt * (y - mu0)))
    return max(llr, 0.0)

def test_positive_mean(y):
    """
    Test H0: mean <= 0 vs H1: mean > 0 using empirical likelihood.
    Returns the likelihood ratio statistic for H0: mean = 0 (which is the least favourable under H0).
    """
    return empirical_likelihood_ratio(y, mu0=0)

def el_ratio_score(returns, macro_today, macro_series, threshold_type='median'):
    """
    Split macro into two regimes (low/high) based on median (or other threshold).
    For the current macro value, select the corresponding subsample of returns.
    Compute empirical likelihood ratio statistic for positive mean.
    """
    if len(returns) < config.MIN_OBS:
        return 0.0
    # Determine macro regime split point
    if threshold_type == 'median':
        split = np.median(macro_series)
    else:
        split = np.percentile(macro_series, threshold_type)
    # Indicator: 1 if macro > split
    high_macro = (macro_series > split).astype(bool)
    # Current regime
    current_high = macro_today > split
    # Select subsample
    if current_high:
        subsample = returns[high_macro]
    else:
        subsample = returns[~high_macro]
    if len(subsample) < config.MIN_OBS:
        return 0.0
    # Test for positive mean (H0: mean <= 0)
    stat = test_positive_mean(subsample)
    return stat
