# Bayesian readout for a 0/1 metric. Beta prior + binomial data gives a Beta posterior per arm,
# then the lift and the expected loss come from Monte Carlo draws (no nice closed form for those).

import numpy as np


def beta_posterior(successes, n, prior=(1, 1)):
    """Beta(a, b) prior and k successes out of n gives a Beta(a + k, b + n - k) posterior.
    The default (1, 1) is flat on [0, 1], so it's worth about 2 fake users and does nothing
    at Criteo sizes."""
    a, b = prior
    return a + successes, b + n - successes


def bayes_readout(k_t, n_t, k_c, n_c, prior=(1, 1), draws=200_000, alpha=0.05, seed=0):
    """Draw p_t and p_c from their posteriors and summarize:

        prob_better    P(p_t > p_c | data), the chance treatment is really better
        lift_mean      posterior mean of p_t / p_c - 1
        lift_ci        equal-tailed credible interval: given the data, the lift is in here
                       with probability 1 - alpha
        expected_loss  E[max(p_c - p_t, 0)], the rate we give up on average if we ship and
                       treatment is actually worse (counts as 0 when it's better)

    A p-value is P(data at least this extreme | no effect), it says nothing directly about how
    likely treatment is to be better. prob_better is exactly that. With a flat prior and big arms
    they line up anyway: prob_better ~ Phi(z) for the usual z of the diff, so p = 0.05
    (two-sided, positive effect) is about prob_better = 0.975.
    """
    rng = np.random.default_rng(seed)
    p_t = rng.beta(*beta_posterior(k_t, n_t, prior), draws)
    p_c = rng.beta(*beta_posterior(k_c, n_c, prior), draws)
    lift = p_t / p_c - 1
    lo, hi = np.quantile(lift, [alpha / 2, 1 - alpha / 2])
    return {
        'prob_better': float(np.mean(p_t > p_c)),
        'lift_mean': float(lift.mean()),
        'lift_ci': [float(lo), float(hi)],
        'expected_loss': float(np.maximum(p_c - p_t, 0).mean()),
    }
