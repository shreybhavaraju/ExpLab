# Pass conditions

Written down before running any of the checks, so the results can't quietly move the criteria.
Results go in `results/` and the README.

The Monte Carlo checks use 1,000 reps. A rate near 5% (or 95%) estimated from 1,000 reps has a
standard error of about 0.7 points, so the bands below are roughly +-2 to 2.5 SE. Anything outside
the band counts as a fail even if it "looks close".

| What | How it's checked | Pass if |
|---|---|---|
| Readout CIs | inject a known lift into resampled control data, 1,000 reps | coverage between 93.5% and 96.5%, for both the absolute-diff CI and the delta-method lift CI |
| Stats functions | pytest against statsmodels / scipy on small inputs | match to 1e-6 |
| SRM check | randomly drop 2% of one arm | check fails (p < 0.001) |
| SRM check | the untouched data, and random fake splits | passes |
| A/A false positives | 1,000 random 50/50 splits of the control group | share of p < 0.05 between 3.5% and 6.5%, p-value histogram flat (KS test against uniform, p > 0.01) |
| CUPAC | same A/A test with the adjustment on | still between 3.5% and 6.5%; CI on the real data narrower than without it |
| Peeking | sequential looks every 5% on A/A data, 1,000 reps | naive "stop at first p < 0.05" FPR above 10%; corrected test between 3.5% and 6.5% (an always-valid test is allowed to be lower, it just can't be above 6.5%) |
| Uplift models | semi-synthetic data with an effect planted in one segment | planted segment gets the highest average predicted uplift, and the Qini curve beats random targeting (positive area between them) |
| Decision layer | unit test for each rule | correct verdict and reason every time |

SRM threshold is p < 0.001 (the usual alarm level, since with millions of users a 0.05 threshold
would flag tiny harmless imbalances all the time).
