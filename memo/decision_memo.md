# Decision memo: do the display ads drive site visits?

Shrey Bhavaraju, Oct 5 2026

## The question

Criteo held 15% of users out of ad targeting and made the other 85% eligible (13.98M users total). Do the ads increase visits to the advertiser's site, and by how much? Visits are the primary metric, conversions are secondary (too rare to be primary: 0.19% in the holdout), and conversions per visit is the guardrail, since ads that bring in junk visits would show up there.

## Recommendation

**Don't size anything off this readout. ExpLab's verdict is INCONCLUSIVE.** Every version of the estimate says the ads increase visits, so there's no case for turning them off. But the size of the effect is what a budget decision depends on, and this dataset can't pin it down: the raw lift in visits is +27%, adjusting for a problem in the assignment brings it to +18 to 20%, and part of the problem can't be adjusted for at all. My recommendation is to keep the current setup and get the per-test data (details below) before deciding anything about spend.

## Evidence

- **The raw readout looks great.** Visits go from 3.82% to 4.85% (+27%, 95% CI +26% to +28%), conversions from 0.19% to 0.31% (+59%), and conversions per visit also go up (+25%), so the extra visits aren't low quality. This isn't a power problem either: the test could detect a lift as small as 1%.
- **The overall split checks out, almost too well.** 85.00001% of users are treated, right on the design. An A/A test on the holdout gives 4.7% false positives, as it should.
- **But the arms differ before treatment.** Inside slices of the user features (which are recorded before treatment), the treated share ranges from 84.6% to 87.7%, far outside chance, and it's highest among users who visit the most anyway. Criteo built the dataset by pooling several tests that had different treatment ratios, subsampling each to the same 85%, and also subsampling non-visitors non-uniformly. Whatever the exact mechanism, treated users end up over-represented among heavy visitors, so part of the raw +27% is just "the treated group had more heavy visitors to begin with".
- **Adjusting for it takes a quarter to a third off, but can't finish the job.** Comparing within feature slices gives +20%, and a model-based adjustment (CUPAC) gives +18%. Both only correct for the imbalance that shows up in the features we have. The subsampling was done partly by outcome, which no adjustment on the features can undo. That's the reason for INCONCLUSIVE.
- **The effect is concentrated.** On held-out users, the 20% with the highest predicted uplift account for about 77% of the incremental visits. The top tenth of users gain about 6 points of visit rate; the bottom half gain close to nothing.

## Why not compare exposed users to the holdout?

Only 3.6% of eligible users actually saw an ad. Those users visit at 41%, vs 3.8% in the holdout, which would make the ad look like a 10x effect. It isn't: users get exposed by being active enough to show up in an ad auction, so exposure mostly measures who was going to visit anyway. That's why the readout compares by assignment (intent-to-treat), which the randomization protects. The effect on users who actually saw an ad can still be recovered from it: since holdout users can't be exposed, it's the ITT effect divided by the exposure rate, 1.03 points / 3.6%, or about +29 points of visit probability for an exposed user. That assumes being eligible only matters through actually seeing an ad, and it inherits the same imbalance problem.

## Risks

- Acting on the raw +27% would overstate the incremental visits by roughly 30 to 40% (1.03 points raw vs 0.74 to 0.80 adjusted) and over-allocate budget.
- Criteo subsampled non-visitors for privacy, so even the rates themselves aren't real-world rates. Any ROI math off this data is shaky regardless of the imbalance.
- The targeting numbers (77% of the gain from the top 20%) come from the same data and are probably a bit optimistic for the same reason.
- If a rerun is monitored daily and stopped at the first significant result, the false positive rate goes from 5% to ~27% (simulated). Any rerun needs a planned stopping rule.

## What I'd test next

1. **Rerun the readout per incrementality test**, with the test ID and without the label subsampling. The fix for different treatment ratios is to estimate within each test and then combine. That should give one trustworthy number.
2. **If that data isn't available, run a fresh holdout:** fixed 85/15 split, sample ratio and feature balance checked daily, O'Brien-Fleming boundaries for the planned looks, visits as the primary metric and conversions per visit as the guardrail.
3. **Test targeting directly:** ads only for the top 20% by predicted uplift vs ads for everyone, comparing cost per incremental visit. The uplift model suggests most of the value is concentrated, which would mean most of the spend is on users who don't respond.
