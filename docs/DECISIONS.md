# Decisions

One line per methodological decision: what you chose, why, and what it changed.
Agents append here via the `/new-feature` skill; you append here by hand.

- Exclude `casual`, `registered`: they sum to `cnt` exactly. Leaky model R^2 1.000 -> honest ridge 0.583.
- Exclude `cnt_2d_bfr`: a shipped 2-day lag we cannot test the alignment of. Build our own lags instead.
- Exclude `instant`, `yr`, `days_since_2011`: time indexes let a tree predict a flat constant past 2011 and a linear model extrapolate a trend we have no grounds for.
- Split 2011 / H1 2012 / H2 2012 by date. Test set touched once, after all feature work.
- Report mean and persistence baselines with every score. Persistence (test R^2 0.510, MAE 840) is within 0.07 R^2 of ridge and beats it on MAE; that caveat goes in the report.
- 2026-11-25 — Exercise 1: Keep `cnt_roll_max_7` for ridge. It uses counts from the seven completed calendar days before the forecast, so all inputs are available at prediction time. On validation, ridge R^2 increased from 0.6475 to 0.7072 (+0.0597), and MAE fell from 832.9 to 731.9 rides. HGB performed better without the feature: its R^2 fell from 0.2185 to 0.1428 (−0.0757), and MAE rose from 1296.6 to 1336.0 rides. Both baselines were unchanged.