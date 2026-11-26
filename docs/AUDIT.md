# Peer leakage audit

- Repository: https://github.com/Capybarachen/bikeshare-open
- Commit reviewed: `d00eba98c346f1f23eba3c2653e71165b82ce373`
- Auditor: Claude Code `leakage-auditor` subagent

## Auditor output

 I read src/config.py, features.py, train.py, ingest.py, the two experiment scripts and tests/test_contract.py. I ran build_features on the raw CSV. I did not run `src.train` or pytest. The processed
  parquet is absent, so I could not recompute metrics.json. The metrics file is consistent with my build: 728 raw rows give 714 feature rows, and n_train=348 matches.

  1. **Target leakage.** Pass. `feature_columns` (features.py:92-95) drops everything in FORBIDDEN. The 15 features are temp, atemp, hum, windspeed, the calendar and weather terms, temp_x_workday, cnt_lag_1,
  cnt_lag_7 and cnt_roll_7. None restates `casual + registered`, and none is a time index or cnt_2d_bfr.
  2. **Temporal ordering.** Pass. `cnt_roll_7` is `shift(1).rolling(7)` (features.py:87). The split is by date (train.py:78-84). The only fitted transformer is the StandardScaler inside the ridge pipeline,
  which is fit on `train[cols]` only. train.py never tunes on test. The experiments discard the test split.
  3. **Calendar gaps.** Pass. `reindex_to_calendar` runs before any lag (features.py:57). The grid has 729 days and the raw data has 728, so only 2011-03-10 is missing.
  4. **Silent row loss.** Pass. Raw data has no NaN, and `encode_weather` raises on unknown labels. The loss is 14 rows: 7 for the warm-up and 7 for the gap. Both are enumerated by the bound in
  `test_only_expected_rows_are_dropped` (tests/test_contract.py:29-44), which allows 14.
     - The bound lives only in the test. `build_features` has no assertion of its own, so `python -m src.features` would not catch an excess loss. That does not change any current number.
  5. **Unreported baseline.** Pass. `MODELS` (train.py:69-75) includes `baseline_mean` and `baseline_persistence`. Both are scored on valid and test in metrics.json (persistence test R2 0.51, ridge 0.59, hgb
  -0.58, hgb_ratio 0.63). The experiment scripts print persistence next to the models they score.


What it did and didn't do: it read config.py, features.py, train.py, ingest.py, the two experiment scripts and tests/test_contract.py. It ran build_features on the raw CSV. It did not run src.train or
  pytest. The processed parquet is absent, so it couldn't recompute metrics.json. It did check that the file is consistent with its own build: 728 raw rows give 714 feature rows, and n_train=348.

  ┌─────────────────┬────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │      Check      │ Result │                                                                                  Evidence                                                                                   │
  ├─────────────────┼────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Target leakage  │ Pass   │ feature_columns drops FORBIDDEN (features.py:92-95). The 15 features include no casual, registered, time index or cnt_2d_bfr.                                               │
  ├─────────────────┼────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Temporal        │ Pass   │ cnt_roll_7 is shift(1).rolling(7) (features.py:87). The split is by date (train.py:78-84). The only fitted transformer is the ridge StandardScaler, fit on train only.      │
  │ ordering        │        │ There is no tuning on test.                                                                                                                                                 │
  ├─────────────────┼────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Calendar gaps   │ Pass   │ reindex_to_calendar runs before any lag (features.py:57). The grid has 729 days against 728 raw rows, so only 2011-03-10 is missing.                                        │
  ├─────────────────┼────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Silent row loss │ Pass   │ encode_weather raises on unknown labels. 14 rows are lost, 7 to warm-up and 7 to the gap. That fits the bound of 14 in tests/test_contract.py:29-44.                        │
  ├─────────────────┼────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Baselines       │ Pass   │ MODELS includes baseline_mean and baseline_persistence (train.py:69-75). Both are scored on valid and test.                                                                 │
  └─────────────────┴────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  One note from the auditor: the row-loss bound exists only in the test. build_features has no assertion of its own, so python -m src.features wouldn't catch an excess loss. That doesn't change any current
  number.

  Test results the auditor reported from metrics.json:
  - persistence R² 0.51
  - ridge 0.59
  - hgb −0.58
  - hgb_ratio 0.63


## My verdict

I agree with the auditor that the review found no issue that changes the peer repository's current results. The audit found that the feature filter excludes forbidden columns, the rolling mean uses only prior days, the splits follow chronological order, and both the mean and persistence baselines are scored. The feature build produced 714 rows from 728 raw rows. The 14 dropped rows matched the expected loss from the seven-day warm-up and the missing calendar day. The row-loss limit is enforced by a test, so running the feature builder alone could miss excess row loss on future data. The auditor rebuilt the features but did not rerun training or pytest, so the scores read from the peer's `metrics.json` were not independently reproduced.