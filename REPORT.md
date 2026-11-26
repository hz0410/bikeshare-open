## Exercise 4: target transformation

I compared two versions of histogram gradient boosting (HGB). The first predicted daily ride counts directly. The second predicted the change from the previous day, `cnt - cnt_lag_1`. I added yesterday's count back to each predicted change so all models were scored on the original ride-count scale. All four models used the same chronological split and evaluation rows.

| Model | Validation R² | Validation MAE (rides) |
|---|---:|---:|
| HGB predicting daily counts | 0.1428 | 1336.0 |
| HGB predicting the change from yesterday | 0.5627 | 891.0 |
| Ridge | 0.7072 | 731.9 |
| Persistence (predict yesterday's count) | 0.4507 | 900.6 |

Predicting the daily change improved HGB's validation R² by 0.4199 and reduced its MAE by 445.0 rides. Yesterday's count gives the model a starting point for the current demand level, leaving HGB to estimate the daily adjustment. The transformation is appropriate if yesterday's final count is available when the forecast is issued. Delta HGB beat persistence by only 9.6 rides in MAE, and ridge still had the best R² and MAE. Based on validation, I would keep ridge as the preferred model and report delta HGB as a separate experiment.

## Exercise 5: ridge vs persistence

version model valid_R2 valid_MAE test_R2 test_MAE
before baseline_mean -0.9318 2002.2 -2.1441 2847.7
before baseline_persistence 0.4507 900.6 0.5099 840.1
before ridge 0.6475 832.9 0.5833 947.1
before hgb 0.2185 1296.6 -0.4975 1972.6
after baseline_mean -0.9318 2002.2 -2.1441 2847.7
after baseline_persistence 0.4507 900.6 0.5099 840.1
after ridge 0.7072 731.9 0.6335 867.2
after hgb 0.1428 1336.0 -0.6976 2105.6