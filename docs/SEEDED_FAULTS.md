# Seeded faults

## Exercise 1: Failing feature test

I added `test_rolling_max_excludes_today` to `tests/test_leakage.py` before implementing `cnt_roll_max_7`. For January 19, 2011, the previous seven calendar days had counts of 1162, 1406, 1421, 1248, 1204, 1000, and 683. The expected maximum is **1421**. January 19 itself had **1650** rides, so including the forecast day would make the test fail.

The initial pytest run failed with `AttributeError` at the assertion for `cnt_roll_max_7` because the feature column had not been implemented yet.

### Actual pytest output

```text
F                                                                        [100%]
=================================== FAILURES ===================================
_______________________ test_rolling_max_excludes_today ________________________

feats =         dteday  instant  season  ...  cnt_lag_1 cnt_lag_7   cnt_roll_7
0   2011-01-10     10.0  WINTER  ...      822.0...    1787.0  1530.142857
713 2012-12-31    731.0  WINTER  ...     1796.0     920.0  1531.428571

[714 rows x 28 columns]

    def test_rolling_max_excludes_today(feats):
        """cnt_roll_max_7 must be the max of the 7 calendar days *before* today.
    
        Jan 12-18 2011 were 1162, 1406, 1421, 1248, 1204, 1000, 683; Jan 19
        itself was 1650, so a window that leaked today would return 1650.
        """
        row = feats.set_index("dteday").loc[pd.Timestamp("2011-01-19")]
        assert row[TARGET] == 1650
>       assert row.cnt_roll_max_7 == 1421
               ^^^^^^^^^^^^^^^^^^

tests/test_leakage.py:68: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = instant                    19.0
season                   WINTER
yr                       2011.0
mnth                  ...         683.0
cnt_lag_7                1162.0
cnt_roll_7          1160.571429
Name: 2011-01-19 00:00:00, dtype: object
name = 'cnt_roll_max_7'

    @final
    def __getattr__(self, name: str):
        """
        After regular attribute access, try looking up the name
        This allows simpler access to columns for interactive use.
        """
        # Note: obj.x will always call obj.__getattribute__('x') prior to
        # calling obj.__getattr__('x').
        if (
            name not in self._internal_names_set
            and name not in self._metadata
            and name not in self._accessors
            and self._info_axis._can_hold_identifiers_and_holds_name(name)
        ):
            return self[name]
>       return object.__getattribute__(self, name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       AttributeError: 'Series' object has no attribute 'cnt_roll_max_7'. Did you mean: 'cnt_roll_7'?

.venv/lib/python3.14/site-packages/pandas/core/generic.py:6194: AttributeError
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_rolling_max_excludes_today - AttributeErro...
1 failed in 0.29s

```


## Fault 1: Allowing a forbidden column

I temporarily changed `feature_columns()` in `src/features.py` to allow `casual` through the `FORBIDDEN` filter. The test `test_no_forbidden_features` failed with `AssertionError: forbidden columns in feature set: {'casual'}`. The isolated run ended with 1 failed and 8 passed.

## Fault 2: Rolling mean includes the forecast day

I temporarily removed `.shift(1)` from `cnt_roll_7` in `src/features.py`. This made the rolling window include the forecast day's ride count and drop the oldest of the seven prior days. The forecast day's count is the target, so using it as an input leaks information.

### Code change

```diff
-    out["cnt_roll_7"] = out[TARGET].shift(1).rolling(LAG_WARMUP).mean()
+    out["cnt_roll_7"] = out[TARGET].rolling(LAG_WARMUP).mean()
```

### Pytest result

`test_rolling_mean_excludes_today` failed at `tests/test_leakage.py:57`. The rolling mean was `4389.0`, but the expected mean from the seven previous calendar days was `4414.857142857143`. The run ended with **1 failed, 8 passed**.
```

## Fault 3: Unknown weather labels are silently accepted

I temporarily removed the check in `encode_weather()` that raises a `ValueError` for unknown weather labels. Without this check, `.map()` converts an unknown label to `NaN`, and the later `dropna()` can silently remove that row.

### Code change

```diff
-    known = series.dropna()
-    unknown = set(known.unique()) - set(WEATHER_ORDER)
-    if unknown:
-        raise ValueError(f"unmapped weather labels: {sorted(unknown)}")
```

### Pytest result

`test_unknown_weather_label_raises` failed with `Failed: DID NOT RAISE ValueError`.

### Restoration

I restored the unknown-weather check. Claude took [number] turn(s) to fix the fault. After restoration, `make test` reported [actual result].


- 2026-11-25 — Exercise 4: I trained HGB to predict `cnt - cnt_lag_1` and added `cnt_lag_1` back to each prediction before scoring. This is defensible for forecasts made after yesterday's final count is available. On validation, the transformation raised HGB R² from 0.1428 to 0.5627 and reduced MAE from 1336.0 to 891.0 rides. It beat persistence on both metrics, but ridge still led with R² of 0.7072 and MAE of 731.9 rides. I kept delta HGB as a separate experiment and retained ridge as my preferred model based on validation.