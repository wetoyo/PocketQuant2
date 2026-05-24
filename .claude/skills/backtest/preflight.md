# Pre-flight Checklist

Before writing any code, answer every question below. If any answer is "no", stop and fix the data source first.

| Check | Rule (rules.md §) | Pass condition |
|:---|:---:|:---|
| Data covers delisted / acquired tickers (if testing a universe) | §1 | No survivorship bias |
| Using **adjusted** prices (splits + dividends accounted for) | §1 | `adjusted=True` or using `ADJ CLOSE` |
| All data sources share the **same timezone** | §1 | UTC or all `tz_localize(None)` |
| Data frequency matches the strategy's execution window | §1 | e.g., daily strategy → `interval="1d"` |
| Features are **pre-shifted** and contain no future information | §2 | `build_features` handles this automatically — do **not** apply an extra `.shift(1)` |
| OOS test set was **never touched** during parameter tuning | §2 | Parameters fit on IS data only |
