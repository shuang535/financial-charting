# 回測績效表

## 共通資料契約

- 輸入是 decimal periodic returns 的 DataFrame；欄名可為策略或任意標的，例如 `TWSE`、`AGG Bond`、`Cash`。
- 所有欄位使用同一個完整樣本，索引必須是排序且不重複的 `DatetimeIndex`。Helper 不讀資料庫、不補值、不猜頻率。
- `periods_per_year` 由呼叫端明確指定：月資料通常為 12，日資料依研究市場設定。輸出維持 decimal，百分比格式留給 Notebook 或簡報層。
- 單一指數沒有策略換倉，turnover 留空；只有能提供完整 target weights 的策略才計算。

## 可呼叫的表格

### `build_performance_summary`

每個輸入欄位一列，包含 Total Return、幾何 CAGR、年化波動、Sharpe、負值 MDD、Calmar、正報酬期比例、觀測數與選配的年化 turnover。

- Sharpe 使用樣本標準差 `ddof=1`；`annual_risk_free_rate` 先轉成每期利率。
- MDD 的歷史高點包含樣本開始前的初始財富 1，避免漏掉第一期虧損。
- `target_weights` 使用 `{策略欄名: 權重 DataFrame}`。Turnover 預設單邊，即 `0.5 × Σ|Δw|`；若改用雙邊，欄名會明確標示。
- 這是 target-weight change turnover。若專案已有 drift-adjusted pre-trade weights，應在專案端先算更精確的 turnover，不要混用定義。

### `build_annual_performance_table`

將每個標的的期間報酬按曆年複利。預設排除不足 `periods_per_year` 的年度；日資料可用 `minimum_periods` 指定完整年度門檻。指定 Benchmark 時，另外產生各欄相減的年度超額報酬。

### `build_benchmark_win_summary`

彙總各標的相對 Benchmark 的勝出年數、勝率、平均與中位超額、最佳／最差年度、Benchmark 負報酬年度的勝出次數，以及落後年度清單。

### `build_regime_return_table`

依 Regime 回傳各標的報酬與 `sample_n`，可選平均、中位、區間複利或按觀測期數年化。`regime_groups` 可合併底層狀態；同一月份可刻意出現在不同彙總群組，因此要在圖說交代群組定義。

### `build_drawdown_episodes`

接受多個標的，一次列出每段回撤的 peak、首次落水、trough、recovery、最大回撤與水下期數。尚未修復的 episode 使用 `NaT` 並標記 `recovered=False`。

## 最小使用方式

```python
from financial_charting import (
    build_annual_performance_table,
    build_benchmark_win_summary,
    build_drawdown_episodes,
    build_performance_summary,
    build_regime_return_table,
)

returns = monthly_returns[["Regime策略", "Benchmark", "TWSE", "AGG Bond"]]

summary = build_performance_summary(
    returns,
    periods_per_year=12,
    target_weights={"Regime策略": strategy_target_weights},
    turnover_sides="one_way",
)

annual = build_annual_performance_table(
    returns,
    benchmark="Benchmark",
)

wins = build_benchmark_win_summary(
    returns[["Regime策略", "TWSE", "AGG Bond", "Benchmark"]],
    benchmark="Benchmark",
)

regime_returns = build_regime_return_table(
    returns,
    monthly_regimes,
    statistic="median",
    regime_groups={
        "Growth": ("G|E", "G|T"),
        "Contraction": ("C|E", "C|T"),
        "Easing": ("G|E", "C|E"),
        "Tightening": ("G|T", "C|T"),
        "G|E": ("G|E",),
        "G|T": ("G|T",),
        "C|E": ("C|E",),
        "C|T": ("C|T",),
    },
)

episodes = build_drawdown_episodes(returns[["Regime策略", "TWSE", "AGG Bond"]])
```

Table builder 只負責可稽核計算，不套用顏色、不四捨五入、不寫 Excel。由呼叫端選欄位、格式化百分比並決定簡報版或研究版顯示內容。
