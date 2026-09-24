# Regime 與投資組合研究圖表

## 圖型選擇

| 研究問題 | 優先圖型 |
|---|---|
| 策略長期是否勝過 Benchmark、何時領先或落後 | 上層累計報酬、下層相對累計報酬 |
| 策略在哪些期間失效或持續落後 | 相對報酬、relative drawdown、rolling relative return 三層診斷圖 |
| 策略績效惡化是否伴隨配置改變 | 累計財富、drawdown、target weights 三層回測圖 |
| Baseline 與 Challenger 何時分歧，事後資產結果如何 | 上層雙訊號、下層可投資結果，分歧月份加底色 |
| Growth、Policy 與資產表現如何串接 | 共用時間軸的三層 Regime dashboard |
| 多個模型最近給出什麼狀態 | 模型 × 月份的離散狀態熱圖 |
| 本期綜合分數為何轉向 | `標準化分數 × 有效權重` 的貢獻熱圖 |
| 總經變數如何影響獲利、估值或債券 | 相同樣本規則的小多圖散點與迴歸線 |

不要在同一張主圖同時回答「長期績效」、「目前訊號」與「本期轉向原因」。10 分鐘投影片通常每張圖只保留一個問題；細部貢獻圖可放在結論頁或備用頁。

簡報版優先保留結論與一個關鍵比較；研究版才展開 drawdown、rolling window、權重與訊號分歧。兩者使用相同底層定義，避免為簡報另算一套績效。

## 資料契約與時間語意

- 投資組合比較函式的 `monthly_returns` 使用 decimal monthly returns，例如 `0.02` 代表 2%。落後期診斷可使用其他固定頻率，但策略與 Benchmark 必須同頻、同日期；所有輸入都要是排序、無重複且沒有缺值的 `DatetimeIndex` 共同樣本。
- 累計報酬為 `cumprod(1 + r) - 1`。
- 相對累計報酬為 `strategy wealth / benchmark wealth - 1`，不是逐月報酬率直接相減後累加。
- Regime 圖的 x 軸預設顯示 allocation／effective return month。若資料日期是 observation month 或 release month，先在繪圖外完成轉換並明確標註。
- 不以 `bfill` 補訊號。沿用已知資料時，另外傳入旗標並以 `†` 或其他符號呈現；真正缺值顯示 `N/A`。
- 多模型比較使用相同資產、配置 mapping、Benchmark、交易成本與共同樣本。模型自己的可用樣本可以另報，但不得與共同樣本績效混在同一比較表。
- Rolling window 的數字代表資料期數，不自動等於月份；月資料可從 12 期開始，日資料不應直接套用 12，研究診斷通常至少 63 期，較穩健時可用 126 期。

## 累計與相對報酬

使用 `assets/financial_charting.py` 的 `plot_cumulative_relative_returns`。上層呈現所有策略的累計報酬，下層省略 Benchmark 本身，只畫各策略相對 Benchmark 的累計財富差。

- Benchmark 使用中性灰或次要線型；主要策略用品牌深藍與較粗線。
- 相對報酬保留零線。線在零以上表示自樣本起點累計領先，而不是當月勝出。
- 若加入 Regime 背景，顏色透明度應低，不得搶過績效線；狀態圖例可放在圖外或交由投影片文字解釋。
- 累計線與相對線共用完全相同的時間範圍。

## 回測與落後期診斷

`plot_backtest_diagnostics` 將累計財富、各策略 drawdown 與單一投資組合的 target weights 放在共同時間軸。它回答「績效惡化時，配置是否同步改變」；若要比較多組權重，分圖處理，不把所有策略權重塞進同一面板。

`plot_underperformance_diagnostics` 專門定位相對 Benchmark 的弱勢期間，並回傳計算後的 DataFrame：

- `relative_return = strategy wealth / benchmark wealth - 1`：自樣本起點累計領先或落後。
- `relative_drawdown = relative wealth / cumulative maximum relative wealth - 1`：相對高點回落幅度。
- rolling relative return：策略與 Benchmark 各自在同一窗口複利後相除；負值區間加淺紅底。

這些圖先用來圈出需追查的月份，再接 Regime、曝險、交易成本或訊號貢獻分析。它們不單獨證明模型失效、結構斷裂或因果關係。

## 訊號確認

`plot_signal_confirmation` 將 Baseline、Challenger 的已完成分數放在上層，資產相對財富或其他可投資 outcome 放在下層；可用 `disagreement_flags` 標出兩者方向不一致的月份。

- 所有輸入必須是已完成 timestamp 對齊的同一樣本；helper 不猜 observation、release 或 allocation month。
- outcome 若是相對財富，參考線設為 `1.0`；若已轉成相對報酬，設為 `0.0`。
- 分歧月份是 challenger review 的樣本入口，不應只挑對某模型有利的案例。

## Regime dashboard

Dashboard 建議共用時間軸並由上而下排列：

1. Growth component／block contribution，零線表示 Growth 與 Contraction 的分界。
2. Policy 指標與門檻，背景標示 Easing／Tightening。
3. 股債相對報酬或策略 active return，背景標示四象限 Regime。

這類圖的欄位與經濟定義通常具有專案差異，因此先用 `plot_two_series`、`shade_periods` 與 Matplotlib axes 組合，不要為單一模型建立硬編碼 helper。各面板要共用 effective month，並在來源註記中說明發布落後與配置生效時點。

## 模型狀態熱圖

使用 `plot_regime_heatmap`，輸入為「月份 × 模型」的狀態 DataFrame，色彩 mapping 由呼叫端傳入。

- 顏色必須在整份簡報維持一致，例如 `G|E` 不可跨頁改色。
- 每格同時保留文字狀態，避免只靠顏色辨識。
- 最新月份使用框線突出；未取得資料顯示 `N/A`。
- `flags` 加上文字尾碼，適合 carried、stale 或資料品質異常。
- `marker_flags` 加上第二種符號，適合模型間訊號分歧。
- `outline_flags` 加上單格框線，適合圈選要回顧的失效月份。旗標語意由圖說交代，不能只放在程式 log。

## 加權貢獻熱圖

使用 `plot_weighted_contribution_heatmap`。輸入底層 component scores 與實際權重；函式繪製 `score × weight`，並回傳計算後的貢獻表供稽核。

- 權重必須對所有 component 一一對應，不允許遺漏或額外欄位。
- 發散色階固定以零為中心；負值表示把總分推向負向狀態，正值表示推向正向狀態。
- 總分列是各 component 貢獻加總，不要再次乘權重。
- 顏色深淺只能比較對總分的實際影響，不能把未加權 Z-score 和加權貢獻混在同一色階。
- carried 旗標應落在 component cell；總分列只要含任一 carried component 就同步標記。

## 最小使用方式

```python
from financial_charting import (
    plot_backtest_diagnostics,
    plot_cumulative_relative_returns,
    plot_regime_heatmap,
    plot_signal_confirmation,
    plot_underperformance_diagnostics,
    plot_weighted_contribution_heatmap,
)

fig, axes = plot_cumulative_relative_returns(
    monthly_returns,
    benchmark="SAA",
    title="資產配置策略累計報酬率",
    source="Internal backtest",
)

fig, ax = plot_regime_heatmap(
    monthly_regimes,
    state_colors={
        "G|E": "#D9EAD3",
        "G|T": "#FFF2CC",
        "C|E": "#D9EAF7",
        "C|T": "#F4CCCC",
        "N/A": "#E7E6E6",
    },
    flags=carried_flags,
    marker_flags=disagreement_flags,
    outline_flags=review_flags,
)

fig, ax, contributions = plot_weighted_contribution_heatmap(
    component_scores,
    component_weights,
    flags=carried_flags,
)

fig, axes = plot_backtest_diagnostics(
    monthly_returns,
    target_weights,
    benchmark="Benchmark",
)

fig, axes, diagnostics = plot_underperformance_diagnostics(
    strategy_returns,
    benchmark_returns,
    rolling_window=12,  # monthly observations
)

fig, axes = plot_signal_confirmation(
    baseline_score,
    challenger_score,
    relative_wealth,
    disagreement_flags=signal_disagreement,
    outcome_reference=1.0,
)
```

圖表函式不讀資料庫、不決定模型權重、不寫檔。由專案端完成資料擷取、PIT 對齊、模型計算與 `save_figure`。
