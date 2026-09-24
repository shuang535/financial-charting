---
name: financial-charting
description: 使用 Python、pandas 與 Matplotlib 製作或重構總經及金融研究圖表與回測表，包含時間序列、迴歸、投資組合比較、績效摘要、逐年與 Regime 分析、模型失效診斷及訊號熱圖。不用於 Plotly 儀表板、一般插畫或與金融研究無關的資料表。
---

# Financial Charting

製作清楚、可重現、可交付的研究圖表。先確認圖表要回答的問題，再選擇圖型；不要讓視覺效果掩蓋資料語意。

## 工作原則

- 將資料取得、轉換、統計估計與繪圖分開；繪圖函式不應偷偷讀資料庫或寫檔。
- 驗證時間索引、頻率、單位、缺值、樣本區間與發布時點。不得用補值或對齊方式暗中製造關係。
- 優先使用 Matplotlib object-oriented API；用 `fig`、`ax` 操作，不依賴 `plt.gcf()`、`plt.xlim()` 等全域狀態。
- 繪圖函式回傳 `Figure` 與 `Axes`，只有呼叫端明確要求時才存檔或顯示。
- 使用雙軸前先確認兩序列量綱不同且比較走勢有分析目的；同量綱資料優先共用單軸。
- 圖表要標示標題、單位、資料來源、樣本或轉換方式中會影響解讀的資訊。預測區間與歷史資料必須有可辨識的分界。
- 保持 import-safe：匯入模組不得讀寫檔案、連線資料源或永久修改全域繪圖設定。

## 選擇圖表

- 兩條時間序列、單雙軸與景氣陰影：讀 [references/time-series.md](references/time-series.md)。
- 月份季節性、基準期與未完整年度：讀 [references/seasonality.md](references/seasonality.md)。
- 散點、分組 OLS 趨勢與統計註記：讀 [references/regression-charts.md](references/regression-charts.md)。
- 累計／相對報酬、回測落後期診斷、訊號確認、Regime dashboard 與熱圖：讀 [references/regime-charts.md](references/regime-charts.md)。
- 績效摘要、逐年績效、Benchmark 勝率、Regime return 與 Drawdown episodes：讀 [references/performance-tables.md](references/performance-tables.md)。
- 色彩、標註、尺度與交付檢查：讀 [references/chart-design.md](references/chart-design.md)。

## 可重用資產

- `assets/financial_charting.py` 提供無資料庫依賴的 helper。需要在專案中落地繪圖程式時，可複製或依現有架構改寫；不要假設所有專案已安裝它。
- `assets/cathaysite.mplstyle` 提供選用的品牌色、中文字型 fallback、DPI 與基礎排版。只有交付物明確需要該品牌格式時才套用；若專案已有設計系統，沿用專案設定。
- 圖表不預設資料來源或機構署名；呼叫端應依實際資料與交付情境明確傳入。

## 完成前檢查

確認輸入沒有被意外改動、雙軸沒有暗示虛假尺度關係、衰退／預測陰影圖例正確、季節圖沒有把不完整年度當完整年度、迴歸線依排序後的 x 繪製。策略比較另確認共同樣本、Benchmark、報酬單位、年化頻率、Sharpe 的 `ddof`、relative-return 公式、rolling window、turnover 單雙邊與訊號生效月份；狀態熱圖不得隱藏 carried／missing／disagreement 狀態。失效診斷圖用來定位待研究期間，不把視覺上的轉折當作結構斷裂或因果證據。最後以目標媒介的實際尺寸檢查文字可讀性。
