# Antigravity Semiconductor Quant Dashboard ⚡

一個基於 Streamlit 的半導體量化回測與策略對照分析平台。支援輸入台股（.TW）或美股半導體標的，整合技術指標與基本面數據，協助投資者快速評估量化策略表現。

---

## 🌟 核心特色
1. **多因子評估系統 (15種因子)**：
   - **10種技術面指標**：MA Cross, RSI, Stochastic KD, MACD, Bollinger Bands, ATR Trend, ADX, CCI, OBV, Williams %R。
   - **5種基本面數據**：本益比 (P/E), 股價淨值比 (P/B), 股東權益報酬率 (ROE), 毛利率 (Gross Margin), 存貨週轉率 (Inventory Turnover)。
2. **模擬實驗室歷史紀錄**：使用 Streamlit `session_state` 儲存每次回測參數與權益曲線，支援隨時載入與刪除。
3. **複合因子加權模式**：可為 15 種因子分別指派 0-100% 的權重，生成複合訊號，並設定交易門檻。
4. **多策略同圖比對 (Equity Curve Comparison)**：可同時在同一張 Plotly 互動式圖表上比對多個已儲存策略的資產增值曲線與基準(Buy & Hold)表現。
5. **高級互動視覺化**：
   - K線與量能圖 (技術均線標註)
   - 策略資產與資金水下回撤圖 (Drawdown Underwater Area Plot)
   - 月度損益分佈長條圖

---

## ⚙️ 快速安裝與啟動

1. **安裝依賴套件**：
   ```bash
   pip install -r requirements.txt
   ```

2. **啟動應用程式**：
   ```bash
   streamlit run app.py
   ```

---

## 📁 專案結構
- `app.py`: Streamlit 網頁主程式與量化核心引擎。
- `requirements.txt`: 依賴的 Python 套件清單。
- `README.md`: 專案說明文件。
