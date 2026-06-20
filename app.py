import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# ==========================================
# 1. 頁面配置與主題樣式
# ==========================================
st.set_page_config(
    page_title="Antigravity Quant Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1.1 載入模擬參數處理 (避免在元件實例化後修改 session_state)
# ==========================================
if 'loaded_params' in st.session_state:
    params = st.session_state.loaded_params
    
    st.session_state['ticker'] = params['ticker']
    st.session_state['start_date'] = params['start_date']
    st.session_state['end_date'] = params['end_date']
    st.session_state['mode'] = params['mode']
    st.session_state['cost'] = params['cost']
    
    if params['mode'] == "單一因子":
        st.session_state['single_factor'] = params['single_factor']
    else:
        st.session_state['threshold'] = params['threshold']
        for factor_name, weight_val in params['weights'].items():
            st.session_state[f"w_{factor_name}"] = weight_val
            
    # 載入詳細因子參數
    if 'ind_params' in params:
        for pk, pv in params['ind_params'].items():
            st.session_state[pk] = pv
            
    # 清除暫存資料
    del st.session_state.loaded_params

# 套用精緻的深色 Glassmorphism 風格 CSS
st.markdown("""
<style>
    /* 全域字體與背景微調 */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* 自訂 KPI 卡片樣式 */
    .kpi-container {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    .kpi-card {
        flex: 1;
        background: rgba(17, 25, 40, 0.65);
        backdrop-filter: blur(16px) saturate(180%);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 20px 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        border: 1px solid rgba(0, 229, 255, 0.25);
        box-shadow: 0 12px 40px 0 rgba(0, 229, 255, 0.1);
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #8a99ad;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.2;
    }
    .kpi-delta {
        font-size: 0.95rem;
        margin-top: 6px;
        font-weight: 600;
    }
    .delta-positive {
        color: #00e676;
    }
    .delta-negative {
        color: #ff1744;
    }
    .delta-neutral {
        color: #b0bec5;
    }
    
    /* 側邊欄與 st.expander 細節微調 */
    .css-1639ggc {
        background-color: #0e1117;
    }
    
    /* 模擬紀錄清單項目 */
    .history-item {
        background: rgba(255, 255, 255, 0.03);
        border-left: 4px solid #00e5ff;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 0 6px 6px 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 技術指標核心算法 (向量化 Pandas 運算)
# ==========================================
def calc_ma_cross(df, fast=20, slow=60):
    sma_fast = df['Close'].rolling(window=fast).mean()
    sma_slow = df['Close'].rolling(window=slow).mean()
    signal = np.where(sma_fast > sma_slow, 1.0, -1.0)
    signal[sma_slow.isna()] = 0.0
    return pd.Series(signal, index=df.index), sma_fast, sma_slow

def calc_rsi(df, period=14, oversold=30, overbought=70):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / np.where(loss == 0, 1e-10, loss)
    rsi = 100 - (100 / (1 + rs))
    signal = np.zeros(len(df))
    signal[rsi < oversold] = 1.0
    signal[rsi > overbought] = -1.0
    signal[rsi.isna()] = 0.0
    return pd.Series(signal, index=df.index), rsi

def calc_kd(df, k_period=14, d_period=3, oversold=20, overbought=80):
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    rsv = 100 * (df['Close'] - low_min) / np.where(high_max - low_min == 0, 1e-10, high_max - low_min)
    k_val = rsv.rolling(window=3).mean()
    d_val = k_val.rolling(window=d_period).mean()
    signal = np.zeros(len(df))
    prev_k = k_val.shift(1)
    prev_d = d_val.shift(1)
    buy_cond = (k_val > d_val) & (prev_k <= prev_d) & (k_val < oversold)
    sell_cond = (k_val < d_val) & (prev_k >= prev_d) & (k_val > overbought)
    signal[buy_cond] = 1.0
    signal[sell_cond] = -1.0
    signal[k_val.isna() | d_val.isna()] = 0.0
    return pd.Series(signal, index=df.index), k_val, d_val

def calc_macd(df, fast=12, slow=26, signal_period=9):
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    hist = macd_line - signal_line
    signal = np.where(hist > 0, 1.0, -1.0)
    signal[hist.isna()] = 0.0
    return pd.Series(signal, index=df.index), macd_line, signal_line, hist

def calc_bbands(df, period=20, std_dev=2):
    middle = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    signal = np.zeros(len(df))
    signal[df['Close'] < lower] = 1.0
    signal[df['Close'] > upper] = -1.0
    signal[middle.isna()] = 0.0
    return pd.Series(signal, index=df.index), upper, middle, lower

def calc_atr(df, period=14):
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    ema20 = df['Close'].ewm(span=20, adjust=False).mean()
    atr_sma = atr.rolling(window=20).mean()
    signal = np.zeros(len(df))
    buy_cond = (df['Close'] > ema20) & (atr > atr_sma)
    sell_cond = (df['Close'] < ema20) & (atr > atr_sma)
    signal[buy_cond] = 1.0
    signal[sell_cond] = -1.0
    signal[atr.isna() | atr_sma.isna()] = 0.0
    return pd.Series(signal, index=df.index), atr

def calc_adx(df, period=14, threshold=25):
    up_move = df['High'] - df['High'].shift(1)
    down_move = df['Low'].shift(1) - df['Low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    smoothed_tr = tr.rolling(window=period).mean()
    smoothed_plus_dm = pd.Series(plus_dm, index=df.index).rolling(window=period).mean()
    smoothed_minus_dm = pd.Series(minus_dm, index=df.index).rolling(window=period).mean()
    plus_di = 100 * (smoothed_plus_dm / np.where(smoothed_tr == 0, 1e-10, smoothed_tr))
    minus_di = 100 * (smoothed_minus_dm / np.where(smoothed_tr == 0, 1e-10, smoothed_tr))
    dx = 100 * (plus_di - minus_di).abs() / np.where((plus_di + minus_di) == 0, 1e-10, plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    signal = np.zeros(len(df))
    buy_cond = (adx > threshold) & (plus_di > minus_di)
    sell_cond = (adx > threshold) & (minus_di > plus_di)
    signal[buy_cond] = 1.0
    signal[sell_cond] = -1.0
    signal[adx.isna()] = 0.0
    return pd.Series(signal, index=df.index), adx

def calc_cci(df, period=20, oversold=-100, overbought=100):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    def mean_deviation(x):
        return np.mean(np.abs(x - np.mean(x)))
    mad = tp.rolling(window=period).apply(mean_deviation, raw=True)
    cci = (tp - sma_tp) / (0.015 * np.where(mad == 0, 1e-10, mad))
    signal = np.zeros(len(df))
    signal[cci < oversold] = 1.0
    signal[cci > overbought] = -1.0
    signal[cci.isna()] = 0.0
    return pd.Series(signal, index=df.index), cci

def calc_obv(df, ema_period=20):
    vol = df['Volume']
    close_diff = df['Close'].diff()
    sign = np.zeros(len(df))
    sign[close_diff > 0] = 1.0
    sign[close_diff < 0] = -1.0
    obv_change = sign * vol
    obv = obv_change.cumsum()
    obv_ema = obv.ewm(span=ema_period, adjust=False).mean()
    signal = np.where(obv > obv_ema, 1.0, -1.0)
    signal[obv.isna() | obv_ema.isna()] = 0.0
    return pd.Series(signal, index=df.index), obv, obv_ema

def calc_williams_r(df, period=14, oversold=-80, overbought=-20):
    high_max = df['High'].rolling(window=period).max()
    low_min = df['Low'].rolling(window=period).min()
    williams = -100 * (high_max - df['Close']) / np.where(high_max - low_min == 0, 1e-10, high_max - low_min)
    signal = np.zeros(len(df))
    signal[williams < oversold] = 1.0
    signal[williams > overbought] = -1.0
    signal[williams.isna()] = 0.0
    return pd.Series(signal, index=df.index), williams

# ==========================================
# 3. 基本面資料擷取與日訊號轉換
# ==========================================
def extract_fundamental_info(info):
    pe = info.get('trailingPE') or info.get('forwardPE')
    pb = info.get('priceToBook')
    roe = info.get('returnOnEquity')
    margin = info.get('grossMargins')
    
    # 針對存貨週轉率，某些美股或台股欄位名稱不同，進行嘗試匹配
    inventory_turnover = info.get('inventoryTurnover')
    
    return {
        'P/E Ratio': pe,
        'P/B Ratio': pb,
        'ROE': roe,
        'Gross Margin': margin,
        'Inventory Turnover': inventory_turnover
    }

def compute_fundamental_signals(df, fundamentals):
    pe = fundamentals.get('P/E Ratio')
    pb = fundamentals.get('P/B Ratio')
    roe = fundamentals.get('ROE')
    margin = fundamentals.get('Gross Margin')
    turnover = fundamentals.get('Inventory Turnover')
    
    n = len(df)
    
    pe_sig = np.zeros(n)
    if pe is not None:
        pe_sig = np.where(pe < 20, 1.0, np.where(pe > 40, -1.0, 0.0))
        
    pb_sig = np.zeros(n)
    if pb is not None:
        pb_sig = np.where(pb < 3, 1.0, np.where(pb > 8, -1.0, 0.0))
        
    roe_sig = np.zeros(n)
    if roe is not None:
        roe_sig = np.where(roe > 0.15, 1.0, np.where(roe < 0.05, -1.0, 0.0))
        
    margin_sig = np.zeros(n)
    if margin is not None:
        margin_sig = np.where(margin > 0.40, 1.0, np.where(margin < 0.20, -1.0, 0.0))
        
    turnover_sig = np.zeros(n)
    if turnover is not None:
        turnover_sig = np.where(turnover > 4.0, 1.0, np.where(turnover < 1.5, -1.0, 0.0))
        
    return {
        'P/E Ratio': pd.Series(pe_sig, index=df.index),
        'P/B Ratio': pd.Series(pb_sig, index=df.index),
        'ROE': pd.Series(roe_sig, index=df.index),
        'Gross Margin': pd.Series(margin_sig, index=df.index),
        'Inventory Turnover': pd.Series(turnover_sig, index=df.index)
    }

# ==========================================
# 4. 回測引擎與指標計算
# ==========================================
def calc_all_signals(df, fundamentals, p=None):
    if p is None:
        p = {}
        
    # 提取參數，若無設定則使用預設值
    ma_fast = p.get('ma_fast', 20)
    ma_slow = p.get('ma_slow', 60)
    rsi_period = p.get('rsi_period', 14)
    rsi_oversold = p.get('rsi_oversold', 30)
    rsi_overbought = p.get('rsi_overbought', 70)
    kd_period = p.get('kd_period', 14)
    kd_d = p.get('kd_d', 3)
    kd_oversold = p.get('kd_oversold', 20)
    kd_overbought = p.get('kd_overbought', 80)
    macd_fast = p.get('macd_fast', 12)
    macd_slow = p.get('macd_slow', 26)
    macd_signal = p.get('macd_signal', 9)
    bb_period = p.get('bb_period', 20)
    bb_std = p.get('bb_std', 2.0)
    atr_period = p.get('atr_period', 14)
    adx_period = p.get('adx_period', 14)
    adx_threshold = p.get('adx_threshold', 25)
    cci_period = p.get('cci_period', 20)
    cci_oversold = p.get('cci_oversold', -100)
    cci_overbought = p.get('cci_overbought', 100)
    obv_ema_period = p.get('obv_ema_period', 20)
    wr_period = p.get('wr_period', 14)
    wr_oversold = p.get('wr_oversold', -80)
    wr_overbought = p.get('wr_overbought', -20)

    signals = {}
    
    # 1. MA Cross
    ma_sig, _, _ = calc_ma_cross(df, ma_fast, ma_slow)
    signals['MA Cross'] = ma_sig
    
    # 2. RSI
    rsi_sig, _ = calc_rsi(df, rsi_period, rsi_oversold, rsi_overbought)
    signals['RSI'] = rsi_sig
    
    # 3. KD
    kd_sig, _, _ = calc_kd(df, kd_period, kd_d, kd_oversold, kd_overbought)
    signals['Stochastic KD'] = kd_sig
    
    # 4. MACD
    macd_sig, _, _, _ = calc_macd(df, macd_fast, macd_slow, macd_signal)
    signals['MACD'] = macd_sig
    
    # 5. BBands
    bb_sig, _, _, _ = calc_bbands(df, bb_period, bb_std)
    signals['Bollinger Bands'] = bb_sig
    
    # 6. ATR
    atr_sig, _ = calc_atr(df, atr_period)
    signals['ATR Trend'] = atr_sig
    
    # 7. ADX
    adx_sig, _ = calc_adx(df, adx_period, adx_threshold)
    signals['ADX'] = adx_sig
    
    # 8. CCI
    cci_sig, _ = calc_cci(df, cci_period, cci_oversold, cci_overbought)
    signals['CCI'] = cci_sig
    
    # 9. OBV
    obv_sig, _, _ = calc_obv(df, obv_ema_period)
    signals['OBV'] = obv_sig
    
    # 10. Williams %R
    wr_sig, _ = calc_williams_r(df, wr_period, wr_oversold, wr_overbought)
    signals['Williams %R'] = wr_sig
    
    # Fundamental signals (5 items)
    fund_sigs = compute_fundamental_signals(df, fundamentals)
    signals.update(fund_sigs)
    
    return signals

def run_backtest(df, positions, transaction_fee=0.0015, initial_capital=100000):
    close = df['Close']
    stock_returns = close.pct_change().fillna(0)
    
    # 訊號今日產生，明日開盤/收盤執行。這裡採用 shift(1) 代表訊號遞延一天生效
    delayed_positions = positions.shift(1).fillna(0)
    
    # 計算交易點
    trades = delayed_positions.diff().fillna(0).abs()
    
    # 計算策略損益
    strat_returns = delayed_positions * stock_returns
    
    # 扣除手續費/滑價 (每次交易買進或賣出皆扣除特定比率)
    fee_deductions = trades * transaction_fee
    net_returns = strat_returns - fee_deductions
    
    # 計算權益曲線 (累乘)
    equity_curve = initial_capital * (1 + net_returns).cumprod()
    benchmark_curve = initial_capital * (1 + stock_returns).cumprod()
    
    # 績效指標計算
    total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital
    benchmark_return = (benchmark_curve.iloc[-1] - initial_capital) / initial_capital
    
    # 年化夏普值 (無風險利率設為 1.5%)
    rf_daily = 0.015 / 252
    excess_returns = net_returns - rf_daily
    if excess_returns.std() != 0:
        sharpe = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    else:
        sharpe = 0.0
        
    # 最大回撤
    cum_max = equity_curve.cummax()
    drawdown = (equity_curve - cum_max) / cum_max
    max_dd = drawdown.min()
    
    # 勝率與交易次數 (基於交易部位的進出)
    win_rate, num_trades = calc_trade_win_rate(close, delayed_positions)
    
    return {
        'equity_curve': equity_curve,
        'benchmark_curve': benchmark_curve,
        'net_returns': net_returns,
        'positions': delayed_positions,
        'metrics': {
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'win_rate': win_rate,
            'num_trades': num_trades
        }
    }

def calc_trade_win_rate(close, positions):
    # 尋找部位變化
    pos_diff = positions.diff().fillna(0)
    buys = pos_diff[pos_diff == 1].index
    sells = pos_diff[pos_diff == -1].index
    
    trade_profits = []
    
    for buy_date in buys:
        # 尋找買入後的第一個賣出時間點
        future_sells = sells[sells > buy_date]
        if not future_sells.empty:
            sell_date = future_sells[0]
            buy_price = close.loc[buy_date]
            sell_price = close.loc[sell_date]
            trade_profits.append((sell_price - buy_price) / buy_price)
        else:
            # 若部位仍未平倉，以回測終止日收盤價計
            buy_price = close.loc[buy_date]
            sell_price = close.iloc[-1]
            trade_profits.append((sell_price - buy_price) / buy_price)
            
    if len(trade_profits) == 0:
        return 0.0, 0
        
    wins = sum(1 for p in trade_profits if p > 0)
    win_rate = wins / len(trade_profits)
    return win_rate, len(trade_profits)

# ==========================================
# 4.5. 視覺化繪圖與 KPI 渲染函數
# ==========================================
def render_kpi_card(title, value, delta_str, is_positive):
    delta_class = "delta-positive" if is_positive else "delta-negative"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-delta {delta_class}">{delta_str}</div>
    </div>
    """, unsafe_allow_html=True)

def plot_candlestick(df, sma20=None, sma60=None):
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        subplot_titles=('價格走勢與均線', '交易量'), 
        row_width=[0.2, 0.8]
    )
    
    # 蠟燭圖
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='K線'
    ), row=1, col=1)
    
    # 移動平均線
    if sma20 is not None:
        fig.add_trace(go.Scatter(x=df.index, y=sma20, name='SMA 20', line=dict(color='#00e676', width=1.5)), row=1, col=1)
    if sma60 is not None:
        fig.add_trace(go.Scatter(x=df.index, y=sma60, name='SMA 60', line=dict(color='#2979ff', width=1.5)), row=1, col=1)
        
    # 量能長條圖 — 使用鮮豔的上漲色彩，傳統紅綠加上 0.85 opacity
    vol_colors = [
        'rgba(0, 230, 118, 0.85)' if cl >= op else 'rgba(255, 23, 68, 0.85)'
        for cl, op in zip(df['Close'], df['Open'])
    ]
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker_color=vol_colors,
        marker_line_width=0
    ), row=2, col=1)
    
    fig.update_layout(
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=30, r=30, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    # 調亮量能區小標題
    fig.update_yaxes(title_text="量 (Volume)", row=2, col=1, title_font=dict(color='#90caf9'))
    return fig

def plot_equity_and_drawdown(equity_curve, benchmark_curve, positions=None):
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05, 
        subplot_titles=('投資組合累計權益 (Equity Curve)', '資金水下回撤 (Drawdown %)'),
        row_width=[0.3, 0.7]
    )
    
    # 策略與基準
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, name='策略複利權益', line=dict(color='#00e5ff', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=benchmark_curve.index, y=benchmark_curve, name='買入持有基準', line=dict(color='#8a99ad', width=1.5, dash='dash')), row=1, col=1)
    
    # 買進 / 賣出標記點
    if positions is not None:
        pos_diff = positions.diff().fillna(0)
        buy_dates  = pos_diff[pos_diff == 1].index
        sell_dates = pos_diff[pos_diff == -1].index

        if len(buy_dates) > 0:
            fig.add_trace(go.Scatter(
                x=buy_dates,
                y=equity_curve.reindex(buy_dates),
                mode='markers',
                name='買進點 ▲',
                marker=dict(symbol='triangle-up', color='#00e676', size=12,
                            line=dict(color='#ffffff', width=1))
            ), row=1, col=1)

        if len(sell_dates) > 0:
            fig.add_trace(go.Scatter(
                x=sell_dates,
                y=equity_curve.reindex(sell_dates),
                mode='markers',
                name='賣出點 ▼',
                marker=dict(symbol='triangle-down', color='#ff1744', size=12,
                            line=dict(color='#ffffff', width=1))
            ), row=1, col=1)
    
    # 水下圖
    cum_max = equity_curve.cummax()
    drawdown = (equity_curve - cum_max) / cum_max * 100
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown,
        name='回撤 (%)',
        fill='tozeroy',
        fillcolor='rgba(255, 23, 68, 0.15)',
        line=dict(color='#ff1744', width=1.2)
    ), row=2, col=1)
    
    fig.update_layout(
        template='plotly_dark',
        height=520,
        margin=dict(l=30, r=30, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_monthly_returns(net_returns):
    try:
        monthly_ret = net_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1) * 100
    except ValueError:
        monthly_ret = net_returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
    
    colors = ['#ff1744' if val < 0 else '#00e676' for val in monthly_ret]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly_ret.index.strftime('%Y-%m'),
        y=monthly_ret,
        name='單月損益 (%)',
        marker_color=colors
    ))
    
    fig.update_layout(
        title='策略月份損益分佈圖 (%)',
        template='plotly_dark',
        height=300,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    return fig

# ==========================================================
# 5. 資料快取下載
# ==========================================
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            return None, None, "下載資料為空，請確認股票代碼或時間範圍。"
        
        # 整理 Multi-Index 欄位 (yfinance 某些版本會回傳多重索引)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
            
        df = df.sort_index()
        
        # 嘗試下載基本面資訊
        t = yf.Ticker(ticker)
        info = t.info
        
        return df, info, None
    except Exception as e:
        return None, None, f"下載發生錯誤: {str(e)}"

# ==========================================
# 6. 初始化 Session State 模擬實驗室
# ==========================================
if 'simulation_history' not in st.session_state:
    st.session_state.simulation_history = []

# 用於將參數載入到 UI 中
def load_params_into_ui(sim):
    # 將參數儲存到暫存區，並呼叫 rerun 在下一次執行的開頭載入，避免 widget 鎖定錯誤
    st.session_state.loaded_params = sim['params']
    st.sidebar.success(f"已載入: {sim['name']}")
    st.rerun()

# ==========================================
# 7. 側邊欄控制與實驗室歷史紀錄
# ==========================================
st.sidebar.title("🧪 Quant Simulation Lab")
st.sidebar.write("調整參數並管理已儲存的量化模擬實驗。")

# 設定載入 session_state 的預設值
_today = datetime.date.today()
_default_start = datetime.date(_today.year - 3, 1, 1)
def_ticker = st.session_state.get('ticker', '2330.TW')
def_start = st.session_state.get('start_date', _default_start)
def_end = st.session_state.get('end_date', _today)
def_mode = st.session_state.get('mode', '單一因子')
def_cost = st.session_state.get('cost', 0.15) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("📈 基本配置")

# ── 選單式個股挑選器 ───────────────────────────────
with st.sidebar.expander("🔍 挑選個股（市場/類股選單）", expanded=False):
    sb_mkt = st.selectbox("市場", ["台股", "美股"], key="sb_mkt")
    sb_sec_list = list(PEER_DB.get(sb_mkt, {}).keys())
    sb_sec = st.selectbox("產業類別", sb_sec_list, key="sb_sec")
    sb_stk_dict = PEER_DB.get(sb_mkt, {}).get(sb_sec, {})
    sb_stk_labels = [f"{t} {n}" for t, n in sb_stk_dict.items()]
    sb_selected = st.selectbox("選擇個股", sb_stk_labels, key="sb_stk")
    if st.button("✅ 套用此個股代碼", use_container_width=True):
        selected_code = sb_selected.split()[0]
        st.session_state['ticker'] = selected_code
        st.rerun()

ticker = st.sidebar.text_input("股票代碼（台股加 .TW）", value=def_ticker, key="ticker")
start_date = st.sidebar.date_input("開始日期", value=def_start, key="start_date")
end_date = st.sidebar.date_input("結束日期", value=def_end, key="end_date")

cost_percent = st.sidebar.slider("單邊交易手續費+滑價 (%)", min_value=0.0, max_value=1.0, value=float(def_cost * 100), step=0.05, key="cost")
transaction_fee = cost_percent / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 因子模式與權重")
mode = st.sidebar.selectbox("回測模式", ["單一因子", "複合因子"], key="mode")

factors_list = [
    "MA Cross", "RSI", "Stochastic KD", "MACD", "Bollinger Bands",
    "ATR Trend", "ADX", "CCI", "OBV", "Williams %R",
    "P/E Ratio", "P/B Ratio", "ROE", "Gross Margin", "Inventory Turnover"
]

FACTOR_DISPLAY = {
    "MA Cross":           "MA Cross 均線交叉",
    "RSI":                "RSI 相對強弱指數",
    "Stochastic KD":      "Stochastic KD 隨機震盪指標",
    "MACD":               "MACD 指數平滑異同移動平均",
    "Bollinger Bands":    "Bollinger Bands 布林通道",
    "ATR Trend":          "ATR Trend 真實波幅趨勢",
    "ADX":                "ADX 平均趨向指數",
    "CCI":                "CCI 商品通道指數",
    "OBV":                "OBV 能量潮指標",
    "Williams %R":        "Williams %R 威廉指標",
    "P/E Ratio":          "P/E Ratio 本益比",
    "P/B Ratio":          "P/B Ratio 股價淨值比",
    "ROE":                "ROE 股東權益報酬率",
    "Gross Margin":       "Gross Margin 毛利率",
    "Inventory Turnover": "Inventory Turnover 存貨週轉率",
}

# 同業標的資料庫 (市場 → 產業 → {代號: 公司名})
PEER_DB = {
    "台股": {
        "半導體製造": {
            "2330.TW": "台積電",
            "2303.TW": "聯電",
            "5347.TWO": "世界先進",
            "2344.TW": "華邦電",
            "2408.TW": "南亞科",
        },
        "IC 設計": {
            "2454.TW": "聯發科",
            "2379.TW": "瑞昱半導體",
            "3034.TW": "聯詠科技",
            "2385.TW": "群光電子",
            "3443.TW": "創意電子",
        },
        "封裝測試": {
            "3711.TW": "日月光投控",
            "2325.TW": "矽品精密",
            "6531.TW": "愛普科技",
        },
        "半導體設備/材料": {
            "3105.TW": "穩懋半導體",
            "3529.TW": "力旺電子",
            "6669.TW": "緯穎科技",
        },
        "電子零組件": {
            "2317.TW": "鴻海精密",
            "2382.TW": "廣達電腦",
            "2357.TW": "華碩電腦",
            "2354.TW": "鴻準精密",
        },
        "金融": {
            "2882.TW": "國泰金控",
            "2881.TW": "富邦金控",
            "2886.TW": "兆豐金控",
            "2884.TW": "玉山金控",
        },
        "傳產/化工": {
            "1301.TW": "台灣塑膠",
            "1303.TW": "南亞塑膠",
            "2002.TW": "中國鋼鐵",
            "1216.TW": "統一企業",
        },
    },
    "美股": {
        "半導體": {
            "TSM":  "台積電 ADR",
            "NVDA": "輝達 Nvidia",
            "AMD":  "超微 AMD",
            "INTC": "英特爾",
            "ASML": "艾司摩爾",
            "QCOM": "高通",
            "AVGO": "博通",
            "MU":   "美光科技",
            "AMAT": "應用材料",
            "LRCX": "科林研發",
        },
        "科技": {
            "AAPL":  "蘋果",
            "MSFT":  "微軟",
            "GOOGL": "Alphabet",
            "META":  "Meta",
            "AMZN":  "亞馬遜",
        },
        "金融": {
            "JPM": "摩根大通",
            "BAC": "美國銀行",
            "GS":  "高盛",
            "MS":  "摩根史坦利",
        },
        "醫療/生技": {
            "JNJ":  "嬌生",
            "PFE":  "輝瑞",
            "MRNA": "莫德納",
            "ABBV": "艾伯維",
        },
        "能源": {
            "XOM": "埃克森美孚",
            "CVX": "雪佛龍",
            "COP": "康菲石油",
        },
    },
}

single_factor = ""
weights = {}
threshold = 0.1

if mode == "單一因子":
    def_single = st.session_state.get('single_factor', 'RSI')
    if def_single not in factors_list:
        def_single = 'RSI'
    single_factor = st.sidebar.selectbox(
        "選擇交易因子",
        factors_list,
        index=factors_list.index(def_single),
        key="single_factor",
        format_func=lambda x: FACTOR_DISPLAY.get(x, x)
    )
else:
    def_thresh = st.session_state.get('threshold', 0.1)
    threshold = st.sidebar.slider("複合訊號交易門檻", min_value=0.0, max_value=0.5, value=def_thresh, step=0.05, key="threshold")
    
    st.sidebar.markdown("**調整因子權重 (0 - 100%)**")
    for f in factors_list:
        def_w = st.session_state.get(f"w_{f}", 20 if f in ["MA Cross", "RSI", "MACD"] else 0)
        weights[f] = st.sidebar.slider(FACTOR_DISPLAY.get(f, f), min_value=0, max_value=100, value=def_w, step=5, key=f"w_{f}")

# ==========================================
# 7.2. 因子參數細部調整 (側邊欄摺疊區)
# ==========================================
with st.sidebar.expander("🔧 因子參數細部調整", expanded=False):
    active_tech_factors = []
    if mode == "單一因子":
        active_tech_factors = [single_factor]
    else:
        active_tech_factors = [f for f in factors_list if weights[f] > 0]
        
    # MA Cross
    if "MA Cross" in active_tech_factors:
        st.markdown("**MA Cross 參數**")
        p_ma_fast = st.slider("Fast SMA 週期", 5, 50, value=st.session_state.get('p_ma_fast', 20), key="p_ma_fast")
        p_ma_slow = st.slider("Slow SMA 週期", 30, 200, value=st.session_state.get('p_ma_slow', 60), key="p_ma_slow")
    else:
        p_ma_fast, p_ma_slow = 20, 60
        
    # RSI
    if "RSI" in active_tech_factors:
        st.markdown("**RSI 參數**")
        p_rsi_period = st.slider("RSI 計算週期", 5, 30, value=st.session_state.get('p_rsi_period', 14), key="p_rsi_period")
        p_rsi_oversold = st.slider("RSI 買入超賣界線", 10, 45, value=st.session_state.get('p_rsi_oversold', 30), key="p_rsi_oversold")
        p_rsi_overbought = st.slider("RSI 賣出超買界線", 55, 90, value=st.session_state.get('p_rsi_overbought', 70), key="p_rsi_overbought")
    else:
        p_rsi_period, p_rsi_oversold, p_rsi_overbought = 14, 30, 70
        
    # Stochastic KD
    if "Stochastic KD" in active_tech_factors:
        st.markdown("**Stochastic KD 參數**")
        p_kd_period = st.slider("KD 計算週期", 5, 30, value=st.session_state.get('p_kd_period', 14), key="p_kd_period")
        p_kd_d = st.slider("D 線平滑週期", 2, 10, value=st.session_state.get('p_kd_d', 3), key="p_kd_d")
        p_kd_oversold = st.slider("KD 買入界線 (%K <)", 5, 40, value=st.session_state.get('p_kd_oversold', 20), key="p_kd_oversold")
        p_kd_overbought = st.slider("KD 賣出界線 (%K >)", 60, 95, value=st.session_state.get('p_kd_overbought', 80), key="p_kd_overbought")
    else:
        p_kd_period, p_kd_d, p_kd_oversold, p_kd_overbought = 14, 3, 20, 80
        
    # MACD
    if "MACD" in active_tech_factors:
        st.markdown("**MACD 參數**")
        p_macd_fast = st.slider("快線 EMA 週期", 5, 25, value=st.session_state.get('p_macd_fast', 12), key="p_macd_fast")
        p_macd_slow = st.slider("慢線 EMA 週期", 20, 60, value=st.session_state.get('p_macd_slow', 26), key="p_macd_slow")
        p_macd_signal = st.slider("訊號線 MACD EMA 週期", 5, 15, value=st.session_state.get('p_macd_signal', 9), key="p_macd_signal")
    else:
        p_macd_fast, p_macd_slow, p_macd_signal = 12, 26, 9
        
    # Bollinger Bands
    if "Bollinger Bands" in active_tech_factors:
        st.markdown("**Bollinger Bands 參數**")
        p_bb_period = st.slider("BBands 計算週期", 10, 50, value=st.session_state.get('p_bb_period', 20), key="p_bb_period")
        p_bb_std = st.slider("標準差倍數", 1.0, 3.0, value=float(st.session_state.get('p_bb_std', 2.0)), step=0.1, key="p_bb_std")
    else:
        p_bb_period, p_bb_std = 20, 2.0
        
    # ATR Trend
    if "ATR Trend" in active_tech_factors:
        st.markdown("**ATR Trend 參數**")
        p_atr_period = st.slider("ATR 計算週期", 5, 30, value=st.session_state.get('p_atr_period', 14), key="p_atr_period")
    else:
        p_atr_period = 14
        
    # ADX
    if "ADX" in active_tech_factors:
        st.markdown("**ADX 參數**")
        p_adx_period = st.slider("ADX 計算週期", 5, 30, value=st.session_state.get('p_adx_period', 14), key="p_adx_period")
        p_adx_threshold = st.slider("ADX 趨勢強度門檻", 15, 40, value=st.session_state.get('p_adx_threshold', 25), key="p_adx_threshold")
    else:
        p_adx_period, p_adx_threshold = 14, 25
        
    # CCI
    if "CCI" in active_tech_factors:
        st.markdown("**CCI 參數**")
        p_cci_period = st.slider("CCI 計算週期", 10, 50, value=st.session_state.get('p_cci_period', 20), key="p_cci_period")
        p_cci_oversold = st.slider("CCI 買入超賣界線", -200, -50, value=st.session_state.get('p_cci_oversold', -100), step=10, key="p_cci_oversold")
        p_cci_overbought = st.slider("CCI 賣出超買界線", 50, 200, value=st.session_state.get('p_cci_overbought', 100), step=10, key="p_cci_overbought")
    else:
        p_cci_period, p_cci_oversold, p_cci_overbought = 20, -100, 100
        
    # OBV
    if "OBV" in active_tech_factors:
        st.markdown("**OBV 參數**")
        p_obv_ema_period = st.slider("OBV 平滑 EMA 週期", 5, 50, value=st.session_state.get('p_obv_ema_period', 20), key="p_obv_ema_period")
    else:
        p_obv_ema_period = 20
        
    # Williams %R
    if "Williams %R" in active_tech_factors:
        st.markdown("**Williams %R 參數**")
        p_wr_period = st.slider("Williams %R 週期", 5, 30, value=st.session_state.get('p_wr_period', 14), key="p_wr_period")
        p_wr_oversold = st.slider("Williams %R 買入超賣界線", -95, -60, value=st.session_state.get('p_wr_oversold', -80), step=5, key="p_wr_oversold")
        p_wr_overbought = st.slider("Williams %R 賣出超買界線", -40, -5, value=st.session_state.get('p_wr_overbought', -20), step=5, key="p_wr_overbought")
    else:
        p_wr_period, p_wr_oversold, p_wr_overbought = 14, -80, -20

# 建立參數包傳入計算函數
indicator_params = {
    'ma_fast': p_ma_fast, 'ma_slow': p_ma_slow,
    'rsi_period': p_rsi_period, 'rsi_oversold': p_rsi_oversold, 'rsi_overbought': p_rsi_overbought,
    'kd_period': p_kd_period, 'kd_d': p_kd_d, 'kd_oversold': p_kd_oversold, 'kd_overbought': p_kd_overbought,
    'macd_fast': p_macd_fast, 'macd_slow': p_macd_slow, 'macd_signal': p_macd_signal,
    'bb_period': p_bb_period, 'bb_std': p_bb_std,
    'atr_period': p_atr_period,
    'adx_period': p_adx_period, 'adx_threshold': p_adx_threshold,
    'cci_period': p_cci_period, 'cci_oversold': p_cci_oversold, 'cci_overbought': p_cci_overbought,
    'obv_ema_period': p_obv_ema_period,
    'wr_period': p_wr_period, 'wr_oversold': p_wr_oversold, 'wr_overbought': p_wr_overbought
}

# ==========================================
# 7.3 自動最佳參數引擎 (Grid Search)
# ==========================================
def run_grid_search(df, factor_name):
    """For a given factor, sweep parameter grid and return best params by cumulative return."""
    grids = {
        "MA Cross": [{"ma_fast": f, "ma_slow": s} for f in [10, 15, 20, 25] for s in [40, 50, 60, 80] if s > f + 10],
        "RSI": [{"rsi_period": p, "rsi_oversold": o, "rsi_overbought": ob}
                for p in [7, 14, 21] for o in [20, 25, 30, 35] for ob in [65, 70, 75, 80]],
        "Stochastic KD": [{"kd_period": p, "kd_oversold": o, "kd_overbought": ob}
                          for p in [9, 14, 21] for o in [15, 20, 25] for ob in [75, 80, 85]],
        "MACD": [{"macd_fast": f, "macd_slow": s, "macd_signal": sig}
                 for f in [8, 12] for s in [21, 26] for sig in [7, 9]],
        "Bollinger Bands": [{"bb_period": p, "bb_std": sd}
                            for p in [15, 20, 25] for sd in [1.5, 2.0, 2.5]],
        "ATR Trend": [{"atr_period": p} for p in [10, 14, 18]],
        "ADX": [{"adx_period": p, "adx_threshold": t}
                for p in [10, 14] for t in [20, 25, 30]],
        "CCI": [{"cci_period": p, "cci_oversold": o, "cci_overbought": ob}
                for p in [14, 20] for o in [-120, -100, -80] for ob in [80, 100, 120]],
        "OBV": [{"obv_ema_period": p} for p in [10, 15, 20, 30]],
        "Williams %R": [{"wr_period": p, "wr_oversold": o, "wr_overbought": ob}
                        for p in [10, 14] for o in [-85, -80, -75] for ob in [-25, -20, -15]],
    }
    if factor_name not in grids:
        return {}   # Fundamental factors - no grid search

    best_params = {}
    best_return = -999.0
    fundamentals_dummy = {k: None for k in ['P/E Ratio','P/B Ratio','ROE','Gross Margin','Inventory Turnover']}

    for param_combo in grids[factor_name]:
        p = param_combo.copy()
        try:
            sigs = calc_all_signals(df, fundamentals_dummy, p=p)
            raw = sigs[factor_name]
            pos = []
            curr = 0
            for v in raw:
                if v == 1.0: curr = 1
                elif v == -1.0: curr = 0
                pos.append(curr)
            pos_series = pd.Series(pos, index=df.index)
            result = run_backtest(df, pos_series, transaction_fee=0.0015)
            ret = result['metrics']['total_return']
            if ret > best_return:
                best_return = ret
                best_params = p
        except Exception:
            continue

    return best_params

def build_trade_log(df, positions, indicator_series, factor_name):
    """Build a detailed trade log DataFrame with buy/sell date, price, indicator value, PnL."""
    pos_diff = positions.diff().fillna(0)
    buys = pos_diff[pos_diff == 1].index
    sells = pos_diff[pos_diff == -1].index
    close = df['Close']

    rows = []
    for i, buy_date in enumerate(buys):
        future_sells = sells[sells > buy_date]
        sell_date = future_sells[0] if not future_sells.empty else close.index[-1]
        buy_price = float(close.loc[buy_date])
        sell_price = float(close.loc[sell_date])
        pnl = (sell_price - buy_price) / buy_price * 100
        ind_buy = float(indicator_series.loc[buy_date]) if buy_date in indicator_series.index else float('nan')
        ind_sell = float(indicator_series.loc[sell_date]) if sell_date in indicator_series.index else float('nan')
        rows.append({
            "#": i + 1,
            "買入日期": buy_date.strftime("%Y-%m-%d"),
            "買入股價": f"{buy_price:.2f}",
            f"買入時{factor_name}値": f"{ind_buy:.2f}",
            "賣出日期": sell_date.strftime("%Y-%m-%d"),
            "賣出股價": f"{sell_price:.2f}",
            f"賣出時{factor_name}値": f"{ind_sell:.2f}",
            "單筆損益 (%)": f"{pnl:+.2f}%"
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600)
def fetch_peer_stock_data(ticker_list, start_date, end_date):
    """Download OHLCV for a basket of peer stocks."""
    results = {}
    for t in ticker_list:
        try:
            df_peer = yf.download(t, start=start_date, end=end_date, progress=False)
            if df_peer.empty:
                continue
            if isinstance(df_peer.columns, pd.MultiIndex):
                df_peer.columns = [col[0] for col in df_peer.columns]
            results[t] = df_peer.sort_index()
        except Exception:
            continue
    return results

# 對於單一因子模式，自動執行最佳參數網格搜尋
if mode == "單一因子" and single_factor not in ["P/E Ratio", "P/B Ratio", "ROE", "Gross Margin", "Inventory Turnover"]:
    # 編製缓存鍵値：個股+日期+因子組合沒變化就不重跡運算
    opt_key = f"opt_{ticker}_{start_date}_{end_date}_{single_factor}"
    if opt_key not in st.session_state:
        with st.spinner(f"正在對 {single_factor} 執行最佳參數搜尋，請稍候..."):
            # 第一次要下載資料來做 grid search
            tmp_df, _, _ = fetch_stock_data(ticker, start_date, end_date)
            if tmp_df is not None:
                best = run_grid_search(tmp_df, single_factor)
                st.session_state[opt_key] = best
            else:
                st.session_state[opt_key] = {}

    best_params = st.session_state.get(opt_key, {})
    # 將最佳參數回填到 session_state（下一次 rerun 會自動載入滑桿）
    for k, v in best_params.items():
        sess_key = f"p_{k}"
        if sess_key not in st.session_state:
            st.session_state[sess_key] = v

# ==========================================
# 7.4. 同業回測輔助函數
# ==========================================
def run_sim_on_df(df, sim_params, transaction_fee=0.0015):
    """Replay a saved simulation's strategy on a different stock DataFrame."""
    fund_dummy = {k: None for k in ['P/E Ratio','P/B Ratio','ROE','Gross Margin','Inventory Turnover']}
    p = sim_params.get('ind_params', {})
    mode_s = sim_params.get('mode', '單一因子')
    sigs = calc_all_signals(df, fund_dummy, p=p)

    if mode_s == '單一因子':
        factor = sim_params.get('single_factor', 'RSI')
        raw = sigs.get(factor, pd.Series(0.0, index=df.index))
        pos, curr = [], 0
        for v in raw:
            if v == 1.0: curr = 1
            elif v == -1.0: curr = 0
            pos.append(curr)
        pos_series = pd.Series(pos, index=df.index, dtype=float)
    else:
        weights_s = sim_params.get('weights', {})
        threshold_s = sim_params.get('threshold', 0.1)
        total_w = sum(weights_s.values()) or 1
        composite = sum(
            sigs.get(f, pd.Series(0.0, index=df.index)) * (w / 100.0)
            for f, w in weights_s.items()
        )
        pos, curr = [], 0
        for v in composite:
            if v > threshold_s: curr = 1
            elif v < -threshold_s: curr = 0
            pos.append(curr)
        pos_series = pd.Series(pos, index=df.index, dtype=float)

    return run_backtest(df, pos_series, transaction_fee=transaction_fee)

# 下載資料
with st.spinner("獲取 yfinance 歷史數據與基本面..."):
    df, info, err = fetch_stock_data(ticker, start_date, end_date)

# ==========================================
# 8. 主畫面渲染
# ==========================================
st.title("Antigravity Quant Lab ⚡")
st.write("精密的半導體量化回測與策略對照平台")

if err:
    st.error(err)
    st.stop()

# 提取基本面
fundamentals = extract_fundamental_info(info)

# 模擬計算與資料準備
if df is not None:
    signals_dict = calc_all_signals(df, fundamentals, p=indicator_params)
    
    if mode == "單一因子":
        # 單一因子直接映射為 position (1 或 0)
        # 原訊號為 -1, 0, 1. 做 Long-only 轉換: 1 -> Buy, -1 -> Sell, 0 -> 维持
        # 使用 hysteresis 邏輯來處理單一因子的買賣狀態
        raw_sig = signals_dict[single_factor]
        positions = []
        curr_pos = 0
        for val in raw_sig:
            if val == 1.0:
                curr_pos = 1
            elif val == -1.0:
                curr_pos = 0
            positions.append(curr_pos)
        positions = pd.Series(positions, index=df.index)
        strategy_name = f"單一因子: {single_factor}"
    else:
        # 複合因子加權
        total_w = sum(weights.values())
        if total_w > 0:
            agg_signal = np.zeros(len(df))
            for f in factors_list:
                agg_signal += signals_dict[f] * (weights[f] / total_w)
            agg_signal = pd.Series(agg_signal, index=df.index)
        else:
            agg_signal = pd.Series(np.zeros(len(df)), index=df.index)
            
        positions = []
        curr_pos = 0
        for val in agg_signal:
            if val > threshold:
                curr_pos = 1
            elif val < -threshold:
                curr_pos = 0
            positions.append(curr_pos)
        positions = pd.Series(positions, index=df.index)
        strategy_name = "複合因子策略"

    # 執行回測
    res = run_backtest(df, positions, transaction_fee=transaction_fee)
    metrics = res['metrics']

    # 新增模擬至 Session State
    if st.sidebar.button("🚀 開始模擬並儲存記錄", use_container_width=True):
        sim_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sim_name = f"{ticker} | {strategy_name} | {metrics['total_return']*100:.1f}%"
        
        sim_record = {
            'id': sim_id,
            'name': sim_name,
            'ticker': ticker,
            'strategy_name': strategy_name,
            'params': {
                'ticker': ticker,
                'start_date': start_date,
                'end_date': end_date,
                'mode': mode,
                'cost': cost_percent,
                'single_factor': single_factor,
                'threshold': threshold,
                'weights': weights.copy(),
                'ind_params': {
                    'p_ma_fast': p_ma_fast, 'p_ma_slow': p_ma_slow,
                    'p_rsi_period': p_rsi_period, 'p_rsi_oversold': p_rsi_oversold, 'p_rsi_overbought': p_rsi_overbought,
                    'p_kd_period': p_kd_period, 'p_kd_d': p_kd_d, 'p_kd_oversold': p_kd_oversold, 'p_kd_overbought': p_kd_overbought,
                    'p_macd_fast': p_macd_fast, 'p_macd_slow': p_macd_slow, 'p_macd_signal': p_macd_signal,
                    'p_bb_period': p_bb_period, 'p_bb_std': p_bb_std,
                    'p_atr_period': p_atr_period,
                    'p_adx_period': p_adx_period, 'p_adx_threshold': p_adx_threshold,
                    'p_cci_period': p_cci_period, 'p_cci_oversold': p_cci_oversold, 'p_cci_overbought': p_cci_overbought,
                    'p_obv_ema_period': p_obv_ema_period,
                    'p_wr_period': p_wr_period, 'p_wr_oversold': p_wr_oversold, 'p_wr_overbought': p_wr_overbought
                }
            },
            'metrics': metrics,
            'equity_curve': res['equity_curve'].tolist(),
            'benchmark_curve': res['benchmark_curve'].tolist(),
            'dates': [d.strftime("%Y-%m-%d") for d in res['equity_curve'].index]
        }
        st.session_state.simulation_history.append(sim_record)
        st.sidebar.success(f"已儲存模擬: {sim_name}")
        st.rerun()

# 渲染側邊欄實驗室歷史紀錄
st.sidebar.markdown("---")
st.sidebar.subheader("🧪 模擬歷史記錄")
if len(st.session_state.simulation_history) == 0:
    st.sidebar.info("尚無模擬紀錄，請點擊上方「開始模擬」按鈕。")
else:
    for idx, sim in enumerate(st.session_state.simulation_history):
        col1, col2 = st.sidebar.columns([4, 1])
        with col1:
            if st.button(sim['name'], key=f"sim_{sim['id']}", use_container_width=True):
                load_params_into_ui(sim)
        with col2:
            if st.button("❌", key=f"del_{sim['id']}"):
                st.session_state.simulation_history.pop(idx)
                st.sidebar.warning("已刪除該筆模擬紀錄")
                st.rerun()

# ==========================================
# 9. 主要 Tabs 分頁與繪圖
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 單一回測分析", "⚖️ 策略比對對照", "🔍 參數最佳化與同業比對"])

# ----------------- Tab 1: 單一回測 -----------------
with tab1:
    # 顯示目前公司基本面狀況 (美化版 Expandable Table)
    with st.expander(f"🔍 {ticker} 目前基本面指標與因子狀態", expanded=False):
        c1, c2, c3, c4, c5 = st.columns(5)
        
        def format_fund(val, is_pct=False):
            if val is None:
                return "N/A"
            if is_pct:
                return f"{val * 100:.2f}%"
            return f"{val:.2f}"
            
        pe_f = format_fund(fundamentals['P/E Ratio'])
        pb_f = format_fund(fundamentals['P/B Ratio'])
        roe_f = format_fund(fundamentals['ROE'], is_pct=True)
        gm_f = format_fund(fundamentals['Gross Margin'], is_pct=True)
        it_f = format_fund(fundamentals['Inventory Turnover'])
        
        # 顯示評分
        pe_score = "偏低 (看多 +1)" if fundamentals['P/E Ratio'] is not None and fundamentals['P/E Ratio'] < 20 else ("偏高 (看空 -1)" if fundamentals['P/E Ratio'] is not None and fundamentals['P/E Ratio'] > 40 else "中性 0")
        pb_score = "偏低 (看多 +1)" if fundamentals['P/B Ratio'] is not None and fundamentals['P/B Ratio'] < 3 else ("偏高 (看空 -1)" if fundamentals['P/B Ratio'] is not None and fundamentals['P/B Ratio'] > 8 else "中性 0")
        roe_score = "優秀 (看多 +1)" if fundamentals['ROE'] is not None and fundamentals['ROE'] > 0.15 else ("偏低 (看空 -1)" if fundamentals['ROE'] is not None and fundamentals['ROE'] < 0.05 else "中性 0")
        gm_score = "高毛利 (看多 +1)" if fundamentals['Gross Margin'] is not None and fundamentals['Gross Margin'] > 0.40 else ("低毛利 (看空 -1)" if fundamentals['Gross Margin'] is not None and fundamentals['Gross Margin'] < 0.20 else "中性 0")
        it_score = "流速快 (看多 +1)" if fundamentals['Inventory Turnover'] is not None and fundamentals['Inventory Turnover'] > 4.0 else ("流速慢 (看空 -1)" if fundamentals['Inventory Turnover'] is not None and fundamentals['Inventory Turnover'] < 1.5 else "中性 0")
        
        c1.metric("P/E Ratio (本益比)", pe_f, pe_score)
        c2.metric("P/B Ratio (股價淨值比)", pb_f, pb_score)
        c3.metric("ROE (股東權益報酬率)", roe_f, roe_score)
        c4.metric("Gross Margin (毛利率)", gm_f, gm_score)
        c5.metric("Inventory Turnover (週轉率)", it_f, it_score)

    # 策略決策規則描述
    st.markdown("### 📝 當前策略決策規則")
    if mode == "單一因子":
        if single_factor == "MA Cross":
            desc = f"當短期均線 SMA({p_ma_fast}) 向上突破長期均線 SMA({p_ma_slow}) 時買入做多；向下跌破時平倉出局。"
        elif single_factor == "RSI":
            desc = f"當 RSI({p_rsi_period}) 跌破超賣界線 {p_rsi_oversold} 時買入做多（超賣反彈）；漲破超買界線 {p_rsi_overbought} 時平倉出局（超買回檔）。"
        elif single_factor == "Stochastic KD":
            desc = f"當 KD指標 %K線 跌破超賣界線 {p_kd_oversold} 且向上金叉 %D線(平滑週期 {p_kd_d}) 時買入做多；在超買界線 > {p_kd_overbought} 且死叉時平倉出局。"
        elif single_factor == "MACD":
            desc = f"當 MACD柱狀圖 (快線:{p_macd_fast}, 慢線:{p_macd_slow}, 訊號線:{p_macd_signal}) 翻正 (> 0) 時買入做多；翻負 (< 0) 時平倉出局。"
        elif single_factor == "Bollinger Bands":
            desc = f"當股價跌破布林通道下軌 (週期:{p_bb_period}, 標準差:{p_bb_std}) 時買入做多；漲破布林通道上軌時平倉出局。"
        elif single_factor == "ATR Trend":
            desc = f"當股價大於 20日EMA 且 ATR({p_atr_period}) 呈擴張狀態時，判定為波動度上升之順勢行情買入做多；當股價低於 20日EMA 且 ATR 擴張時平倉出局。"
        elif single_factor == "ADX":
            desc = f"當 ADX({p_adx_period}) 大於趨勢強度門檻 {p_adx_threshold} 且 +DI > -DI 時買入做多；當 -DI > +DI 時平倉出局。"
        elif single_factor == "CCI":
            desc = f"當 CCI({p_cci_period}) 跌破超賣界線 {p_cci_oversold} 時買入做多；漲破超買界線 {p_cci_overbought} 時平倉出局。"
        elif single_factor == "OBV":
            desc = f"當 OBV 指標大於其 OBV_EMA({p_obv_ema_period}) 時（資金量能向上積聚）買入做多；低於 EMA 時平倉出局。"
        elif single_factor == "Williams %R":
            desc = f"當威廉指標 %R({p_wr_period}) 跌破超賣界線 {p_wr_oversold} 時買入做多；漲破超買界線 {p_wr_overbought} 時平倉出局。"
        else:
            # Fundamentals
            score_map = {
                "P/E Ratio": f"當 P/E 本益比 < 20 時看多，本益比 > 40 時看空，其餘中性。",
                "P/B Ratio": f"當 P/B 股價淨值比 < 3 時看多，股價淨值比 > 8 時看空，其餘中性。",
                "ROE": f"當 ROE 股東權益報酬率 > 15% 時看多，ROE < 5% 時看空，其餘中性。",
                "Gross Margin": f"當 Gross Margin 毛利率 > 40% 時看多，毛利率 < 20% 時看空，其餘中性。",
                "Inventory Turnover": f"當 Inventory Turnover 存貨週轉率 > 4.0 時看多，存貨週轉率 < 1.5 時看空，其餘中性。"
            }
            desc = score_map.get(single_factor, "基本面因子篩選看多/看空。")
            
        st.info(f"**【{strategy_name}】** {desc}")
    else:
        st.info(f"**【複合權重策略】** 綜合選定之多個技術與基本面因子，以加權分數形式（-1.0 ~ 1.0）計算。當複合訊號值大於交易門檻 {threshold} 時買入做多，小於 -{threshold} 時平倉出局。")

    # KPI 區塊
    st.markdown("### 🏆 當前回測結果 KPI")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    total_ret_val = metrics['total_return']
    benchmark_ret_val = metrics['benchmark_return']
    diff_ret = total_ret_val - benchmark_ret_val
    
    # 總損益(%) 與 總損益(金額) KPI
    initial_cap = 100000
    net_pnl_amt = (res['equity_curve'].iloc[-1] - initial_cap)
    
    with kpi_col1:
        render_kpi_card(
            title="策略累積損益 (%)",
            value=f"{total_ret_val * 100:.2f}%",
            delta_str=f"{diff_ret * 100:+.2f}% 較基準",
            is_positive=total_ret_val >= 0
        )
    with kpi_col2:
        render_kpi_card(
            title="策略淨收益 (金額)",
            value=f"${net_pnl_amt:,.2f}",
            delta_str=f"初始資金 ${initial_cap:,.0f}",
            is_positive=net_pnl_amt >= 0
        )
    with kpi_col3:
        render_kpi_card(
            title="年化夏普值 (Sharpe)",
            value=f"{metrics['sharpe']:.2f}",
            delta_str="無風險利率: 1.5%",
            is_positive=metrics['sharpe'] > 1
        )
    with kpi_col4:
        render_kpi_card(
            title="最大回撤 (Max DD)",
            value=f"{metrics['max_dd'] * 100:.2f}%",
            delta_str="資金池最大跌幅",
            is_positive=metrics['max_dd'] > -0.15 # 大於 -15% 為綠色，否則紅色
        )

    # 績效詳情表格
    st.markdown("### 📋 策略績效對照表")
    metrics_data = {
        "分析指標": ["累積損益 (%)", "年化夏普值", "最大回撤 (%)", "回測勝率 (%)", "總交易次數", "基準(Buy & Hold)損益 (%)"],
        "量化數值": [
            f"{metrics['total_return']*100:.2f}%",
            f"{metrics['sharpe']:.2f}",
            f"{metrics['max_dd']*100:.2f}%",
            f"{metrics['win_rate']*100:.1f}%",
            f"{metrics['num_trades']} 次",
            f"{metrics['benchmark_return']*100:.2f}%"
        ]
    }
    st.table(pd.DataFrame(metrics_data).set_index("分析指標"))

    # Plotly 互動圖表區
    st.markdown("### 📈 互動式圖表分析")
    
    # 圖表 1: 權益與水下圖（含買賣點）
    st.plotly_chart(
        plot_equity_and_drawdown(res['equity_curve'], res['benchmark_curve'], res['positions']),
        use_container_width=True
    )
    
    # 圖表 2: K線圖與指標 (只在單一因子模式下顯示相應技術線)
    st.markdown("### 🕯️ 標的 K 線圖與量價分析")
    sma20_line = None
    sma60_line = None
    if single_factor == "MA Cross":
        _, sma20_line, sma60_line = calc_ma_cross(df)
    
    st.plotly_chart(plot_candlestick(df, sma20_line, sma60_line), use_container_width=True)
    
    # 圖表 3: 月損益長條圖
    st.plotly_chart(plot_monthly_returns(res['net_returns']), use_container_width=True)

    # 買賣導明明細表
    if mode == "單一因子" and single_factor not in ["P/E Ratio", "P/B Ratio", "ROE", "Gross Margin", "Inventory Turnover"]:
        with st.expander("📋 最佳參數之實際交易明細", expanded=True):
            # 取得用於交易日誌的指標專屬數列
            ind_series = None
            if single_factor == "RSI":
                _, ind_series = calc_rsi(df, p_rsi_period, p_rsi_oversold, p_rsi_overbought)
            elif single_factor == "MA Cross":
                _, ind_series, _ = calc_ma_cross(df, p_ma_fast, p_ma_slow)
            elif single_factor == "Stochastic KD":
                _, ind_series, _ = calc_kd(df, p_kd_period, p_kd_d, p_kd_oversold, p_kd_overbought)
            elif single_factor == "MACD":
                _, _, _, ind_series = calc_macd(df, p_macd_fast, p_macd_slow, p_macd_signal)
            elif single_factor == "Bollinger Bands":
                _, _, ind_series, _ = calc_bbands(df, p_bb_period, p_bb_std)
            elif single_factor == "ATR Trend":
                _, ind_series = calc_atr(df, p_atr_period)
            elif single_factor == "ADX":
                _, ind_series = calc_adx(df, p_adx_period, p_adx_threshold)
            elif single_factor == "CCI":
                _, ind_series = calc_cci(df, p_cci_period, p_cci_oversold, p_cci_overbought)
            elif single_factor == "OBV":
                _, ind_series, _ = calc_obv(df, p_obv_ema_period)
            elif single_factor == "Williams %R":
                _, ind_series = calc_williams_r(df, p_wr_period, p_wr_oversold, p_wr_overbought)

            delayed_pos = res['positions']

            if ind_series is not None:
                trade_log = build_trade_log(df, delayed_pos, ind_series, single_factor)
                if trade_log.empty:
                    st.info("在設定的日期區間內，此因子組合沒有產生交易訊號。")
                else:
                    # 紙表色彩標註
                    def color_pnl(val):
                        try:
                            return 'color: #00e676' if float(str(val).replace('%','').replace('+','')) > 0 else 'color: #ff1744'
                        except:
                            return ''
                    styled = trade_log.set_index("#").style.map(color_pnl, subset=["單筆損益 (%)"])
                    st.dataframe(styled, use_container_width=True)

# ----------------- Tab 2: 策略比對 -----------------
with tab2:
    st.markdown("### ⚖️ 多因子策略對照實驗室")
    st.write("在側邊欄進行多輪模擬回測並儲存後，在此可選擇多個模擬結果進行同時比對。")
    
    if len(st.session_state.simulation_history) < 2:
        st.warning("⚠️ 請先在左側邊欄設定不同參數並點擊「開始模擬並儲存記錄」，儲存至少 2 個策略，才能在此進行比對！")
    else:
        # 選擇要比對的策略
        sim_options = {sim['id']: sim['name'] for sim in st.session_state.simulation_history}
        selected_ids = st.multiselect(
            "選擇要比對的策略記錄",
            options=list(sim_options.keys()),
            default=list(sim_options.keys())[:3], # 預設選前三個
            format_func=lambda x: sim_options[x]
        )
        
        if len(selected_ids) > 0:
            # 建立多條權益曲線 Plotly Chart
            fig_compare = go.Figure()
            
            # 先畫基準 Benchmark
            # 找到其中一個選定策略的 Benchmark 資料
            ref_sim = next(s for s in st.session_state.simulation_history if s['id'] == selected_ids[0])
            dates_parsed = [datetime.datetime.strptime(d, "%Y-%m-%d") for d in ref_sim['dates']]
            
            fig_compare.add_trace(go.Scatter(
                x=dates_parsed,
                y=ref_sim['benchmark_curve'],
                name=f"基準 {ref_sim['ticker']} (Buy & Hold)",
                line=dict(color='#8a99ad', width=1.5, dash='dash')
            ))
            
            # 畫策略曲線
            compare_data = []
            for sim_id in selected_ids:
                sim = next(s for s in st.session_state.simulation_history if s['id'] == sim_id)
                sim_dates = [datetime.datetime.strptime(d, "%Y-%m-%d") for d in sim['dates']]
                
                fig_compare.add_trace(go.Scatter(
                    x=sim_dates,
                    y=sim['equity_curve'],
                    name=sim['name'].split(" | ")[1] + f" ({sim['ticker']})",
                    line=dict(width=2.5)
                ))
                
                # 蒐集表格資料
                m = sim['metrics']
                compare_data.append({
                    "模擬名稱": sim['name'],
                    "標的": sim['ticker'],
                    "累積損益 (%)": f"{m['total_return']*100:.2f}%",
                    "夏普值": f"{m['sharpe']:.2f}",
                    "最大回撤 (%)": f"{m['max_dd']*100:.2f}%",
                    "勝率 (%)": f"{m['win_rate']*100:.1f}%",
                    "交易次數": f"{m['num_trades']} 次"
                })
                
            fig_compare.update_layout(
                title="複合權益曲線 (Equity Curves) 對照圖",
                template='plotly_dark',
                xaxis_title="時間",
                yaxis_title="投資組合淨值 ($)",
                height=500,
                margin=dict(l=30, r=30, t=50, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_compare, use_container_width=True)
            
            # 績效對照表
            st.markdown("### 📋 綜合績效指標對照表")
            st.table(pd.DataFrame(compare_data).set_index("模擬名稱"))
        else:
            st.info("請在上方下拉選單中勾選至少一個模擬策略以顯示對照圖。")

# ----------------- Tab 3: 同業策略適用性比對 -----------------
with tab3:
    st.markdown("### 🔍 同業策略適用性比對")
    st.write("從左側模擬歷史紀錄中選擇策略，套用到同市場其他標的，驗證投資規則的普適性。")

    # ── Section A: 選擇策略 ──────────────────────────────────
    st.markdown("#### 📂 Step 1｜選擇要套用的策略")
    if not st.session_state.simulation_history:
        st.warning("⚠️ 尚無模擬歷史紀錄。請先在左側邊欄設定參數並點擊「開始模擬並儲存記錄」，至少儲存一個策略後再使用此功能。")
    else:
        sim_options_t3 = {sim['id']: sim['name'] for sim in st.session_state.simulation_history}
        selected_sim_ids = st.multiselect(
            "勾選要比對的策略（可多選）",
            options=list(sim_options_t3.keys()),
            default=list(sim_options_t3.keys())[:1],
            format_func=lambda x: sim_options_t3[x]
        )

        if not selected_sim_ids:
            st.info("請至少勾選一個策略。")
        else:
            selected_sims = [s for s in st.session_state.simulation_history if s['id'] in selected_sim_ids]

            # ── Section B: 選擇比對標的 ────────────────────────────
            st.markdown("---")
            st.markdown("#### 🗺️ Step 2｜選擇比對標的")

            col_mkt, col_sec, col_stk = st.columns([1, 1, 2])
            with col_mkt:
                mkt_choice = st.selectbox("市場", list(PEER_DB.keys()), key="peer_mkt")
            with col_sec:
                sec_list = list(PEER_DB[mkt_choice].keys())
                sec_choice = st.selectbox("產業類別", sec_list, key="peer_sec")
            with col_stk:
                stk_dict = PEER_DB[mkt_choice][sec_choice]   # {ticker: name}
                stk_options = list(stk_dict.keys())
                stk_labels  = [f"{t} {n}" for t, n in stk_dict.items()]
                sel_stk_labels = st.multiselect(
                    "個股選擇（可多選）",
                    options=stk_labels,
                    default=stk_labels[:5],
                    key="peer_stocks"
                )
            # 額外手動輸入
            extra_input = st.text_input(
                "額外手動輸入代碼（逗號分隔，選填）",
                value="",
                placeholder="例: 2330.TW, NVDA",
                key="peer_extra"
            )
            # 組合最終清單
            peer_list_t3 = [lbl.split()[0] for lbl in sel_stk_labels]
            if extra_input.strip():
                peer_list_t3 += [t.strip() for t in extra_input.split(",") if t.strip()]
            peer_list_t3 = list(dict.fromkeys(peer_list_t3))   # 去重

            st.caption(f"📋 將比對的標的（共 {len(peer_list_t3)} 檔）：{', '.join(peer_list_t3)}")

            # 參考連結：查詢股票代碼
            with st.expander("🔗 股票代碼查詢參考連結", expanded=False):
                st.markdown("""
| 市場 | 平台 | 說明 |
|------|------|------|
| 🇹🇼 台股 | [台灣證券交易所 TWSE 上市公司查詢](https://isin.twse.com.tw/isin/C_public.jsp?strMode=2) | 完整上市股票代碼清單 |
| 🇹🇼 台股OTC | [台灣OTC上櫃公司查詢](https://isin.twse.com.tw/isin/C_public.jsp?strMode=4) | 上櫃股票（代碼加 .TWO）|
| 🇹🇼 台股 | [Yahoo 台股代碼搜尋](https://tw.stock.yahoo.com/s/list.php?c=%E5%8D%8A%E5%B0%8E%E9%AB%94) | 依產業類別瀏覽 |
| 🇺🇸 美股 | [Yahoo Finance Stock Screener](https://finance.yahoo.com/screener/) | 依產業/市值篩選 |
| 🇺🇸 美股 | [Nasdaq Company Listing](https://www.nasdaq.com/market-activity/stocks/screener) | Nasdaq 完整清單 |

**代碼格式說明：**
- 台股上市：`代碼.TW`（例：`2330.TW`）
- 台股上櫃：`代碼.TWO`（例：`5347.TWO`）
- 美股：直接輸入代碼（例：`NVDA`）
                """)


            st.markdown("---")
            if st.button("🚀 執行同業策略比對", use_container_width=True, key="run_peer_t3"):
                if not peer_list_t3:
                    st.error("請先選擇至少一個比對標的。")
                else:
                    with st.spinner("下載同業歷史資料並套用策略回測中，請稍候..."):
                        peer_data_t3 = fetch_peer_stock_data(tuple(peer_list_t3), start_date, end_date)

                    if not peer_data_t3:
                        st.error("所有標的資料下載失敗，請確認代碼正確性及網路連線。")
                    else:
                        # 計算每個 (策略, 標的) 的回測結果
                        all_results  = []   # for table
                        chart_traces = []   # for bar chart

                        bar_colors = ['#00e5ff','#7c4dff','#00e676','#ff6d00','#ff4081']

                        for sim_idx, sim in enumerate(selected_sims):
                            sim_label = sim['name'].replace(" | ", "\n")
                            sim_rets  = []
                            sim_bh    = []
                            sim_tickers = []

                            for pt, pdf in peer_data_t3.items():
                                try:
                                    pres = run_sim_on_df(pdf, sim['params'], transaction_fee=sim['params'].get('cost', 0.15) / 100.0)
                                    pm   = pres['metrics']
                                    pt_name = PEER_DB.get(mkt_choice, {}).get(sec_choice, {}).get(pt, pt)
                                    display_name = f"{pt} {pt_name}" if pt_name != pt else pt

                                    all_results.append({
                                        '標的':          display_name,
                                        '策略':          sim['name'],
                                        '策略累積報酬 (%)': round(pm['total_return'] * 100, 2),
                                        'Buy & Hold (%)':  round(pm['benchmark_return'] * 100, 2),
                                        '夏普值':        round(pm['sharpe'], 2),
                                        '最大回撤 (%)':   round(pm['max_dd'] * 100, 2),
                                        '勝率 (%)':       round(pm['win_rate'] * 100, 1),
                                        '交易次數':       pm['num_trades'],
                                    })
                                    sim_rets.append(round(pm['total_return'] * 100, 2))
                                    sim_bh.append(round(pm['benchmark_return'] * 100, 2))
                                    sim_tickers.append(display_name)
                                except Exception:
                                    continue

                            if sim_rets:
                                color = bar_colors[sim_idx % len(bar_colors)]
                                chart_traces.append((sim['name'], sim_tickers, sim_rets, sim_bh, color))

                        if not all_results:
                            st.error("所有組合計算均失敗，請確認資料下載正常。")
                        else:
                            # ── 長條圖 ─────────────────────────────────────
                            fig_t3 = go.Figure()

                            # 先畫 Buy & Hold（只畫一次，用第一個策略的標的清單）
                            if chart_traces:
                                _, bh_tickers, _, bh_vals, _ = chart_traces[0]
                                fig_t3.add_trace(go.Bar(
                                    x=bh_tickers, y=bh_vals,
                                    name='Buy & Hold 基準',
                                    marker_color='#455a64',
                                    opacity=0.75
                                ))

                            for sim_name, stickers, s_rets, _, color in chart_traces:
                                fig_t3.add_trace(go.Bar(
                                    x=stickers, y=s_rets,
                                    name=sim_name,
                                    marker_color=color
                                ))

                            fig_t3.update_layout(
                                title=f"同業策略績效比對 — {mkt_choice} / {sec_choice}（{start_date} ~ {end_date}）",
                                barmode='group',
                                template='plotly_dark',
                                height=460,
                                margin=dict(l=20, r=20, t=55, b=50),
                                yaxis_title='累積報酬 (%)',
                                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                            )
                            st.plotly_chart(fig_t3, use_container_width=True)

                            # ── 詳細數據表格 ───────────────────────────────
                            st.markdown("#### 📋 同業回測績效詳細表")
                            result_df = pd.DataFrame(all_results)

                            def _color_ret(val):
                                try:
                                    v = float(val)
                                    return 'color: #00e676; font-weight:600' if v > 0 else 'color: #ff1744; font-weight:600'
                                except:
                                    return ''

                            styled_t3 = (
                                result_df.set_index(['策略','標的'])
                                .style.map(_color_ret, subset=['策略累積報酬 (%)'])
                                .map(_color_ret, subset=['Buy & Hold (%)'])
                            )
                            st.dataframe(styled_t3, use_container_width=True)
            else:
                st.info("完成上方策略與標的的選擇後，點擊「執行同業策略比對」開始運算。")
