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
def calc_ma_cross(df):
    sma20 = df['Close'].rolling(window=20).mean()
    sma60 = df['Close'].rolling(window=60).mean()
    signal = np.where(sma20 > sma60, 1.0, -1.0)
    signal[sma60.isna()] = 0.0
    return pd.Series(signal, index=df.index), sma20, sma60

def calc_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / np.where(loss == 0, 1e-10, loss)
    rsi = 100 - (100 / (1 + rs))
    signal = np.zeros(len(df))
    signal[rsi < 30] = 1.0
    signal[rsi > 70] = -1.0
    signal[rsi.isna()] = 0.0
    return pd.Series(signal, index=df.index), rsi

def calc_kd(df, k_period=14, d_period=3):
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    rsv = 100 * (df['Close'] - low_min) / np.where(high_max - low_min == 0, 1e-10, high_max - low_min)
    k_val = rsv.rolling(window=3).mean()
    d_val = k_val.rolling(window=d_period).mean()
    signal = np.zeros(len(df))
    prev_k = k_val.shift(1)
    prev_d = d_val.shift(1)
    buy_cond = (k_val > d_val) & (prev_k <= prev_d) & (k_val < 20)
    sell_cond = (k_val < d_val) & (prev_k >= prev_d) & (k_val > 80)
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

def calc_adx(df, period=14):
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
    buy_cond = (adx > 25) & (plus_di > minus_di)
    sell_cond = (adx > 25) & (minus_di > plus_di)
    signal[buy_cond] = 1.0
    signal[sell_cond] = -1.0
    signal[adx.isna()] = 0.0
    return pd.Series(signal, index=df.index), adx

def calc_cci(df, period=20):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    def mean_deviation(x):
        return np.mean(np.abs(x - np.mean(x)))
    mad = tp.rolling(window=period).apply(mean_deviation, raw=True)
    cci = (tp - sma_tp) / (0.015 * np.where(mad == 0, 1e-10, mad))
    signal = np.zeros(len(df))
    signal[cci < -100] = 1.0
    signal[cci > 100] = -1.0
    signal[cci.isna()] = 0.0
    return pd.Series(signal, index=df.index), cci

def calc_obv(df):
    vol = df['Volume']
    close_diff = df['Close'].diff()
    sign = np.zeros(len(df))
    sign[close_diff > 0] = 1.0
    sign[close_diff < 0] = -1.0
    obv_change = sign * vol
    obv = obv_change.cumsum()
    obv_ema = obv.ewm(span=20, adjust=False).mean()
    signal = np.where(obv > obv_ema, 1.0, -1.0)
    signal[obv.isna() | obv_ema.isna()] = 0.0
    return pd.Series(signal, index=df.index), obv, obv_ema

def calc_williams_r(df, period=14):
    high_max = df['High'].rolling(window=period).max()
    low_min = df['Low'].rolling(window=period).min()
    williams = -100 * (high_max - df['Close']) / np.where(high_max - low_min == 0, 1e-10, high_max - low_min)
    signal = np.zeros(len(df))
    signal[williams < -80] = 1.0
    signal[williams > -20] = -1.0
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
def calc_all_signals(df, fundamentals):
    signals = {}
    
    # 1. MA Cross
    ma_sig, _, _ = calc_ma_cross(df)
    signals['MA Cross'] = ma_sig
    
    # 2. RSI
    rsi_sig, _ = calc_rsi(df)
    signals['RSI'] = rsi_sig
    
    # 3. KD
    kd_sig, _, _ = calc_kd(df)
    signals['Stochastic KD'] = kd_sig
    
    # 4. MACD
    macd_sig, _, _, _ = calc_macd(df)
    signals['MACD'] = macd_sig
    
    # 5. BBands
    bb_sig, _, _, _ = calc_bbands(df)
    signals['Bollinger Bands'] = bb_sig
    
    # 6. ATR
    atr_sig, _ = calc_atr(df)
    signals['ATR Trend'] = atr_sig
    
    # 7. ADX
    adx_sig, _ = calc_adx(df)
    signals['ADX'] = adx_sig
    
    # 8. CCI
    cci_sig, _ = calc_cci(df)
    signals['CCI'] = cci_sig
    
    # 9. OBV
    obv_sig, _, _ = calc_obv(df)
    signals['OBV'] = obv_sig
    
    # 10. Williams %R
    wr_sig, _ = calc_williams_r(df)
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
        
    # 量能長條圖 (紅綠比對)
    colors = ['#ff1744' if cl < op else '#00e676' for cl, op in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker_color=colors
    ), row=2, col=1)
    
    fig.update_layout(
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=30, r=30, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_equity_and_drawdown(equity_curve, benchmark_curve):
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
        height=500,
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
def_ticker = st.session_state.get('ticker', '2330.TW')
def_start = st.session_state.get('start_date', datetime.date(2021, 1, 1))
def_end = st.session_state.get('end_date', datetime.date(2025, 1, 1))
def_mode = st.session_state.get('mode', '單一因子')
def_cost = st.session_state.get('cost', 0.15) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("📈 基本配置")
ticker = st.sidebar.text_input("股票代碼", value=def_ticker, key="ticker")
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

single_factor = ""
weights = {}
threshold = 0.1

if mode == "單一因子":
    def_single = st.session_state.get('single_factor', 'MA Cross')
    if def_single not in factors_list:
        def_single = 'MA Cross'
    single_factor = st.sidebar.selectbox("選擇交易因子", factors_list, index=factors_list.index(def_single), key="single_factor")
else:
    def_thresh = st.session_state.get('threshold', 0.1)
    threshold = st.sidebar.slider("複合訊號交易門檻", min_value=0.0, max_value=0.5, value=def_thresh, step=0.05, key="threshold")
    
    st.sidebar.markdown("**調整因子權重 (0 - 100%)**")
    for f in factors_list:
        def_w = st.session_state.get(f"w_{f}", 20 if f in ["MA Cross", "RSI", "MACD"] else 0)
        weights[f] = st.sidebar.slider(f, min_value=0, max_value=100, value=def_w, step=5, key=f"w_{f}")

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
    signals_dict = calc_all_signals(df, fundamentals)
    
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
                'weights': weights.copy()
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
tab1, tab2 = st.tabs(["📊 單一回測分析", "⚖️ 策略比對對照"])

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
    
    # 圖表 1: 權益與水下圖
    st.plotly_chart(plot_equity_and_drawdown(res['equity_curve'], res['benchmark_curve']), use_container_width=True)
    
    # 圖表 2: K線圖與指標 (只在單一因子模式下顯示相應技術線)
    st.markdown("### 🕯️ 標的 K 線圖與量價分析")
    sma20_line = None
    sma60_line = None
    if single_factor == "MA Cross":
        _, sma20_line, sma60_line = calc_ma_cross(df)
    
    st.plotly_chart(plot_candlestick(df, sma20_line, sma60_line), use_container_width=True)
    
    # 圖表 3: 月損益長條圖
    st.plotly_chart(plot_monthly_returns(res['net_returns']), use_container_width=True)

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
