import streamlit as st
import pandas as pd
import requests
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import StringIO
import streamlit.components.v1 as components
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- CONFIGURATION & TOOLS ---
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
MIN_BARS = 40
LOOKBACK_CANDLES = 20  # Aligned to 1 month worth of daily market movement
MAX_TICKERS_LIMIT = 500

SECTOR_MAP = {
    "XLK": "Technology", "XLY": "Consumer Discretionary", "XLC": "Communications",
    "XLF": "Financials", "XLV": "Healthcare", "XLI": "Industrials",
    "XLP": "Consumer Staples", "XLE": "Energy", "XLB": "Materials",
    "XLU": "Utilities", "XLRE": "Real Estate"
}

# --- PERSISTENT SESSION WITH RETRIES (CRITICAL FOR CLOUD HOSTING) ---
@st.cache_resource
def get_http_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


@dataclass
class ScreenResult:
    ticker: str
    close: float
    sma8: float
    sma20: float
    return_7: float
    spy_return_7: float
    rs_factor: float
    spy_rs_factor: float

    @property
    def factor_edge(self) -> float:
        return self.rs_factor - self.spy_rs_factor

    @property
    def dist_sma8(self) -> float:
        return ((self.close - self.sma8) / self.sma8) * 100

    @property
    def dist_sma20(self) -> float:
        return ((self.close - self.sma20) / self.sma20) * 100

    @property
    def entry_risk_sma20(self) -> float:
        return abs(self.dist_sma20)


# --- MATH & DATA FUNCTIONS ---
def calc_rs_factor(closes, lookback=LOOKBACK_CANDLES):
    window = closes.iloc[-lookback:]
    if len(window) < lookback:
        return None
    period_return = (window.iloc[-1] / window.iloc[0] - 1) * 100
    bar_returns = window.pct_change().dropna()
    if len(bar_returns) < 2:
        return None
    volatility = bar_returns.std() * 100
    if pd.isna(volatility) or volatility <= 0:
        return None
    return float(period_return), float(period_return / volatility)


@st.cache_data(ttl=86400)
def get_us_tickers(cap_size="Mid"):
    session = get_http_session()
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies" if cap_size == "Mid" else "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        resp = session.get(url, timeout=15)
        df = pd.read_html(StringIO(resp.text))[0]

        ticker_col = next((col for col in df.columns if "symbol" in str(col).lower() or "ticker" in str(col).lower()), df.columns[0])
        tickers = df[ticker_col].dropna().astype(str).str.strip().str.replace('.', '-', regex=False).tolist()

        return sorted(list(set(tickers[:MAX_TICKERS_LIMIT])))
    except Exception:
        return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "V", "JPM", "TSLA"]


def fetch_daily_ohlcv(ticker):
    session = get_http_session()
    try:
        response = session.get(
            YAHOO_CHART_URL.format(ticker=ticker),
            params={"interval": "1d", "range": "60d"},
            timeout=10
        )
        if response.status_code != 200:
            return None
        result = response.json()["chart"]["result"][0]
        quote = result["indicators"]["quote"][0]
        df = pd.DataFrame(
            {"close": quote.get("close")},
            index=pd.to_datetime(result["timestamp"], unit="s", utc=True)
        )
        df = df.dropna(subset=["close"])
        return df if len(df) >= MIN_BARS else None
    except Exception:
        return None


def get_quant_sector_strength(spy_df, mode="Long"):
    spy_metrics = calc_rs_factor(spy_df["close"])
    if not spy_metrics:
        return None

    sector_results = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(fetch_daily_ohlcv, etf): etf for etf in SECTOR_MAP.keys()}
        for future in as_completed(futures):
            etf = futures[future]
            try:
                df = future.result()
                if df is not None:
                    sec_metrics = calc_rs_factor(df["close"])
                    if sec_metrics:
                        edge = sec_metrics[1] - spy_metrics[1]
                        sector_results.append({
                            "ETF": etf,
                            "Sector Name": SECTOR_MAP[etf],
                            "Return %": sec_metrics[0],
                            "RS Factor Edge": edge
                        })
            except Exception:
                pass

    if sector_results:
        df_out = pd.DataFrame(sector_results)
        is_ascending = (mode == "Short")
        return df_out.sort_values(by="RS Factor Edge", ascending=is_ascending)
    return None


def screen_ticker(ticker, df, spy_metrics, mode="Long"):
    df = df.copy()
    df["SMA8"] = df["close"].rolling(8).mean()
    df["SMA20"] = df["close"].rolling(20).mean()
    latest = df.iloc[-1]

    if pd.isna(latest["SMA8"]) or pd.isna(latest["SMA20"]):
        return None

    close, sma8, sma20 = float(latest["close"]), float(latest["SMA8"]), float(latest["SMA20"])
    stock_metrics = calc_rs_factor(df["close"])
    if not stock_metrics:
        return None

    if mode == "Long":
        if latest["SMA8"] <= latest["SMA20"]: return None
        if close < sma8 and close < sma20: return None
        if stock_metrics[1] <= spy_metrics[1]: return None
    else:
        if latest["SMA8"] >= latest["SMA20"]: return None
        if close > sma8 and close > sma20: return None
        if stock_metrics[1] >= spy_metrics[1]: return None

    return ScreenResult(ticker, close, sma8, sma20, stock_metrics[0], spy_metrics[0], stock_metrics[1], spy_metrics[1])


def render_tv_chart(ticker: str, mode="Long"):
    ma_color_1 = "#00BFFF" if mode == "Long" else "#FF4B4B"
    ma_color_2 = "#1E90FF" if mode == "Long" else "#FFA500"
    tv_symbol = ticker.replace('-', '.')

    widget_html = f"""
    <div class="tradingview-widget-container" style="height:350px; margin-bottom:15px;">
        <div id="tv-chart-{ticker}" style="height:100%;"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">
        new TradingView.widget({{
            "autosize": true,
            "symbol": "{tv_symbol}",
            "interval": "D",
            "timezone": "Exchange",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "enable_publishing": false,
            "hide_side_toolbar": true,
            "allow_symbol_change": false,
            "container_id": "tv-chart-{ticker}",
            "studies": [
                {{ "id": "MASimple@tv-basicstudies", "inputs": {{ "length": 8 }} }},
                {{ "id": "MASimple@tv-basicstudies", "inputs": {{ "length": 20 }} }}
            ],
            "studies_overrides": {{
                "moving average.moving average.color": "{ma_color_1}",
                "moving average.moving average.linewidth": 2,
                "moving average 2.moving average.color": "{ma_color_2}",
                "moving average 2.moving average.linewidth": 2
            }}
        }});
        </script>
    </div>
    """
    components.html(widget_html, height=360)


# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="Daily Quant Multi-Directional Screener", layout="wide")

st.sidebar.header("🧭 Navigation & Strategy")
strategy_mode = st.sidebar.radio(
    "Select Trading Bias Mode:",
    [
        "🚀 Mid Cap Long Setup",
        "💥 Mid Cap Short Setup",
        "📈 Large Cap Long Setup",
        "📉 Large Cap Short Setup"
    ]
)

cap_clean = "Mid" if "Mid Cap" in strategy_mode else "Large"
mode_clean = "Long" if "Long Setup" in strategy_mode else "Short"

st.title(f"{strategy_mode} (Daily Chart Strategy)")

# --- RESPONSIVE MOBILE CSS FIX FOR METRICS ---
st.markdown(
    """
    <style>
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        white-space: normal !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
        white-space: normal !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.75rem !important;
    }
    [data-testid="stMetric"] { 
        background-color: #1E1E1E;
        border: 1px solid #333333;
        padding: 6px 10px;
        border-radius: 6px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if "spy_data" not in st.session_state or st.session_state.spy_data is None:
    st.session_state.spy_data = fetch_daily_ohlcv("SPY")

tab_home, tab_screener, tab_entries = st.tabs([
    "📊 Sector Performance Tracker",
    "🔍 Live Daily Quant Filters",
    "🎯 Optimal Entry Risk Profile"
])

# --- TAB 1: QUANT SECTOR OVERVIEW ---
with tab_home:
    st.header(f"📊 Quant Sector Relative {'Strength' if mode_clean == 'Long' else 'Weakness'}")
    st.write(f"Sectors ranked dynamically against SPY over the last {LOOKBACK_CANDLES} daily candles.")

    if st.session_state.spy_data is None:
        st.error("Failed to fetch SPY benchmark data.")
    else:
        cache_key = f"sector_df_{mode_clean}"
        if st.button("🔄 Refresh Sector Metrics") or cache_key not in st.session_state:
            with st.spinner("Re-calculating math variables..."):
                st.session_state[cache_key] = get_quant_sector_strength(st.session_state.spy_data, mode=mode_clean)

        sectors_df = st.session_state.get(cache_key)

        if sectors_df is not None and not sectors_df.empty:
            prime_sector = sectors_df.iloc[0]
            if mode_clean == "Long":
                st.success(f"🔥 **Leading Market Sector:** {prime_sector['Sector Name']} ({prime_sector['ETF']}) with an RS Edge of **{prime_sector['RS Factor Edge']:+.2f}**")
            else:
                st.error(f"⚠️ **Weakest Market Sector:** {prime_sector['Sector Name']} ({prime_sector['ETF']}) with an RS Deficit of **{prime_sector['RS Factor Edge']:.2f}**")

            st.dataframe(sectors_df, hide_index=True, use_container_width=True)

# --- SIDEBAR CONTROL SCAN ACTION ---
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Execute Scan Engine")
run_scan = st.sidebar.button(f"🚀 Execute Selected Scan Setup", type="primary", use_container_width=True)

scan_key = f"scan_results_{cap_clean.lower()}_{mode_clean.lower()}"
if scan_key not in st.session_state:
    st.session_state[scan_key] = None

if run_scan:
    if st.session_state.spy_data is None:
        st.session_state.spy_data = fetch_daily_ohlcv("SPY")

    if st.session_state.spy_data is None:
        st.error("Could not fetch SPY baseline structural indexes.")
    else:
        status_text = st.empty()
        progress_bar = st.progress(0)

        status_text.text("Calculating Benchmark Metrics...")
        spy_metrics = calc_rs_factor(st.session_state.spy_data["close"])

        tickers = get_us_tickers(cap_size=cap_clean)
        results = []

        status_text.text(f"Screening {len(tickers)} symbols for {cap_clean} Cap {mode_clean} parameters...")

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(fetch_daily_ohlcv, t): t for t in tickers}
            for i, future in enumerate(as_completed(futures)):
                ticker = futures[future]
                progress_bar.progress((i + 1) / len(tickers))
                try:
                    df = future.result()
                    if df is not None:
                        hit = screen_ticker(ticker, df, spy_metrics, mode=mode_clean)
                        if hit: results.append(hit)
                except Exception:
                    pass

        status_text.text("Completed scanning algorithm successfully!")
        progress_bar.empty()
        st.session_state[scan_key] = results


def display_paginated_charts(results_list, key_prefix="screener", run_mode="Long", run_cap="Mid"):
    if not results_list:
        st.info("No tickers matched the current scan criteria.")
        return

    st.success(f"Found {len(results_list)} structurally strong {run_cap} Cap matches!")

    CHARTS_PER_PAGE = 4
    total_pages = max(1, (len(results_list) - 1) // CHARTS_PER_PAGE + 1)

    page_state_key = f"page_num_{key_prefix}_{run_cap.lower()}_{run_mode.lower()}"
    if page_state_key not in st.session_state:
        st.session_state[page_state_key] = 1

    if st.session_state[page_state_key] > total_pages:
        st.session_state[page_state_key] = total_pages

    current_page = st.session_state[page_state_key]

    # --- TOP NAVIGATION BAR ---
    st.markdown("---")
    col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
    with col_p1:
        if st.button("⬅️ Prev", key=f"top_prev_{key_prefix}_{run_cap}_{run_mode}", use_container_width=True):
            if st.session_state[page_state_key] > 1:
                st.session_state[page_state_key] -= 1
                st.rerun()
    with col_p2:
        new_page_top = st.number_input(
            "Page", min_value=1, max_value=total_pages, value=int(current_page), step=1,
            key=f"top_num_{key_prefix}_{run_cap}_{run_mode}", label_visibility="collapsed"
        )
        if new_page_top != st.session_state[page_state_key]:
            st.session_state[page_state_key] = new_page_top
            st.rerun()
    with col_p3:
        if st.button("Next ➡️", key=f"top_next_{key_prefix}_{run_cap}_{run_mode}", use_container_width=True):
            if st.session_state[page_state_key] < total_pages:
                st.session_state[page_state_key] += 1
                st.rerun()

    current_page = st.session_state[page_state_key]
    start_idx = (current_page - 1) * CHARTS_PER_PAGE
    end_idx = start_idx + CHARTS_PER_PAGE
    page_results = results_list[start_idx:end_idx]

    st.markdown(f"<p style='text-align: center; color: #888; font-size: 0.85rem;'>Page <b>{current_page}</b> of <b>{total_pages}</b> &nbsp;|&nbsp; Showing matches <b>{start_idx + 1}</b> to <b>{min(end_idx, len(results_list))}</b> of {len(results_list)}</p>", unsafe_allow_html=True)

    # --- CHARTS GRID ---
    for idx in range(0, len(page_results), 3):
        chunk = page_results[idx:idx + 3]
        cols = st.columns(3)
        for col, r in zip(cols, chunk):
            with col:
                st.markdown(f"### 📊 {r.ticker}")
                m_c1, m_c2, m_c3 = st.columns(3)
                m_c1.metric("Close", f"${r.close:.2f}")
                m_c2.metric("RS Edge", f"{r.factor_edge:+.2f}" if run_mode == "Long" else f"{r.factor_edge:.2f}")
                m_c3.metric("Dist 20SMA", f"{r.dist_sma20:.2f}%")
                render_tv_chart(r.ticker, mode=run_mode)

    # --- BOTTOM NAVIGATION BAR ---
    st.markdown("---")
    col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
    with col_b1:
        if st.button("⬅️ Prev", key=f"bot_prev_{key_prefix}_{run_cap}_{run_mode}", use_container_width=True):
            if st.session_state[page_state_key] > 1:
                st.session_state[page_state_key] -= 1
                st.rerun()
    with col_b2:
        new_page_bot = st.number_input(
            "Page", min_value=1, max_value=total_pages, value=int(current_page), step=1,
            key=f"bot_num_{key_prefix}_{run_cap}_{run_mode}", label_visibility="collapsed"
        )
        if new_page_bot != st.session_state[page_state_key]:
            st.session_state[page_state_key] = new_page_bot
            st.rerun()
    with col_b3:
        if st.button("Next ➡️", key=f"bot_next_{key_prefix}_{run_cap}_{run_mode}", use_container_width=True):
            if st.session_state[page_state_key] < total_pages:
                st.session_state[page_state_key] += 1
                st.rerun()


# --- TAB 2: ACTIVE SCREENER ---
with tab_screener:
    if st.session_state.get(scan_key) is not None:
        is_reverse = (mode_clean == "Long")
        rs_sorted = sorted(st.session_state[scan_key], key=lambda r: r.factor_edge, reverse=is_reverse)
        display_paginated_charts(rs_sorted, key_prefix="rs", run_mode=mode_clean, run_cap=cap_clean)
    else:
        st.info("👈 Please click **'Execute Selected Scan Setup'** in the sidebar to run the momentum screener.")

# --- TAB 3: OPTIMAL ENTRIES ---
with tab_entries:
    if st.session_state.get(scan_key) is not None:
        entry_sorted = sorted(st.session_state[scan_key], key=lambda r: r.entry_risk_sma20)
        display_paginated_charts(entry_sorted, key_prefix="entry", run_mode=mode_clean, run_cap=cap_clean)
    else:
        st.info("👈 Please click **'Execute Selected Scan Setup'** in the sidebar to load optimal entries.")
