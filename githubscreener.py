from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import StringIO
import re
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- CONFIGURATION & TOOLS ---
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
        " like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
MIN_BARS = 130  # Required for 120-day (6-month) RS calculation
MAX_TICKERS_LIMIT = 500

# Expanded Sector & Industry Bellwethers Map
SECTOR_MAP = {
    # Broad GICS Sectors
    "XLK": "Technology",
    "XLY": "Consumer Discretionary",
    "XLC": "Communications",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    # High-Beta Industry Focus
    "SMH": "Semiconductors",
    "IGV": "Software",
    "XBI": "Biotech",
    "XHB": "Homebuilders",
    "ITA": "Aerospace & Defense",
    "XRT": "Retail",
}


# --- PERSISTENT SESSION WITH RETRIES ---
@st.cache_resource
def get_http_session():
  session = requests.Session()
  session.headers.update(HEADERS)
  retry_strategy = Retry(
      total=3,
      backoff_factor=0.3,
      status_forcelist=[429, 500, 502, 503, 504],
  )
  adapter = HTTPAdapter(
      pool_connections=20, pool_maxsize=20, max_retries=retry_strategy
  )
  session.mount("https://", adapter)
  session.mount("http://", adapter)
  return session


@dataclass
class ScreenResult:
  ticker: str
  close: float
  ema10: float
  sma20: float
  adr_pct: float
  vol_dryup_ratio: float
  composite_rs: float
  spy_composite_rs: float

  @property
  def factor_edge(self) -> float:
    return self.composite_rs - self.spy_composite_rs

  @property
  def dist_ema10(self) -> float:
    return ((self.close - self.ema10) / self.ema10) * 100

  @property
  def dist_sma20(self) -> float:
    return ((self.close - self.sma20) / self.sma20) * 100

  @property
  def entry_risk_sma20(self) -> float:
    return abs(self.dist_sma20)


# --- ASYMMETRIC MULTI-TIMEFRAME RS & QUANT MATH ---
def calc_rs_metrics(df, mode="Long"):
  """Calculates Asymmetric Composite RS Score weighted across 1M, 3M, and 6M.

  - Longs: Denominator is Downside Deviation (penalizes only red pullbacks).
  - Shorts: Denominator is Upside Deviation (penalizes only green squeeze
  bounces).
  """
  closes = df["close"]
  if len(closes) < MIN_BARS:
    return None

  # Multi-window returns
  ret_20 = (closes.iloc[-1] / closes.iloc[-20] - 1) * 100
  ret_60 = (closes.iloc[-1] / closes.iloc[-60] - 1) * 100
  ret_120 = (closes.iloc[-1] / closes.iloc[-120] - 1) * 100
  composite_return = (0.4 * ret_20) + (0.4 * ret_60) + (0.2 * ret_120)

  # Asymmetric Volatility Calculation over recent month (20 bars)
  bar_returns = closes.iloc[-20:].pct_change().dropna()

  if mode == "Long":
    downside_returns = bar_returns[bar_returns < 0]
    risk_metric = (
        float((downside_returns**2).mean() ** 0.5) * 100
        if len(downside_returns) > 0
        else 0.001
    )
    if risk_metric <= 0:
      risk_metric = 0.001
    composite_rs = composite_return / risk_metric
  else:
    upside_returns = bar_returns[bar_returns > 0]
    risk_metric = (
        float((upside_returns**2).mean() ** 0.5) * 100
        if len(upside_returns) > 0
        else 0.001
    )
    if risk_metric <= 0:
      risk_metric = 0.001
    inverted_return = -composite_return
    composite_rs = inverted_return / risk_metric

  # ADR% Calculation (Average Daily Range over 20 days)
  if "high" in df.columns and "low" in df.columns:
    daily_range = (
        (df["high"].iloc[-20:] - df["low"].iloc[-20:]) / df["low"].iloc[-20:]
    ) * 100
    adr_pct = float(daily_range.mean())
  else:
    adr_pct = float(risk_metric * 1.5)

  # Volume Dry-Up Ratio (5-day avg volume vs 20-day avg volume)
  if "volume" in df.columns:
    vol_5 = df["volume"].iloc[-5:].mean()
    vol_20 = df["volume"].iloc[-20:].mean()
    vol_dryup = float(vol_5 / vol_20) if vol_20 > 0 else 1.0
  else:
    vol_dryup = 1.0

  return (
      float(composite_return),
      float(composite_rs),
      float(adr_pct),
      float(vol_dryup),
  )


@st.cache_data(ttl=86400)
def get_us_tickers(cap_size="Mid"):
  session = get_http_session()
  try:
    url = (
        "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
        if cap_size == "Mid"
        else "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )
    resp = session.get(url, timeout=15)
    df = pd.read_html(StringIO(resp.text))[0]

    ticker_col = next(
        (
            col
            for col in df.columns
            if "symbol" in str(col).lower() or "ticker" in str(col).lower()
        ),
        df.columns[0],
    )
    tickers = (
        df[ticker_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.replace(".", "-", regex=False)
        .tolist()
    )

    return sorted(list(set(tickers[:MAX_TICKERS_LIMIT])))
  except Exception:
    return [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL",
        "META",
        "BRK-B",
        "LLY",
        "AVGO",
        "V",
        "JPM",
        "TSLA",
    ]


def fetch_daily_ohlcv(ticker):
  session = get_http_session()
  try:
    response = session.get(
        YAHOO_CHART_URL.format(ticker=ticker),
        params={"interval": "1d", "range": "180d"},
        timeout=10,
    )
    if response.status_code != 200:
      return None
    result = response.json()["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "close": quote.get("close"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "volume": quote.get("volume"),
        },
        index=pd.to_datetime(result["timestamp"], unit="s", utc=True),
    )
    df = df.dropna(subset=["close"])
    return df if len(df) >= MIN_BARS else None
  except Exception:
    return None


def get_quant_sector_strength(spy_df, mode="Long"):
  spy_metrics = calc_rs_metrics(spy_df, mode=mode)
  if not spy_metrics:
    return None

  sector_results = []
  with ThreadPoolExecutor(max_workers=10) as pool:
    futures = {
        pool.submit(fetch_daily_ohlcv, etf): etf for etf in SECTOR_MAP.keys()
    }
    for future in as_completed(futures):
      etf = futures[future]
      try:
        df = future.result()
        if df is not None:
          sec_metrics = calc_rs_metrics(df, mode=mode)
          if sec_metrics:
            edge = sec_metrics[1] - spy_metrics[1]
            sector_results.append({
                "ETF": etf,
                "Sector Name": SECTOR_MAP[etf],
                "Composite Return %": sec_metrics[0],
                "RS Factor Edge": edge,
            })
      except Exception:
        pass

  if sector_results:
    df_out = pd.DataFrame(sector_results)
    return df_out.sort_values(by="RS Factor Edge", ascending=False)
  return None


# --- DEDICATED LONG SCREENER ---
def screen_long_ticker(ticker, df, spy_metrics):
  df = df.copy()
  df["EMA10"] = df["close"].ewm(span=10, adjust=False).mean()
  df["SMA20"] = df["close"].rolling(20).mean()
  latest = df.iloc[-1]

  if pd.isna(latest["EMA10"]) or pd.isna(latest["SMA20"]):
    return None

  close, ema10, sma20 = (
      float(latest["close"]),
      float(latest["EMA10"]),
      float(latest["SMA20"]),
  )
  stock_metrics = calc_rs_metrics(df, mode="Long")
  if not stock_metrics:
    return None

  _, stock_rs, adr_pct, vol_dryup = stock_metrics
  spy_rs = spy_metrics[1]

  if adr_pct < 2.5:
    return None
  if ema10 <= sma20:
    return None
  if close < ema10 and close < sma20:
    return None
  if stock_rs <= spy_rs:
    return None

  return ScreenResult(
      ticker, close, ema10, sma20, adr_pct, vol_dryup, stock_rs, spy_rs
  )


# --- DEDICATED SHORT SCREENER ---
def screen_short_ticker(ticker, df, spy_metrics):
  df = df.copy()
  df["EMA10"] = df["close"].ewm(span=10, adjust=False).mean()
  df["SMA20"] = df["close"].rolling(20).mean()
  latest = df.iloc[-1]

  if pd.isna(latest["EMA10"]) or pd.isna(latest["SMA20"]):
    return None

  close, ema10, sma20 = (
      float(latest["close"]),
      float(latest["EMA10"]),
      float(latest["SMA20"]),
  )
  stock_metrics = calc_rs_metrics(df, mode="Short")
  if not stock_metrics:
    return None

  _, stock_rs, adr_pct, vol_dryup = stock_metrics
  spy_rs = spy_metrics[1]

  if adr_pct < 2.5:
    return None
  if ema10 >= sma20:
    return None
  if close > ema10 and close > sma20:
    return None
  if stock_rs <= spy_rs:
    return None

  return ScreenResult(
      ticker, close, ema10, sma20, adr_pct, vol_dryup, stock_rs, spy_rs
  )


def render_tv_chart(ticker: str, mode="Long"):
  ma_color_1 = "#00BFFF" if mode == "Long" else "#FF4B4B"
  ma_color_2 = "#1E90FF" if mode == "Long" else "#FFA500"
  tv_symbol = ticker.replace("-", ".")

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
                {{ "id": "MAExp@tv-basicstudies", "inputs": {{ "length": 10 }} }},
                {{ "id": "MASimple@tv-basicstudies", "inputs": {{ "length": 20 }} }}
            ],
            "studies_overrides": {{
                "moving average exponential.moving average.color": "{ma_color_1}",
                "moving average exponential.moving average.linewidth": 2,
                "moving average.moving average.color": "{ma_color_2}",
                "moving average.moving average.linewidth": 2
            }}
        }});
        </script>
    </div>
    """
  components.html(widget_html, height=360)


# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="Asymmetric Multi-Directional Screener", layout="wide")

st.sidebar.header("🧭 Navigation & Strategy")
strategy_mode = st.sidebar.radio(
    "Select Trading Bias Mode:",
    [
        "🚀 Mid Cap Long Setup",
        "💥 Mid Cap Short Setup",
        "📈 Large Cap Long Setup",
        "📉 Large Cap Short Setup",
    ],
)

cap_clean = "Mid" if "Mid Cap" in strategy_mode else "Large"
mode_clean = "Long" if "Long Setup" in strategy_mode else "Short"

st.title(
    f"{strategy_mode} ({'Asymmetric Downside Risk Long' if mode_clean == 'Long' else 'Asymmetric Upside Risk Short'})"
)

st.markdown(
    """
    <style>
    [data-testid="stMetricLabel"] { font-size: 0.8rem !important; white-space: normal !important; }
    [data-testid="stMetricValue"] { font-size: 1.1rem !important; white-space: normal !important; }
    [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }
    [data-testid="stMetric"] { 
        background-color: #1E1E1E; border: 1px solid #333333; padding: 6px 10px; border-radius: 6px; overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "spy_data" not in st.session_state or st.session_state.spy_data is None:
  st.session_state.spy_data = fetch_daily_ohlcv("SPY")

tab_home, tab_screener, tab_entries = st.tabs([
    "📊 Sector Performance Tracker",
    "🔍 Live Multi-Timeframe RS Screener",
    "🎯 Optimal Entry Risk Profile",
])

# --- TAB 1: QUANT SECTOR OVERVIEW ---
with tab_home:
  st.header(
      f"📊 Asymmetric Sector RS Tracker ({'Long Leader' if mode_clean == 'Long' else 'Short Deficit'})"
  )
  st.write(
      "Sectors and high-beta industry ETFs ranked against SPY using"
      " asymmetric risk metrics."
  )

  if st.session_state.spy_data is None:
    st.error("Failed to fetch SPY benchmark data.")
  else:
    cache_key = f"sector_df_{mode_clean}"
    if (
        st.button("🔄 Refresh Sector Metrics")
        or cache_key not in st.session_state
    ):
      with st.spinner("Re-calculating asymmetric sector math..."):
        st.session_state[cache_key] = get_quant_sector_strength(
            st.session_state.spy_data, mode=mode_clean
        )

    sectors_df = st.session_state.get(cache_key)

    if sectors_df is not None and not sectors_df.empty:
      prime_sector = sectors_df.iloc[0]
      if mode_clean == "Long":
        st.success(
            f"🔥 **Leading Market Sector:** {prime_sector['Sector Name']}"
            f" ({prime_sector['ETF']}) with Asymmetric RS Edge of"
            f" **{prime_sector['RS Factor Edge']:+.2f}**"
        )
      else:
        st.error(
            f"⚠️ **Weakest Market Sector:** {prime_sector['Sector Name']}"
            f" ({prime_sector['ETF']}) with Asymmetric RS Score of"
            f" **{prime_sector['RS Factor Edge']:+.2f}**"
        )

      st.dataframe(sectors_df, hide_index=True, use_container_width=True)

# --- SIDEBAR CONTROL SCAN ACTION ---
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Execute Scan Engine")
run_scan = st.sidebar.button(
    f"🚀 Execute Selected Scan Setup", type="primary", use_container_width=True
)

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

    status_text.text("Calculating Asymmetric Multi-Timeframe SPY Baseline...")
    spy_metrics = calc_rs_metrics(st.session_state.spy_data, mode=mode_clean)

    tickers = get_us_tickers(cap_size=cap_clean)
    results = []

    status_text.text(
        f"Screening {len(tickers)} symbols for {cap_clean} Cap {mode_clean}"
        " parameters..."
    )

    with ThreadPoolExecutor(max_workers=10) as pool:
      futures = {pool.submit(fetch_daily_ohlcv, t): t for t in tickers}
      for i, future in enumerate(as_completed(futures)):
        ticker = futures[future]
        progress_bar.progress((i + 1) / len(tickers))
        try:
          df = future.result()
          if df is not None:
            if mode_clean == "Long":
              hit = screen_long_ticker(ticker, df, spy_metrics)
            else:
              hit = screen_short_ticker(ticker, df, spy_metrics)
            if hit:
              results.append(hit)
        except Exception:
          pass

    status_text.text("Completed asymmetric scan successfully!")
    progress_bar.empty()
    st.session_state[scan_key] = results


def change_page(state_key, delta, max_p):
  st.session_state[state_key] = max(
      1, min(st.session_state[state_key] + delta, max_p)
  )


def set_page_direct(state_key, input_key):
  st.session_state[state_key] = st.session_state[input_key]


def display_paginated_charts(
    results_list, key_prefix="screener", run_mode="Long", run_cap="Mid"
):
  if not results_list:
    st.info("No tickers matched the current scan criteria.")
    return

  st.success(f"Found {len(results_list)} matching {run_cap} Cap candidates!")

  CHARTS_PER_PAGE = 4
  total_pages = max(1, (len(results_list) - 1) // CHARTS_PER_PAGE + 1)

  page_state_key = (
      f"page_num_{key_prefix}_{run_cap.lower()}_{run_mode.lower()}"
  )
  if page_state_key not in st.session_state:
    st.session_state[page_state_key] = 1

  if st.session_state[page_state_key] > total_pages:
    st.session_state[page_state_key] = total_pages

  current_page = st.session_state[page_state_key]

  st.markdown("---")
  col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
  with col_p1:
    st.button(
        "⬅️ Prev",
        key=f"top_prev_{key_prefix}_{run_cap}_{run_mode}",
        use_container_width=True,
        on_click=change_page,
        args=(page_state_key, -1, total_pages),
    )
  with col_p2:
    top_num_key = f"top_num_{key_prefix}_{run_cap}_{run_mode}"
    st.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        value=int(current_page),
        step=1,
        key=top_num_key,
        label_visibility="collapsed",
        on_change=set_page_direct,
        args=(page_state_key, top_num_key),
    )
  with col_p3:
    st.button(
        "Next ➡️",
        key=f"top_next_{key_prefix}_{run_cap}_{run_mode}",
        use_container_width=True,
        on_click=change_page,
        args=(page_state_key, 1, total_pages),
    )

  start_idx = (current_page - 1) * CHARTS_PER_PAGE
  end_idx = start_idx + CHARTS_PER_PAGE
  page_results = results_list[start_idx:end_idx]

  st.markdown(
      f"<p style='text-align: center; color: #888; font-size: 0.85rem;'>Page"
      f" <b>{current_page}</b> of <b>{total_pages}</b> &nbsp;|&nbsp; Showing"
      f" matches <b>{start_idx + 1}</b> to"
      f" <b>{min(end_idx, len(results_list))}</b> of {len(results_list)}</p>",
      unsafe_allow_html=True,
  )

  for idx in range(0, len(page_results), 3):
    chunk = page_results[idx : idx + 3]
    cols = st.columns(3)
    for col, r in zip(cols, chunk):
      with col:
        st.markdown(f"### 📊 {r.ticker}")
        m_c1, m_c2, m_c3 = st.columns(3)
        m_c1.metric("Close", f"${r.close:.2f}")
        m_c2.metric("Asymmetric RS Edge", f"{r.factor_edge:+.2f}")
        m_c3.metric("ADR%", f"{r.adr_pct:.2f}%")
        render_tv_chart(r.ticker, mode=run_mode)

  st.markdown("---")
  col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
  with col_b1:
    st.button(
        "⬅️ Prev",
        key=f"bot_prev_{key_prefix}_{run_cap}_{run_mode}",
        use_container_width=True,
        on_click=change_page,
        args=(page_state_key, -1, total_pages),
    )
  with col_b2:
    bot_num_key = f"bot_num_{key_prefix}_{run_cap}_{run_mode}"
    st.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        value=int(current_page),
        step=1,
        key=bot_num_key,
        label_visibility="collapsed",
        on_change=set_page_direct,
        args=(page_state_key, bot_num_key),
    )
  with col_b3:
    st.button(
        "Next ➡️",
        key=f"bot_next_{key_prefix}_{run_cap}_{run_mode}",
        use_container_width=True,
        on_click=change_page,
        args=(page_state_key, 1, total_pages),
    )


# --- TAB 2: ACTIVE SCREENER ---
with tab_screener:
  if st.session_state.get(scan_key) is not None:
    rs_sorted = sorted(
        st.session_state[scan_key], key=lambda r: r.factor_edge, reverse=True
    )
    display_paginated_charts(
        rs_sorted, key_prefix="rs", run_mode=mode_clean, run_cap=cap_clean
    )
  else:
    st.info(
        "👈 Please click **'Execute Selected Scan Setup'** in the sidebar to"
        " run the momentum screener."
    )

# --- TAB 3: OPTIMAL ENTRIES ---
with tab_entries:
  if st.session_state.get(scan_key) is not None:
    entry_sorted = sorted(
        st.session_state[scan_key], key=lambda r: r.entry_risk_sma20
    )
    display_paginated_charts(
        entry_sorted, key_prefix="entry", run_mode=mode_clean, run_cap=cap_clean
    )
  else:
    st.info(
        "👈 Please click **'Execute Selected Scan Setup'** in the sidebar to"
        " load optimal entries."
    )
