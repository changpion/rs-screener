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
MAX_TICKERS_LIMIT = 700  # Safe cap per cap-size bucket

# Sector & Industry Bellwethers Map
SECTOR_MAP = {
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
      pool_connections=25, pool_maxsize=25, max_retries=retry_strategy
  )
  session.mount("https://", adapter)
  session.mount("http://", adapter)
  return session


# --- DATACLASSES ---
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


@dataclass
class EPResult:
  ticker: str
  close: float
  gap_pct: float
  rvol: float
  base_range_pct: float
  adr_pct: float

  @property
  def factor_edge(self) -> float:
    return self.gap_pct

  @property
  def entry_risk_sma20(self) -> float:
    return self.base_range_pct


# --- QUANT MATH & METRICS ---
def calc_rs_metrics(df, mode="Long"):
  closes = df["close"]
  if len(closes) < MIN_BARS:
    return None

  ret_20 = (closes.iloc[-1] / closes.iloc[-20] - 1) * 100
  ret_60 = (closes.iloc[-1] / closes.iloc[-60] - 1) * 100
  ret_120 = (closes.iloc[-1] / closes.iloc[-120] - 1) * 100
  composite_return = (0.4 * ret_20) + (0.4 * ret_60) + (0.2 * ret_120)

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

  if "high" in df.columns and "low" in df.columns:
    daily_range = (
        (df["high"].iloc[-20:] - df["low"].iloc[-20:]) / df["low"].iloc[-20:]
    ) * 100
    adr_pct = float(daily_range.mean())
  else:
    adr_pct = float(risk_metric * 1.5)

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


# --- MARKET HEALTH TRAFFIC LIGHT (SUGGESTION 3) ---
def get_market_health_status(spy_df, qqq_df):
  if spy_df is None or qqq_df is None:
    return "UNKNOWN", "#888888", "Data Unavailable"

  def eval_index(df):
    closes = df["close"]
    ema10 = closes.ewm(span=10, adjust=False).mean().iloc[-1]
    sma20 = closes.rolling(20).mean().iloc[-1]
    last_close = closes.iloc[-1]
    if last_close > ema10 and ema10 > sma20:
      return 2  # Full Bull
    elif last_close > sma20:
      return 1  # Moderate/Pullback
    return 0  # Bearish/Defense

  spy_score = eval_index(spy_df)
  qqq_score = eval_index(qqq_df)
  total = spy_score + qqq_score

  if total >= 3:
    return (
        "🟢 GREEN: Aggressive Long Environment",
        "#00C853",
        "SPY & QQQ > 10 EMA > 20 SMA. Full position sizing enabled.",
    )
  elif total >= 1:
    return (
        "🟡 YELLOW: Selective / Pullback Tape",
        "#FFD600",
        "Indices pulling back near support. Reduce sizing (0.25%-0.5% risk).",
    )
  else:
    return (
        "🔴 RED: Defensive / High Risk",
        "#D50000",
        "Indices below key MAs. Avoid breakout long entries or preserve cash.",
    )


# --- TICKER LIST PARSER ---
@st.cache_data(ttl=86400)
def get_us_tickers(cap_size="Mid"):
  session = get_http_session()
  try:
    if cap_size == "Small":
      url = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
    elif cap_size == "Mid":
      url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    elif cap_size == "Large":
      url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    else:  # Combined S&P 1500
      t_large = get_us_tickers("Large")
      t_mid = get_us_tickers("Mid")
      t_small = get_us_tickers("Small")
      return sorted(list(set(t_large + t_mid + t_small)))

    resp = session.get(url, timeout=15)
    resp.raise_for_status()

    dfs = pd.read_html(StringIO(resp.text), match="Symbol")
    df = dfs[0]

    ticker_col = next(
        (
            col
            for col in df.columns
            if "symbol" in str(col).lower() or "ticker" in str(col).lower()
        ),
        None,
    )

    if ticker_col is None:
      raise ValueError("Ticker column not identified.")

    tickers = (
        df[ticker_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.replace(".", "-", regex=False)
        .tolist()
    )

    return sorted(list(set(tickers)))[:MAX_TICKERS_LIMIT]

  except Exception as e:
    st.warning(
        f"⚠️ Warning: Live ticker fetch for {cap_size} Cap failed ({e})."
        " Using default fallbacks."
    )
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
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
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


# --- SCREENER ENGINES ---
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


def screen_ep_ticker(ticker, df):
  if df is None or len(df) < 65:
    return None

  prev_close = float(df["close"].iloc[-2])
  today_open = float(
      df["open"].iloc[-1] if not pd.isna(df["open"].iloc[-1]) else prev_close
  )
  today_close = float(df["close"].iloc[-1])
  today_vol = float(df["volume"].iloc[-1])

  gap_pct = ((today_open - prev_close) / prev_close) * 100
  if gap_pct < 8.0:
    return None

  vol_20_sma = df["volume"].iloc[-21:-1].mean()
  if vol_20_sma <= 0:
    return None

  rvol = today_vol / vol_20_sma
  if rvol < 2.5:
    return None

  prior_60 = df.iloc[-61:-1]
  base_high = prior_60["high"].max()
  base_low = prior_60["low"].min()
  base_range_pct = (
      ((base_high - base_low) / base_low) * 100 if base_low > 0 else 0.0
  )

  daily_range = (
      (df["high"].iloc[-20:] - df["low"].iloc[-20:]) / df["low"].iloc[-20:]
  ) * 100
  adr_pct = float(daily_range.mean())

  return EPResult(
      ticker=ticker,
      close=today_close,
      gap_pct=gap_pct,
      rvol=rvol,
      base_range_pct=base_range_pct,
      adr_pct=adr_pct,
  )


def render_tv_chart(ticker: str, mode="Long"):
  ma_color_1 = "#00BFFF" if mode != "Short" else "#FF4B4B"
  ma_color_2 = "#1E90FF" if mode != "Short" else "#FFA500"
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
st.set_page_config(
    page_title="Asymmetric Momentum & EP Screener", layout="wide"
)

# Fetch Market Health Baseline Data
if "spy_data" not in st.session_state or st.session_state.spy_data is None:
  st.session_state.spy_data = fetch_daily_ohlcv("SPY")
if "qqq_data" not in st.session_state or st.session_state.qqq_data is None:
  st.session_state.qqq_data = fetch_daily_ohlcv("QQQ")

# --- SIDEBAR: MARKET HEALTH TRAFFIC LIGHT ---
st.sidebar.header("🚥 Market Health Status")
status_title, status_color, status_desc = get_market_health_status(
    st.session_state.spy_data, st.session_state.qqq_data
)
st.sidebar.markdown(
    f"""
    <div style="background-color: #1E1E1E; border: 1.5px solid {status_color}; padding: 10px; border-radius: 8px; margin-bottom: 15px;">
        <h4 style="color: {status_color}; margin: 0 0 5px 0; font-size: 0.95rem;">{status_title}</h4>
        <p style="color: #CCCCCC; margin: 0; font-size: 0.8rem;">{status_desc}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("🧭 Navigation & Strategy")
strategy_mode = st.sidebar.radio(
    "Select Trading Bias Mode:",
    [
        "🔥 Episodic Pivot (EP) Gap Setup",
        "🚀 Mid Cap Long Setup",
        "💥 Mid Cap Short Setup",
        "📈 Large Cap Long Setup",
        "📉 Large Cap Short Setup",
        "🔬 Small Cap Long Setup",
        "⚡ Small Cap Short Setup",
    ],
)

if "EP" in strategy_mode:
  cap_clean = "Combined S&P 1500"
  mode_clean = "EP"
elif "Mid Cap" in strategy_mode:
  cap_clean = "Mid"
  mode_clean = "Long" if "Long" in strategy_mode else "Short"
elif "Small Cap" in strategy_mode:
  cap_clean = "Small"
  mode_clean = "Long" if "Long" in strategy_mode else "Short"
else:
  cap_clean = "Large"
  mode_clean = "Long" if "Long" in strategy_mode else "Short"

st.title(
    f"{strategy_mode} ({'Qullamaggie Institutional Gap' if mode_clean == 'EP' else ('Downside Risk Long' if mode_clean == 'Long' else 'Upside Risk Short')})"
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

# --- SIDEBAR: POSITION SIZING & ADR RISK CALCULATOR (SUGGESTION 1) ---
st.sidebar.markdown("---")
with st.sidebar.expander("🧮 Position Sizer & ADR Risk Calculator"):
  acc_size = st.number_input(
      "Account Capital ($)", value=100000, step=5000, key="calc_acc"
  )
  risk_pct = st.number_input(
      "Account Risk %",
      value=0.5,
      step=0.1,
      min_value=0.1,
      max_value=3.0,
      key="calc_risk",
  )
  entry_p = st.number_input(
      "Planned Entry Price ($)", value=100.0, step=1.0, key="calc_entry"
  )
  stop_p = st.number_input(
      "Stop Loss Price ($)", value=95.0, step=1.0, key="calc_stop"
  )
  est_adr = st.number_input(
      "Stock ADR %", value=4.5, step=0.5, key="calc_adr"
  )

  risk_dollars = acc_size * (risk_pct / 100.0)
  per_share_risk = abs(entry_p - stop_p)

  if per_share_risk > 0:
    shares = int(risk_dollars / per_share_risk)
    total_position = shares * entry_p
    stop_dist_pct = (per_share_risk / entry_p) * 100.0

    st.markdown("---")
    st.markdown(f"**Shares to Buy:** `{shares:,}` shares")
    st.markdown(f"**Total Position:** `${total_position:,.2f}`")
    st.markdown(f"**Total Dollar Risk:** `${risk_dollars:,.2f}`")
    st.markdown(f"**Stop Distance:** `{stop_dist_pct:.2f}%`")

    # ADR Compliance Safeguard Check
    if stop_dist_pct > est_adr:
      st.error(
          f"⚠️ **Wide Stop Warning:** Stop distance ({stop_dist_pct:.2f}%) is"
          f" larger than 1.0x ADR ({est_adr:.2f}%). Consider a tighter low-of-day stop."
      )
    else:
      st.success("✅ Stop distance compliant with ADR volatility limits.")
  else:
    st.info("Entry and Stop price must be different.")

tab_home, tab_screener, tab_entries = st.tabs([
    "📊 Sector Performance Tracker",
    "🔍 Live Multi-Timeframe RS / EP Screener",
    "🎯 Optimal Entry / Tight Base Profile",
])

# --- TAB 1: QUANT SECTOR OVERVIEW ---
with tab_home:
  st.header(
      f"📊 Asymmetric Sector RS Tracker ({'Long Leader' if mode_clean != 'Short' else 'Short Deficit'})"
  )
  st.write(
      "Sectors and high-beta industry ETFs ranked against SPY using"
      " asymmetric risk metrics."
  )

  if st.session_state.spy_data is None:
    st.error("Failed to fetch SPY benchmark data.")
  else:
    cache_key = f"sector_df_{'Long' if mode_clean != 'Short' else 'Short'}"
    if (
        st.button("🔄 Refresh Sector Metrics")
        or cache_key not in st.session_state
    ):
      with st.spinner("Re-calculating sector metrics..."):
        st.session_state[cache_key] = get_quant_sector_strength(
            st.session_state.spy_data,
            mode=("Long" if mode_clean != "Short" else "Short"),
        )

    sectors_df = st.session_state.get(cache_key)
    if sectors_df is not None and not sectors_df.empty:
      prime_sector = sectors_df.iloc[0]
      st.success(
          f"🔥 **Top Sector Focus:** {prime_sector['Sector Name']}"
          f" ({prime_sector['ETF']}) with RS Edge of"
          f" **{prime_sector['RS Factor Edge']:+.2f}**"
      )
      st.dataframe(sectors_df, hide_index=True, use_container_width=True)

# --- SIDEBAR CONTROL SCAN ACTION ---
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Execute Scan Engine")
run_scan = st.sidebar.button(
    f"🚀 Execute Selected Scan Setup", type="primary", use_container_width=True
)

scan_key = f"scan_results_{cap_clean.replace(' ', '_').lower()}_{mode_clean.lower()}"
if scan_key not in
