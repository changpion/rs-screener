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
  sector: str = "Unmapped"
  sector_rank: int = 99

  @property
  def factor_edge(self) -> float:
    return self.composite_rs - self.spy_composite_rs

  @property
  def entry_risk_sma20(self) -> float:
    return abs(((self.close - self.sma20) / self.sma20) * 100)


@dataclass
class EPResult:
  ticker: str
  close: float
  gap_pct: float
  rvol: float
  base_range_pct: float
  adr_pct: float
  sector: str = "Unmapped"
  sector_rank: int = 99

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


# --- MARKET HEALTH TRAFFIC LIGHT ---
def get_market_health_status(spy_df, qqq_df):
  if spy_df is None or qqq_df is None:
    return "UNKNOWN", "#888888", "Data Unavailable"

  def eval_index(df):
    closes = df["close"]
    ema10 = closes.ewm(span=10, adjust=False).mean().iloc[-1]
    sma20 = closes.rolling(20).mean().iloc[-1]
    last_close = closes.iloc[-1]
    if last_close > ema10 and ema10 > sma20:
      return 2
    elif last_close > sma20:
      return 1
    return 0

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


# --- WIKIPEDIA TICKER & SECTOR PARSER ---
@st.cache_data(ttl=86400)
def get_us_tickers_with_sectors(cap_size="Mid"):
  session = get_http_session()
  sector_dict = {}
  try:
    if cap_size == "Small":
      urls = ["https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"]
    elif cap_size == "Mid":
      urls = ["https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"]
    elif cap_size == "Large":
      urls = ["https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"]
    else:  # Combined S&P 1500
      urls = [
          "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
          "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
          "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
      ]

    for url in urls:
      resp = session.get(url, timeout=15)
      resp.raise_for_status()
      dfs = pd.read_html(StringIO(resp.text), match="Symbol")
      df = dfs[0]

      t_col = next(
          (
              c
              for c in df.columns
              if "symbol" in str(c).lower() or "ticker" in str(c).lower()
          ),
          None,
      )
      s_col = next(
          (c for c in df.columns if "sector" in str(c).lower()), None
      )

      if t_col:
        for _, row in df.iterrows():
          t = str(row[t_col]).strip().replace(".", "-").upper()
          sec = (
              str(row[s_col]).strip()
              if s_col and not pd.isna(row[s_col])
              else "General"
          )
          sector_dict[t] = sec

    if not sector_dict:
      raise ValueError("Empty ticker map")
    return sector_dict
  except Exception as e:
    st.warning(
        f"⚠️ Warning: Live sector/ticker fetch failed ({e}). Using default"
        " fallback universe."
    )
    defaults = [
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
    return {
        t: "Technology"
        if t in ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO"]
        else "General"
        for t in defaults
    }


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
      ticker=ticker,
      close=close,
      ema10=ema10,
      sma20=sma20,
      adr_pct=adr_pct,
      vol_dryup_ratio=vol_dryup,
      composite_rs=stock_rs,
      spy_composite_rs=spy_rs,
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
      ticker=ticker,
      close=close,
      ema10=ema10,
      sma20=sma20,
      adr_pct=adr_pct,
      vol_dryup_ratio=vol_dryup,
      composite_rs=stock_rs,
      spy_composite_rs=spy_rs,
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
  base_high
