
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import vectorbt as vbt
from datetime import date

# ----------------------------
# Page & Title
# ----------------------------
st.set_page_config(page_title="SMA Crossover Backtest", layout="wide")
st.title("📈 SMA Crossover Backtesting with VectorBT")

# ----------------------------
# Cache the data
# ----------------------------
@st.cache_data
def load_data(symbol: str, start, end) -> pd.DataFrame:
    """Download OHLCV and return Close-only DataFrame."""
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["Close"])
    if "Close" not in df.columns:
        return pd.DataFrame(columns=["Close"])
    return df[["Close"]]

# ----------------------------
# Strategy: SMA Crossover
# ----------------------------
def sma_backtest(df: pd.DataFrame, fast_window: int, slow_window: int, fees: float, slippage: float):
    """Run SMA crossover backtest with vectorbt and return all artifacts."""
    fast_ma = vbt.MA.run(df["Close"], fast_window)
    slow_ma = vbt.MA.run(df["Close"], slow_window)

    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)

    pf = vbt.Portfolio.from_signals(
        close=df["Close"],
        entries=entries,
        exits=exits,
        size=1.0,
        fees=fees,
        slippage=slippage,
        init_cash=100_000.0,
        cash_sharing=True,
        freq="D",
    )
    return fast_ma, slow_ma, entries, exits, pf

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("Settings")
symbol = st.sidebar.text_input("Stock Symbol", "TCS.NS")
start_date = st.sidebar.date_input("Start Date", date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", date(2024, 1, 1))
fast_window = st.sidebar.number_input("Fast SMA Window", min_value=1, value=10, step=1)
slow_window = st.sidebar.number_input("Slow SMA Window", min_value=2, value=30, step=1)
fees_pct = st.sidebar.number_input("Fees (%) per trade", min_value=0.0, value=0.10, step=0.01)
slip_pct = st.sidebar.number_input("Slippage (%)", min_value=0.0, value=0.05, step=0.01)
fees = fees_pct / 100.0
slippage = slip_pct / 100.0

# ----------------------------
# Run Backtest
# ----------------------------
if st.sidebar.button("🚀 Run Backtest"):
    df = load_data(symbol, start_date, end_date)

    if df.empty:
        st.error("No data found for this symbol/date range.")
        st.stop()

    if fast_window >= slow_window:
        st.error("Fast SMA window must be smaller than Slow SMA window.")
        st.stop()

    fast_ma, slow_ma, entries, exits, pf = sma_backtest(df, fast_window, slow_window, fees, slippage)

    # ----------------------------
    # Price + SMA Plot with buy/sell markers
    # ----------------------------
    price_fig = df["Close"].vbt.plot(trace_kwargs=dict(name="Close"))
    fast_ma.ma.vbt.plot(trace_kwargs=dict(name=f"SMA {fast_window}"), fig=price_fig)
    slow_ma.ma.vbt.plot(trace_kwargs=dict(name=f"SMA {slow_window}"), fig=price_fig)

    ent_mask = entries.values.flatten().astype(bool)
    ex_mask = exits.values.flatten().astype(bool)

    price_fig.add_scatter(
        x=df.index[ent_mask],
        y=df["Close"][ent_mask],
        mode="markers",
        marker=dict(symbol="triangle-up", size=10),
        name="Buy"
    )
    price_fig.add_scatter(
        x=df.index[ex_mask],
        y=df["Close"][ex_mask],
        mode="markers",
        marker=dict(symbol="triangle-down", size=10),
        name="Sell"
    )

    st.subheader("📊 Price Chart with SMA Signals")
    st.plotly_chart(price_fig, use_container_width=True)

    # ----------------------------
    # Equity Curve
    # ----------------------------
    st.subheader("📈 Equity Curve")
    st.plotly_chart(pf.value().vbt.plot(), use_container_width=True)

    # ----------------------------
    # Drawdown
    # ----------------------------
    st.subheader("📉 Drawdown")
    st.plotly_chart(pf.drawdown().vbt.plot(), use_container_width=True)

    # ----------------------------
    # Stats Table
    # ----------------------------
    st.subheader("📑 Key Stats")
    stats = pf.stats()
    if isinstance(stats, pd.Series):
        st.dataframe(stats.to_frame("Value"))
    else:
        st.dataframe(stats)

    # ----------------------------
    # Trade Log (robust)
    # ----------------------------
    st.subheader("📒 Trade Log")

    # Always prefer the raw records for consistent column names
    trades = getattr(pf.trades, "records", None)
    # Fallback: some versions only expose records_readable
    if (trades is None or len(trades) == 0) and hasattr(pf.trades, "records_readable"):
        trades = pf.trades.records_readable

    if trades is None or len(trades) == 0:
        st.info("No trades were generated for this configuration.")
    else:
        tr = trades.copy()

        # Helper to find best-matching column
        cols_lower = {c.lower(): c for c in tr.columns}
        def pick(*names):
            for n in names:
                if n in cols_lower:
                    return cols_lower[n]
            return None

        # Expect these standard names in raw records; provide common fallbacks
        entry_idx_col   = pick("entry_idx", "start_idx", "entry index", "start index")
        exit_idx_col    = pick("exit_idx", "end_idx", "exit index", "end index")
        entry_price_col = pick("entry_price", "start_price", "open_price")
        exit_price_col  = pick("exit_price", "end_price", "close_price")
        size_col        = pick("size", "qty", "quantity")
        pnl_col         = pick("pnl", "profit")
        ret_col         = pick("return", "ret", "return_pct", "return %")

        # Compute dates from index positions (if available)
        if entry_idx_col is not None and entry_idx_col in tr.columns:
            tr["Entry Date"] = tr[entry_idx_col].map(lambda i: df.index[int(i)] if pd.notna(i) else pd.NaT)
        else:
            tr["Entry Date"] = pd.NaT

        if exit_idx_col is not None and exit_idx_col in tr.columns:
            tr["Exit Date"] = tr[exit_idx_col].map(lambda i: df.index[int(i)] if pd.notna(i) else pd.NaT)
        else:
            tr["Exit Date"] = pd.NaT

        # Compute convenience columns
        tr["Entry Price"] = tr[entry_price_col].round(4) if entry_price_col in tr.columns else np.nan
        tr["Exit Price"]  = tr[exit_price_col].round(4)  if exit_price_col in tr.columns  else np.nan
        tr["Size"]        = tr[size_col]                 if size_col in tr.columns        else np.nan
        tr["PnL"]         = tr[pnl_col].round(2)         if pnl_col in tr.columns         else np.nan

        if ret_col in tr.columns:
            # If return looks like a fraction, express as percent
            if tr[ret_col].dropna().abs().max() <= 1.5:
                tr["Return %"] = (tr[ret_col] * 100).round(2)
            else:
                tr["Return %"] = tr[ret_col].round(2)
        else:
            tr["Return %"] = np.nan

        tr["Status"] = np.where(tr["Exit Date"].isna(), "Open", "Closed")
        tr["Duration"] = (tr["Exit Date"] - tr["Entry Date"]).dt.days

        # Compact view
        show_cols = ["Status", "Entry Date", "Exit Date", "Duration",
                     "Size", "Entry Price", "Exit Price", "PnL", "Return %"]
        show_cols = [c for c in show_cols if c in tr.columns]

        trade_log = tr[show_cols].sort_values(by="Entry Date").reset_index(drop=True)

        st.dataframe(trade_log, use_container_width=True)
        st.download_button(
            "⬇️ Download Trade Log (CSV)",
            data=trade_log.to_csv(index=False).encode("utf-8"),
            file_name=f"trade_log_{symbol}.csv",
            mime="text/csv"
        )

        with st.expander("🔧 Debug: Show raw VectorBT trade records"):
            st.write(tr.head())

    # ----------------------------
    # Footer
    # ----------------------------
    with st.expander("ℹ️ Tips"):
        st.markdown(
            "- Fees and slippage are **percent values**. For example, 0.10% fee = 0.10 in the box.\n"
            "- If you see no trades, try different SMA windows or a longer date range.\n"
            "- Use NSE suffix (e.g., `TCS.NS`, `INFY.NS`) for Indian stocks via Yahoo Finance."
        )
else:
    st.info("Set your parameters in the sidebar and click **Run Backtest**.")
