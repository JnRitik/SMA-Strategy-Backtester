import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import vectorbt as vbt
from datetime import date

# ----------------------------
# Cache the data
# ----------------------------
@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    df = df[['Close']]
    return df

# ----------------------------
# SMA Backtest
# ----------------------------
def sma_backtest(df, fast_window, slow_window, fees, slippage):
    fast_ma = vbt.MA.run(df['Close'], fast_window)
    slow_ma = vbt.MA.run(df['Close'], slow_window)

    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)

    pf = vbt.Portfolio.from_signals(
        close=df['Close'],
        entries=entries,
        exits=exits,
        size=1,
        fees=fees,
        slippage=slippage
    )
    return fast_ma, slow_ma, entries, exits, pf

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="SMA Crossover Backtest", layout="wide")
st.title("📈 SMA Crossover Backtesting with VectorBT")

# Sidebar Inputs
st.sidebar.header("Settings")
symbol = st.sidebar.text_input("Stock Symbol", "TCS.NS")
start_date = st.sidebar.date_input("Start Date", date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", date(2024, 1, 1))
fast_window = st.sidebar.number_input("Fast SMA Window", min_value=1, value=10)
slow_window = st.sidebar.number_input("Slow SMA Window", min_value=2, value=30)
fees = st.sidebar.number_input("Fees (%) per trade", min_value=0.0, value=0.1) / 100
slippage = st.sidebar.number_input("Slippage (%)", min_value=0.0, value=0.05) / 100

# Button for backtest
if st.sidebar.button("🚀 Run Backtest"):
    df = load_data(symbol, start_date, end_date)

    if df.empty:
        st.error("No data found for this symbol/date range.")
    elif fast_window >= slow_window:
        st.error("Fast SMA window must be smaller than Slow SMA window.")
    else:
        fast_ma, slow_ma, entries, exits, pf = sma_backtest(df, fast_window, slow_window, fees, slippage)

        # Price + SMA Plot
        price_fig = df['Close'].vbt.plot(trace_kwargs=dict(name="Close"))
        fast_ma.ma.vbt.plot(trace_kwargs=dict(name=f"SMA {fast_window}"), fig=price_fig)
        slow_ma.ma.vbt.plot(trace_kwargs=dict(name=f"SMA {slow_window}"), fig=price_fig)

        # Buy markers
        price_fig.add_scatter(
            x=df.index[entries.values.flatten()],
            y=df['Close'][entries.values.flatten()],
            mode='markers',
            marker=dict(symbol='triangle-up', color='green', size=10),
            name='Buy'
        )
        # Sell markers
        price_fig.add_scatter(
            x=df.index[exits.values.flatten()],
            y=df['Close'][exits.values.flatten()],
            mode='markers',
            marker=dict(symbol='triangle-down', color='red', size=10),
            name='Sell'
        )

        st.subheader("📊 Price Chart with SMA Signals")
        st.plotly_chart(price_fig, use_container_width=True)

        # Portfolio Equity Curve
        st.subheader("📈 Equity Curve")
        st.plotly_chart(pf.value().vbt.plot(), use_container_width=True)

        # Drawdown
        st.subheader("📉 Drawdown")
        st.plotly_chart(pf.drawdown().vbt.plot(), use_container_width=True)

        # Stats Table
        st.subheader("📑 Key Stats")
        stats = pf.stats()
        st.dataframe(stats)
