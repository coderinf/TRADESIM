import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os
import asyncio
from modeltrainer import collect_ticks
st.set_page_config(page_title="Real-Time Trading Cost Estimator", layout="wide")
st.write("Streaming live ticks from WebSocket...")

tick_display = st.empty()  # Placeholder for tick count
TICK_LIMIT = 350
# Config
DEFAULT_PAIR = "SOL-USDT-SWAP"
OKX_BASE_URL = "wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/"


st.title("💹 Real-Time Trading Cost Simulator")

# State init
if "step" not in st.session_state:
    st.session_state.step = 1

# Step 1: Basic Selection
if st.session_state.step == 1:
    st.sidebar.header("Step 1: Select Asset Parameters")
    exchange = st.sidebar.selectbox("Exchange", ["OKX"])
    asset = st.sidebar.text_input("Trading Pair (e.g. SOL-USDT-SWAP)", DEFAULT_PAIR)
    order_type = st.sidebar.selectbox("Order Type", ["Market"])

    if st.sidebar.button("Next →"):
        st.session_state.exchange = exchange
        st.session_state.asset = asset
        st.session_state.order_type = order_type

        # Compose WebSocket URL
        pair = asset
        wss_url = f"{OKX_BASE_URL}{pair}"
        st.session_state.wss_url = wss_url
        tick_display = st.empty()
        tick_placeholder = st.empty()  # Display tick count


        def streamlit_tick_callback(tick_count):
            tick_placeholder.info(f"📡 Ticks Collected: {tick_count}/350")


        with st.spinner("⏳ Collecting ticks and training models..."):
            try:
                asyncio.run(collect_ticks(wss_url, on_tick=streamlit_tick_callback))
                st.session_state.models_trained = True
                st.session_state.step = 2
                st.success("✅ Training complete!")
            except Exception as e:
                st.error(f"❌ Training failed: {e}")

# Step 2: Trade Simulation
elif st.session_state.step == 2:
    st.sidebar.header("Step 2: Set Trade Conditions")

    user_dollars = st.sidebar.number_input("Order Size (USD)", 100, 100000, 1000, step=100)
    volatility = st.sidebar.number_input("Estimated Volatility (%)", 0.00, 10.0, 1.0, step=0.01)
    fee_tier = st.sidebar.selectbox("Fee Tier", ["Tier 1", "Tier 2", "VIP"])
    maker_fee = st.sidebar.slider("Maker Fee (%)", 0.00, 0.10, 0.01)
    volatility = st.sidebar.number_input("Estimated Volatility (%)", min_value=0.001, max_value=1.0, value=0.01,
                                         step=0.001, format="%.3f")
    if st.sidebar.button("▶️ Simulate"):
        with st.spinner("Running model predictions on real-time data..."):
            from response import run_prediction_sync
            try:
                result_df=run_prediction_sync(st.session_state.wss_url, maker_fee, user_dollars, ticks=30)
                st.session_state.results = result_df
                st.success("✅ Simulation completed")
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

    if "results" in st.session_state:
        result_df = st.session_state.results

        # Output table
        st.subheader("📈 Real-Time Trading Cost Breakdown")
        st.dataframe(result_df)

        # Plot 1: Slippage & Market Impact
        st.subheader("📊 Market Impact vs Order Size")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result_df['order_size'], y=result_df['slippage'], mode='lines+markers', name='Slippage'))
        fig.add_trace(go.Scatter(x=result_df['order_size'], y=result_df['market_impact'], mode='lines+markers', name='Market Impact'))
        fig.update_layout(title="Slippage & Market Impact", xaxis_title="Order Size (USD)", yaxis_title="Cost (USDT)", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # Plot 2: Internal Latency
        if "latency" in result_df.columns:
            st.subheader("⏱ Internal Latency Over Time")
            fig2 = go.Figure(go.Scatter(x=result_df["timestamp"], y=result_df["latency"], mode="lines", name="Latency"))
            fig2.update_layout(xaxis_title="Timestamp", yaxis_title="Latency (ms)", template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)

    if st.sidebar.button("← Back"):
        st.session_state.step = 1
