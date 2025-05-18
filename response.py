import websockets
import asyncio
import json
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Async wrapper for Streamlit
def run_prediction_sync(wss_url, maker_fee, user_dollars, ticks=30):
    return asyncio.run(predict_from_realtime(wss_url, maker_fee, user_dollars, ticks))

# Real-time prediction with limited ticks
async def predict_from_realtime(WSS_URL, MAKER_FEE_RATE, USER_DOLLARS, num_ticks=30):
    # Load models inside function
    try:
        slippage_model = joblib.load("slippage_model.pkl")
        almgren_model = joblib.load("almgren_model.pkl")
        maker_taker_model = joblib.load("maker_taker_model.pkl")
        maker_taker_scaler = joblib.load("maker_taker_scaler.pkl")
    except FileNotFoundError as e:
        raise RuntimeError("⚠️ One or more model files not found. Please train first.") from e

    async with websockets.connect(WSS_URL) as websocket:
        print("[INFO] Connected to WebSocket for prediction")

        mid_prices = []
        results = []

        for tick_count in range(num_ticks):
            start_time = time.perf_counter()  # ⏱ Start timing

            msg = await websocket.recv()
            tick = json.loads(msg)

            end_time = time.perf_counter()  # ⏱ End timing
            latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds

            timestamp = pd.Timestamp.now()
            asks = tick.get("asks", [])
            bids = tick.get("bids", [])

            if not asks or not bids:
                continue

            best_ask = float(asks[0][0])
            best_bid = float(bids[0][0])
            mid = (best_ask + best_bid) / 2
            spread = best_ask - best_bid
            depth = float(asks[0][1]) + float(bids[0][1])
            volume = sum(float(a[1]) for a in asks[:5]) + sum(float(b[1]) for b in bids[:5])

            mid_prices.append(mid)
            if len(mid_prices) > 20:
                mid_prices.pop(0)

            if len(mid_prices) >= 2:
                returns = np.log(np.array(mid_prices[1:]) / np.array(mid_prices[:-1]))
                volatility = np.std(returns)
            else:
                volatility = 0.0

            features = np.array([[spread, depth, volatility]])
            features_scaled = maker_taker_scaler.transform(features)

            pred_slippage = slippage_model.predict(features)[0]
            Q = USER_DOLLARS / mid
            Q_by_vol = Q / volume if volume > 0 else 0.0
            Q2_by_vol = (Q ** 2) / volume if volume > 0 else 0.0
            market_impact = almgren_model.predict([[Q_by_vol, Q2_by_vol]])[0]
            prob_taker = maker_taker_model.predict_proba(features_scaled)[0][1]

            impact_cost = USER_DOLLARS * market_impact
            gross_cost = abs(pred_slippage) + impact_cost
            fee = USER_DOLLARS * MAKER_FEE_RATE
            net_cost = gross_cost + fee

            results.append({
                "timestamp": timestamp,
                "order_size": USER_DOLLARS,
                "mid_price": mid,
                "slippage": pred_slippage,
                "fee": fee,
                "market_impact": market_impact,
                "net_cost": net_cost,
                "taker_probability": prob_taker,
                "latency": latency_ms  # ⏱ Use real measured latency
            })

        df = pd.DataFrame(results)
        return df
