import websockets
import asyncio
import json
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import joblib
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

# Global container for storing ticks
tick_data = []
ORDER_SIZES = [10, 20, 30, 50, 75, 100]
# Parameters
TICK_LIMIT = 350
#PAIR = "SOL-USDT-SWAP"
#WSS_URL = f"wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/{PAIR}"

# Collect ticks from WebSocket
async def collect_ticks(WSS_URL, on_tick=None):
    async with websockets.connect(WSS_URL) as websocket:
        print("[INFO] Connected to WebSocket")

        while True:
            msg = await websocket.recv()
            tick = json.loads(msg)

            timestamp = time.time()
            asks = tick.get("asks", [])
            bids = tick.get("bids", [])

            if not asks or not bids:
                continue

            best_ask = float(asks[0][0])
            best_bid = float(bids[0][0])
            mid_price = (best_ask + best_bid) / 2
            spread = best_ask - best_bid
            depth = float(asks[0][1]) + float(bids[0][1])
            volume = sum(float(a[1]) for a in asks[:5]) + sum(float(b[1]) for b in bids[:5])

            tick_data.append({
                "timestamp": timestamp,
                "mid": mid_price,
                "spread": spread,
                "depth": depth,
                "volume": volume
            })

            if on_tick:
                on_tick(len(tick_data))  # Send live count to UI

            if len(tick_data) >= TICK_LIMIT:
                print("[INFO] Tick limit reached. Saving data and training models...")
                break

        df = pd.DataFrame(tick_data)
        df.to_csv("ticks.csv", index=False)
        train_slippage_model(df)
        estimate_almgren_chriss(df)
        train_maker_taker_model(df)


# Train basic regression model for slippage
def train_slippage_model(df):
    df = df.copy()
    df["return"] = np.log(df["mid"] / df["mid"].shift(1))
    df["volatility"] = df["return"].rolling(window=20).std()
    df.dropna(inplace=True)

    df["exec_price"] = df["mid"] + np.random.normal(0, 0.5, len(df))
    df["slippage"] = df["exec_price"] - df["mid"]

    features = df[["spread", "depth", "volatility"]]
    target = df["slippage"]

    model = LinearRegression()
    model.fit(features, target)

    print("[INFO] Slippage model trained. Coefficients:", model.coef_)
    joblib.dump(model, "slippage_model.pkl")
    print("[INFO] Slippage model saved as slippage_model.pkl")

# Estimate Almgren-Chriss parameters (eta, gamma)
def estimate_almgren_chriss(df):
    df["return"] = np.log(df["mid"] / df["mid"].shift(1))
    df["volatility"] = df["return"].rolling(window=20).std()
    df.dropna(inplace=True)

    market_vol = df["volume"].mean()
    simulated = []

    for i in range(len(df)):
        row = df.iloc[i]
        mid = row["mid"]
        vol = row["volume"] if row["volume"] > 0 else 1.0

        for Q in ORDER_SIZES:
            temp_impact = 0.01 * (Q / vol)
            exec_price = mid + temp_impact
            slippage = exec_price - mid

            simulated.append({
                "order_size": Q,
                "slippage": slippage,
                "vol": vol
            })

    sim_df = pd.DataFrame(simulated)
    sim_df["Q_by_vol"] = sim_df["order_size"] / sim_df["vol"]
    sim_df["Q2_by_vol"] = (sim_df["order_size"] ** 2) / sim_df["vol"]

    X = sim_df[["Q_by_vol", "Q2_by_vol"]]
    y = sim_df["slippage"]

    model = LinearRegression()
    model.fit(X, y)

    eta = model.coef_[0]
    gamma = model.coef_[1]

    print(f"[RESULT] Almgren-Chriss - η (eta): {eta:.6f}, γ (gamma): {gamma:.6f}")
    joblib.dump(model, "almgren_model.pkl")
    print("[INFO] Almgren-Chriss model saved as almgren_model.pkl")

# Train logistic regression to predict maker vs taker likelihood

def train_maker_taker_model(df):
    df = df.copy()
    df["return"] = np.log(df["mid"] / df["mid"].shift(1))
    df["volatility"] = df["return"].rolling(window=20).std()
    df.dropna(inplace=True)

    # Label based on normalized spread percentile
    df["spread_rank"] = df["spread"].rank(pct=True)
    df["is_taker"] = (df["spread_rank"] < 0.3).astype(int)  # Lower spread → more likely a taker

    if df["is_taker"].nunique() < 2:
        print("[WARN] Not enough class diversity. Skipping model training.")
        return

    # Balance the dataset to avoid class imbalance
    df_taker = df[df["is_taker"] == 1]
    df_maker = df[df["is_taker"] == 0]

    if len(df_taker) < len(df_maker):
        df_maker = resample(df_maker, replace=False, n_samples=len(df_taker), random_state=42)
    else:
        df_taker = resample(df_taker, replace=False, n_samples=len(df_maker), random_state=42)

    df_balanced = pd.concat([df_taker, df_maker])

    features = df_balanced[["spread", "depth", "volatility"]]
    target = df_balanced["is_taker"]

    # Feature scaling for better convergence
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    model = LogisticRegression()
    model.fit(features_scaled, target)
    joblib.dump(model, "maker_taker_model.pkl")
    joblib.dump(scaler, "maker_taker_scaler.pkl")
    print("[INFO] Maker/Taker model trained and saved.")



