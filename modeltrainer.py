import websockets
import asyncio
import json
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import joblib

# Constants
tick_data = []
TICK_LIMIT = 350
ORDER_SIZES = [10, 20, 30, 50, 75, 100]
STANDARD_ORDER_SIZE = 20  # Used for slippage simulation


#Simulate VWAP execution for a market buy order
def simulate_execution_price(order_size, asks):
    qty_remaining = order_size
    cost = 0.0
    for price, qty in asks:
        px = float(price)
        q = float(qty)
        fill_qty = min(q, qty_remaining)
        cost += fill_qty * px
        qty_remaining -= fill_qty
        if qty_remaining <= 0:
            break
    if qty_remaining > 0:
        return None  # Not enough liquidity
    return cost / order_size


#  WebSocket tick collection
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
                "volume": volume,
                "asks": json.dumps(asks),
                "bids": json.dumps(bids)
            })

            if on_tick:
                on_tick(len(tick_data))

            if len(tick_data) >= TICK_LIMIT:
                print("[INFO] Tick limit reached. Saving data and training models...")
                break

        df = pd.DataFrame(tick_data)
        df.to_csv("ticks.csv", index=False)
        train_slippage_model(df)
        estimate_almgren_chriss(df)
        train_maker_taker_model(df)


#  Train slippage model using real simulated slippage (VWAP)
def train_slippage_model(df):
    df = df.copy()
    df["asks"] = df["asks"].apply(json.loads)
    df["return"] = np.log(df["mid"] / df["mid"].shift(1))
    df["volatility"] = df["return"].rolling(window=20).std()
    df.dropna(inplace=True)

    # Simulate slippage
    df["exec_price"] = df.apply(
        lambda row: simulate_execution_price(STANDARD_ORDER_SIZE, row["asks"]),
        axis=1
    )
    df.dropna(subset=["exec_price"], inplace=True)
    df["slippage"] = df["exec_price"] - df["mid"]

    features = df[["spread", "depth", "volatility"]]
    target = df["slippage"]

    model = LinearRegression()
    model.fit(features, target)
    joblib.dump(model, "slippage_model.pkl")
    print("[INFO] Slippage model trained and saved.")


#  Estimate Almgren-Chriss model (η, γ)
def estimate_almgren_chriss(df):
    df["return"] = np.log(df["mid"] / df["mid"].shift(1))
    df["volatility"] = df["return"].rolling(window=20).std()
    df.dropna(inplace=True)

    simulated = []
    for _, row in df.iterrows():
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

    joblib.dump(model, "almgren_model.pkl")
    print(f"[RESULT] Almgren-Chriss model saved. η: {model.coef_[0]:.6f}, γ: {model.coef_[1]:.6f}")


#  Train logistic regression for taker probability
def train_maker_taker_model(df):
    df["return"] = np.log(df["mid"] / df["mid"].shift(1))
    df["volatility"] = df["return"].rolling(window=20).std()
    df.dropna(inplace=True)

    df["spread_rank"] = df["spread"].rank(pct=True)
    df["is_taker"] = (df["spread_rank"] < 0.3).astype(int)

    if df["is_taker"].nunique() < 2:
        print("[WARN] Not enough diversity in taker/maker labels. Skipping.")
        return

    # Balance classes
    df_taker = df[df["is_taker"] == 1]
    df_maker = df[df["is_taker"] == 0]
    min_len = min(len(df_taker), len(df_maker))

    df_taker = resample(df_taker, n_samples=min_len, random_state=42)
    df_maker = resample(df_maker, n_samples=min_len, random_state=42)
    df_balanced = pd.concat([df_taker, df_maker])

    features = df_balanced[["spread", "depth", "volatility"]]
    target = df_balanced["is_taker"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    model = LogisticRegression()
    model.fit(X_scaled, target)

    joblib.dump(model, "maker_taker_model.pkl")
    joblib.dump(scaler, "maker_taker_scaler.pkl")
    print("[INFO] Maker/Taker model trained and saved.")
