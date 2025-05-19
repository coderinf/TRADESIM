📉 Real-Time Trading Cost Estimator
A real-time tick-level trading analytics engine that estimates slippage, market impact, and maker/taker probability using machine learning on live L2 order book data from crypto exchanges like OKX.

🚀 Project Overview
This project connects to a live crypto exchange WebSocket, streams L2 order book data, and performs real-time analytics to estimate the true trading cost of a given order size. It answers:

How much slippage can I expect when placing an order?

What is the market impact of my order size?

Will my trade likely be executed as a maker or taker?

What is the total net cost after fees, latency, and price movement?

The system uses statistical and machine learning models trained on recent tick data collected in real time.

🎯 Key Features
✅ Real-time WebSocket streaming of order book
✅ Tick data collection and storage
✅ Automated ML model training (slippage, Almgren-Chriss, maker/taker)
✅ Order cost prediction using trained models
✅ Integrated simulation for market impact
✅ CLI or UI-ready (Streamlit-compatible)

📊 Models Used
Model	Type	Purpose
Slippage Model	Linear Regression	Predicts expected slippage based on spread, depth, volatility
Almgren-Chriss Model	Linear Regression	Estimates permanent and temporary market impact of orders
Maker/Taker Classifier	Logistic Regression	Classifies order as taker (market) or maker (limit)

System UI
![input Interface](https://github.com/user-attachments/assets/23957b20-ffc1-4f12-b7f8-7a0399e1b3a3)

Input Parameters and Output Responses
![output response](https://github.com/user-attachments/assets/61d3bcab-c130-4a20-8318-700530eae67b)

Market Impact Over Size
![Graph -1](https://github.com/user-attachments/assets/0f33ff6a-db89-4be9-bf2d-01b94bef3767)

Latency over Time
![Graph -2](https://github.com/user-attachments/assets/cd797ad5-2c85-4f5c-a498-803a44b70752)

📡 How It Works
✅ Tick Data Collection
Streams live order book data and computes features:

Mid-price

Spread

Depth

Volume

Returns & volatility

✅ Model Training
After collecting ~350 ticks:

Trains a slippage regression model

Estimates Almgren-Chriss impact (η, γ)

Trains logistic regression classifier for maker/taker prediction

✅ Real-Time Prediction
Loads trained models

Streams a few live ticks

Predicts:

Slippage

Market Impact

Fee

Probability of being a taker

Net Cost
📈 Sample Outputs
Order Size	Slippage	Market Impact	Fee	Net Cost	Taker Probability
$50	0.024	0.013	0.015	0.052	0.68
$100	0.045	0.033	0.030	0.108	0.74

Input Parameters and Output Responses
![output response](https://github.com/user-attachments/assets/61d3bcab-c130-4a20-8318-700530eae67b)









