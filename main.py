import datetime
import logging
import yfinance as yf
from fastapi import FastAPI
from pydantic import BaseModel
import ta
import pandas as pd
import pickle
import os

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def get_data(stock_ticker, start_date, end_date):
    # Download NVDA stock data
    nvda = yf.download(stock_ticker, start=start_date, end=end_date)

    # Download macroeconomic indicators
    # Example: US 10-Year Treasury Yield, USD Index, S&P 500
    macro = yf.download(['^TNX', 'DX-Y.NYB', '^GSPC'], start=start_date, end=end_date)['Close']
    macro.columns = ['TNX', 'USD_Index', 'SP500']

    # Merge NVDA and macro data
    df = nvda[['High', 'Low', 'Open', 'Close', 'Volume']].copy()
    df = df.droplevel(1, axis=1)
    df = df.join(macro)

    # Add technical indicators
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['macd'] = ta.trend.MACD(df['Close']).macd()
    df['macd_signal'] = ta.trend.MACD(df['Close']).macd_signal()
    df['ema_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
    df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()

    df.drop(['Low', 'High'], inplace=True, axis=1)
    # Create target variable (next-day close)
    df['target'] = df['Close'].shift(-1)
    df = df.drop(columns=['target'])

    df.dropna(inplace=True)

    return df


class StockPriceDate(BaseModel):
    stock_date: str

app = FastAPI(
    title="Stock Price Predictor",
    description="A stock price prediction app",
    version="1.0.0"
)

model_path = os.path.join("models/", "model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)


@app.post("/predict")
def predict_price(date: StockPriceDate):
    date = date.stock_date
    logging.info(f"date is {date}")
    stock_date = pd.to_datetime(date)
    logging.info(f"date is {stock_date}")
    start_date = stock_date - pd.to_timedelta(240, unit='d')
    stock_data = get_data('NVDA', start_date, stock_date)
    prediction = model.predict(stock_data)
    return {"prediction": float(prediction[-1])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
