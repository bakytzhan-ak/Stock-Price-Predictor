# Import necessary libs
import os
import pickle
import sys
import yfinance as yf
import pandas as pd
import ta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


ticker = 'NVDA'
end_date = pd.to_datetime('today')
start_date = end_date - pd.to_timedelta(240, unit='d')

def get_data(ticker, start_date, end_date):
    # Download NVDA stock data
    nvda = yf.download(ticker, start=start_date, end=end_date)

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
    df.dropna(inplace=True)

    return df

# Load data
df = get_data(ticker, start_date, end_date)

# Train/test split
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)

# GridSearchCV for XGBoost
param_grid = {
    'n_estimators': [5, 10, 20, 50, 100, 200],
    'max_depth': [3, 5, 7, 9, 15],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

model = XGBRegressor(objective='reg:squarederror')
grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_root_mean_squared_error', verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

# Evaluate best model
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)
print(f"Best RMSE: {rmse:.2f}")
print("Best Parameters:", grid.best_params_)

# Create a directory and save model
os.makedirs('models', exist_ok=True)

with open(os.path.join('models', 'model.pkl'), 'wb') as f:
    pickle.dump(best_model, f)

sys.exit()
