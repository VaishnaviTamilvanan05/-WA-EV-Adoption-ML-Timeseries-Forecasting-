# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# %% Data Preparation
data_path = 'E:\\Capstone\\data\\EV\\processed\\ev_cleaned_data.csv'
data = pd.read_csv(data_path)
data['transaction_date'] = pd.to_datetime(data['transaction_date'])

# Monthly aggregation for EV counts and average sale price
ev_monthly = data.groupby(data['transaction_date'].dt.to_period('M')).size()
ev_monthly.index = ev_monthly.index.to_timestamp()

df_prophet = ev_monthly.reset_index()
df_prophet.columns = ['ds', 'y']

df_prophet_ext = (data.groupby(data['transaction_date'].dt.to_period('M'))
    .agg({'dol_vehicle_id': 'count', 'sale_price': 'mean'})
    .reset_index()
)
df_prophet_ext['transaction_date'] = df_prophet_ext['transaction_date'].dt.to_timestamp()
df_prophet_ext.rename(columns={'transaction_date': 'ds', 'dol_vehicle_id': 'y', 'sale_price': 'avg_sale_price'}, inplace=True)
df_prophet_ext = df_prophet_ext.sort_values('ds')

# Train-Test Split
train_arima = ev_monthly[:-12]
test_arima = ev_monthly[-12:]
train_prophet_ext = df_prophet_ext.iloc[:-12]
test_prophet_ext = df_prophet_ext.iloc[-12:]

# %% ARIMA Model
arima_model = ARIMA(train_arima, order=(2, 1, 2)).fit()
arima_forecast = arima_model.forecast(steps=len(test_arima))

print("ARIMA Model:")
print("Test RMSE:", np.sqrt(mean_squared_error(test_arima, arima_forecast)))
print("Test MAE:", mean_absolute_error(test_arima, arima_forecast))
print("AIC:", arima_model.aic)
print("BIC:", arima_model.bic)

# Future forecast
future_arima_forecast = arima_model.forecast(steps=48)
forecast_dates = pd.date_range(start=ev_monthly.index[-1], periods=49, freq='M')[1:]

# Plot ARIMA Results
plt.figure(figsize=(12,6))
plt.plot(ev_monthly, label='Observed', marker='o')
plt.plot(test_arima.index, arima_forecast, label='Test Forecast', color='red', linestyle='--')
plt.plot(forecast_dates, future_arima_forecast, label='Future Forecast', color='green', linestyle='--')
plt.title("ARIMA Forecast")
plt.xlabel("Date")
plt.ylabel("EV Registrations")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% SARIMA Model
sarima_model = SARIMAX(train_arima, order=(2,1,2), seasonal_order=(1,1,1,12)).fit(disp=False)
sarima_forecast = sarima_model.forecast(steps=len(test_arima))

print("\nSARIMA Model:")
print("Test RMSE:", np.sqrt(mean_squared_error(test_arima, sarima_forecast)))
print("Test MAE:", mean_absolute_error(test_arima, sarima_forecast))
print("AIC:", sarima_model.aic)
print("BIC:", sarima_model.bic)

future_sarima_forecast = sarima_model.forecast(steps=48)

# Plot SARIMA Results
plt.figure(figsize=(12,6))
plt.plot(ev_monthly, label='Observed', marker='o')
plt.plot(test_arima.index, sarima_forecast, label='Test Forecast', color='red', linestyle='--')
plt.plot(forecast_dates, future_sarima_forecast, label='Future Forecast', color='green', linestyle='--')
plt.title("SARIMA Forecast")
plt.xlabel("Date")
plt.ylabel("EV Registrations")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% Prophet Basic Model
model_basic = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                      changepoint_prior_scale=3, seasonality_prior_scale=10, n_changepoints=50)
model_basic.fit(df_prophet.iloc[:-12])

future_basic = model_basic.make_future_dataframe(periods=60, freq='M')
forecast_basic = model_basic.predict(future_basic)

# Evaluate
merged_basic = pd.merge_asof(df_prophet.iloc[-12:], forecast_basic[['ds', 'yhat']], on='ds', tolerance=pd.Timedelta('15 days')).dropna()

if not merged_basic.empty:
    print("\nProphet (Basic) Model:")
    print("Test RMSE:", np.sqrt(mean_squared_error(merged_basic['y'], merged_basic['yhat'])))
    print("Test MAE:", mean_absolute_error(merged_basic['y'], merged_basic['yhat']))

# Plot Prophet Basic
fig_basic = model_basic.plot(forecast_basic)
plt.title("Prophet Basic Forecast")
plt.xlabel("Date")
plt.ylabel("EV Registrations")
plt.show()

# %% Prophet with External Regressor (Log + Auto-Tuning)
train_prophet_ext['y'] = np.log1p(train_prophet_ext['y'])

# Forecast sale price
df_price = train_prophet_ext[['ds', 'avg_sale_price']].rename(columns={'avg_sale_price': 'y'})
price_model = Prophet(yearly_seasonality=True)
price_model.fit(df_price)
future_price = price_model.make_future_dataframe(periods=60, freq='M')
forecast_price = price_model.predict(future_price)[['ds', 'yhat']].rename(columns={'yhat': 'avg_sale_price'})

# Hyperparameter grid search
param_grid = [
    {'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 1, 'n_changepoints': 50},
    {'seasonality_mode': 'multiplicative', 'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 5, 'n_changepoints': 50},
    {'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10, 'n_changepoints': 50}
]

best_params, best_rmse = None, float('inf')
for params in param_grid:
    try:
        model = Prophet(**params)
        model.add_regressor('avg_sale_price')
        model.fit(train_prophet_ext)

        df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='365 days', parallel="processes")
        df_perf = performance_metrics(df_cv)
        rmse = df_perf['rmse'].mean()
        print(f"Params: {params}, RMSE: {rmse:.2f}")

        if rmse < best_rmse:
            best_rmse, best_params = rmse, params
    except Exception as e:
        print(f"⚠️ Failed with {params}: {e}")

print(f"\n✅ Best Parameters: {best_params}")

# Train final model
final_model = Prophet(**best_params)
final_model.add_regressor('avg_sale_price')
final_model.fit(train_prophet_ext)

# Forecast future
future = final_model.make_future_dataframe(periods=60, freq='M')
future = future.merge(forecast_price, on='ds', how='left')

# Handle missing sale prices
avg_growth_rate = train_prophet_ext['avg_sale_price'].pct_change().mean()
last_price = train_prophet_ext['avg_sale_price'].iloc[-1]
future['avg_sale_price'].fillna(method='ffill', inplace=True)

forecast = final_model.predict(future)
forecast['yhat_orig'] = np.expm1(forecast['yhat'])

# Evaluate on Test Set
test_eval = pd.merge_asof(test_prophet_ext.sort_values('ds'), forecast[['ds', 'yhat_orig']].sort_values('ds'), on='ds', tolerance=pd.Timedelta('30 days'))

print("\nTest Set Metrics:")
print("RMSE:", np.sqrt(mean_squared_error(test_prophet_ext['y'], test_eval['yhat_orig'])))
print("MAE:", mean_absolute_error(test_prophet_ext['y'], test_eval['yhat_orig']))

# Plot final results
plt.figure(figsize=(16,8))
plt.plot(df_prophet_ext['ds'], df_prophet_ext['y'], label='Historical', marker='o')
plt.plot(test_prophet_ext['ds'], test_prophet_ext['y'], label='Actual Test', color='green', marker='x')
plt.plot(forecast['ds'], forecast['yhat_orig'], label='Forecast', color='orange', linestyle='--')
plt.title("Final EV Registration Forecast")
plt.xlabel("Date")
plt.ylabel("Monthly Registrations")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Save model
save_path = r"E:\\Capstone\\models\\best_model_Prophet1_ext_log.pkl"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, "wb") as f:
    pickle.dump(final_model, f)

print("\nModel saved to:", save_path)
