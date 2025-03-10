# %% Global Data Preparation
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt
from prophet import Prophet
import pickle
import os

# Load raw data
data = pd.read_csv('E:\\Capstone\\data\\EV\\processed\\ev_cleaned_data.csv')
data['transaction_date'] = pd.to_datetime(data['transaction_date'])

# Global aggregation for EV registrations (for SARIMA/Prophet Basic)
ev_monthly = data.groupby(data['transaction_date'].dt.to_period('M')).size()
ev_monthly.index = ev_monthly.index.to_timestamp()
df_prophet = ev_monthly.reset_index()
df_prophet.columns = ['ds', 'y']
df_prophet = df_prophet.sort_values('ds')

# Global aggregation for external regressor: average sale price
monthly_agg = data.groupby(data['transaction_date'].dt.to_period('M')).agg({
    'dol_vehicle_id': 'count',
    'sale_price': 'mean'
}).reset_index()
monthly_agg['transaction_date'] = monthly_agg['transaction_date'].dt.to_timestamp()
df_prophet_ext = monthly_agg.rename(columns={
    'transaction_date': 'ds',
    'dol_vehicle_id': 'y',
    'sale_price': 'avg_sale_price'
}).sort_values('ds')

# %% SARIMA
# Split data into training and test sets (last 12 months as test)
train = ev_monthly[:-12]
test = ev_monthly[-12:]

sarima_order = (2, 1, 2)
seasonal_order = (1, 1, 1, 12)
sarima_model = SARIMAX(train, order=sarima_order, seasonal_order=seasonal_order)
sarima_fit = sarima_model.fit(disp=False)

forecast_steps = len(test)
sarima_forecast = sarima_fit.forecast(steps=forecast_steps)

rmse_sarima = np.sqrt(mean_squared_error(test, sarima_forecast))
mae_sarima = mean_absolute_error(test, sarima_forecast)
aic = sarima_fit.aic
bic = sarima_fit.bic

print("SARIMA Model:")
print("Test RMSE:", rmse_sarima)
print("Test MAE:", mae_sarima)
print("AIC:", aic)
print("BIC:", bic)

future_steps = 48
future_forecast = sarima_fit.forecast(steps=future_steps)
forecast_dates = pd.date_range(start=ev_monthly.index[-1], periods=future_steps+1, freq='M')[1:]

plt.figure(figsize=(12,6))
plt.plot(ev_monthly, label='Observed', marker='o')
plt.plot(test.index, sarima_forecast, label='Test Forecast', color='red', linestyle='--')
plt.plot(forecast_dates, future_forecast, label='Future Forecast', color='green', linestyle='--')
plt.title("SARIMA Model Forecast of EV Registrations")
plt.xlabel("Date")
plt.ylabel("EV Registrations")
plt.legend()
plt.show()

# %% Prophet (Basic)
train_size = len(df_prophet) - 12
train_df = df_prophet.iloc[:train_size].copy()
test_df = df_prophet.iloc[train_size:].copy()

model_basic = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                       changepoint_prior_scale=3, seasonality_prior_scale=10, n_changepoints=50)
model_basic.fit(train_df)

forecast_periods = 12 + 48
future_basic = model_basic.make_future_dataframe(periods=forecast_periods, freq='M')
forecast_basic = model_basic.predict(future_basic)

# Evaluate Prophet (Basic)
test_df = test_df.sort_values('ds')
future_basic = future_basic.sort_values('ds')
test_forecast_basic = pd.merge_asof(test_df, forecast_basic[['ds', 'yhat']], on='ds', tolerance=pd.Timedelta('15 days'))
test_forecast_basic = test_forecast_basic.dropna(subset=['yhat'])
if len(test_forecast_basic) != 0:
    rmse_basic = np.sqrt(mean_squared_error(test_forecast_basic['y'], test_forecast_basic['yhat']))
    mae_basic = mean_absolute_error(test_forecast_basic['y'], test_forecast_basic['yhat'])
    print("\nProphet (Basic) Model:")
    print("Test RMSE:", rmse_basic)
    print("Test MAE:", mae_basic)

plt.figure(figsize=(12,6))
fig_basic = model_basic.plot(forecast_basic)
plt.title("Prophet (Basic) Forecast of EV Registrations")
plt.xlabel("Date")
plt.ylabel("EV Registrations")
plt.show()
fig_components_basic = model_basic.plot_components(forecast_basic)
plt.show()

# %% Prophet with External Regressor (sale price)
# Use all available data for training (up to Dec 31, 2024)
train_df_ext = df_prophet_ext.copy()

model_ext = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                    changepoint_prior_scale=3, seasonality_prior_scale=10, n_changepoints=50, interval_width=0.85)
model_ext.add_regressor('avg_sale_price')
model_ext.fit(train_df_ext)

# Create future DataFrame: forecast for 48 months beyond Dec 2024
future_ext = model_ext.make_future_dataframe(periods=48, freq='M')
historical_prices = train_df_ext[['ds', 'avg_sale_price']]
future_ext = future_ext.merge(historical_prices, on='ds', how='left')

# Project future sale prices
df_prophet_ext['growth_rate'] = df_prophet_ext['avg_sale_price'].pct_change()
clean_growth = df_prophet_ext['growth_rate'].replace([np.inf, -np.inf], np.nan).dropna().clip(lower=-0.5, upper=0.5)
avg_growth_rate = clean_growth.mean()
if pd.isna(avg_growth_rate) or abs(avg_growth_rate) > 0.5:
    avg_growth_rate = 0.02
print("\nAverage monthly growth rate for sale price:", avg_growth_rate)

last_price_ext = train_df_ext['avg_sale_price'].iloc[-1]
first_future_idx = future_ext[future_ext['avg_sale_price'].isna()].index[0]
for i in range(first_future_idx, len(future_ext)):
    months_ahead = i - first_future_idx + 1
    future_ext.loc[future_ext.index[i], 'avg_sale_price'] = last_price_ext * ((1 + avg_growth_rate) ** months_ahead)

forecast_ext = model_ext.predict(future_ext)

# Evaluate Prophet with External Regressor using last 12 months as test set
test_df_ext = df_prophet_ext.iloc[-12:].copy()
test_df_ext = test_df_ext.sort_values('ds')
future_ext = future_ext.sort_values('ds')
test_forecast_ext = pd.merge_asof(test_df_ext, forecast_ext[['ds', 'yhat']], on='ds', tolerance=pd.Timedelta('15 days'))
test_forecast_ext = test_forecast_ext.dropna(subset=['yhat'])
metrics_ext = {}
if len(test_forecast_ext) != 0:
    metrics_ext['rmse'] = np.sqrt(mean_squared_error(test_forecast_ext['y'], test_forecast_ext['yhat']))
    metrics_ext['mae'] = mean_absolute_error(test_forecast_ext['y'], test_forecast_ext['yhat'])
    print("\nProphet with External Regressor (sale price) Model:")
    print("Test RMSE:", metrics_ext['rmse'])
    print("Test MAE:", metrics_ext['mae'])
else:
    print("No matching forecast dates found for test data in External Regressor model.")
    
plt.figure(figsize=(12,6))
fig_ext = model_ext.plot(forecast_ext)
plt.xlim(pd.Timestamp('2023-01-01'), pd.Timestamp('2029-01-01'))
plt.title("Forecast of EV Registrations with External Regressor (sale price) Until 2028")
plt.xlabel("Date")
plt.ylabel("EV Registrations")
plt.show()

fig2_ext = model_ext.plot_components(forecast_ext)
plt.show()

# Save the best model (Prophet with External Regressor)
model_save_path = r"E:\Capstone\models\best_model_Prophet_ext.pkl"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
with open(model_save_path, "wb") as f:
    pickle.dump(model_ext, f)
print("\nModel saved at:", model_save_path)

# %% Function Definition for County-Level Forecasting (Using Saved Best Model Approach)
def forecast_county(county_name, data, forecast_periods=48, test_periods=12, save_model_dir=r"E:\Capstone\models"):
    """
    Processes data for a specific county, generates forecasts using the best Prophet model with external regressor,
    evaluates performance on the test set, plots results, and saves the county-specific model.
    """
    county_data = data[data['county'] == county_name].copy()
    county_data['transaction_date'] = pd.to_datetime(county_data['transaction_date'])
    
    county_monthly = county_data.groupby(county_data['transaction_date'].dt.to_period('M')).agg({
        'dol_vehicle_id': 'count',
        'sale_price': 'mean'
    }).reset_index()
    county_monthly['transaction_date'] = county_monthly['transaction_date'].dt.to_timestamp()
    
    df_county = county_monthly.rename(columns={
        'transaction_date': 'ds',
        'dol_vehicle_id': 'y',
        'sale_price': 'avg_sale_price'
    }).sort_values('ds')
    
    train_size = len(df_county) - test_periods
    train_df = df_county.iloc[:train_size].copy()
    test_df = df_county.iloc[train_size:].copy()
    
    model = Prophet(yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_prior_scale=3,
                    seasonality_prior_scale=10,
                    n_changepoints=50,
                    interval_width=0.85)
    model.add_regressor('avg_sale_price')
    model.fit(train_df)
    
    total_steps = test_periods + forecast_periods
    future_df = model.make_future_dataframe(periods=total_steps, freq='M')
    historical_prices = train_df[['ds', 'avg_sale_price']]
    future_df = future_df.merge(historical_prices, on='ds', how='left')
    
    train_df['growth_rate'] = train_df['avg_sale_price'].pct_change()
    clean_growth = train_df['growth_rate'].replace([np.inf, -np.inf], np.nan).dropna().clip(lower=-0.5, upper=0.5)
    avg_growth_rate = clean_growth.mean()
    if pd.isna(avg_growth_rate) or abs(avg_growth_rate) > 0.5:
        avg_growth_rate = 0.02
    print(f"{county_name} - Average monthly growth rate for sale price:", avg_growth_rate)
    
    last_price = train_df['avg_sale_price'].iloc[-1]
    first_future_idx = future_df[future_df['avg_sale_price'].isna()].index[0]
    for i in range(first_future_idx, len(future_df)):
        months_ahead = i - first_future_idx + 1
        future_df.loc[future_df.index[i], 'avg_sale_price'] = last_price * ((1 + avg_growth_rate) ** months_ahead)
    
    forecast_df = model.predict(future_df)
    
    test_df = test_df.sort_values('ds')
    future_df = future_df.sort_values('ds')
    test_forecast = pd.merge_asof(test_df, forecast_df[['ds', 'yhat']], on='ds', tolerance=pd.Timedelta('15 days'))
    test_forecast = test_forecast.dropna(subset=['yhat'])
    
    metrics = {}
    if len(test_forecast) != 0:
        metrics['rmse'] = np.sqrt(mean_squared_error(test_forecast['y'], test_forecast['yhat']))
        metrics['mae'] = mean_absolute_error(test_forecast['y'], test_forecast['yhat'])
        print(f"\n{county_name} - Prophet with External Regressor Model:")
        print("Test RMSE:", metrics['rmse'])
        print("Test MAE:", metrics['mae'])
    else:
        print(f"No matching forecast dates found for test data in {county_name}.")
    
    plt.figure(figsize=(12,6))
    fig = model.plot(forecast_df)
    plt.xlim(pd.Timestamp('2023-01-01'), pd.Timestamp('2029-01-01'))
    plt.title(f"{county_name} County: Forecast with External Regressor Until 2028")
    plt.xlabel("Date")
    plt.ylabel("EV Registrations")
    plt.show()
    
    fig2 = model.plot_components(forecast_df)
    plt.show()
    
    os.makedirs(save_model_dir, exist_ok=True)
    model_save_path = os.path.join(save_model_dir, f"best_model_Prophet_ext_{county_name}.pkl")
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"{county_name} Model saved at:", model_save_path)
    
    return forecast_df, metrics

# %% Run Forecasting for Specified Counties
counties = ["King", "Snohomish", "Pierce", "Clark", "Kitsap",
            "Thurston", "Spokane", "Whatcom", "Benton", "Skagit"]

county_forecasts = {}
county_metrics = {}

for county in counties:
    print(f"\nProcessing forecast for {county} County:")
    fc_df, met = forecast_county(county, data)
    county_forecasts[county] = fc_df
    county_metrics[county] = met

for county, met in county_metrics.items():
    print(f"\n{county} County Metrics:")
    print(met)
