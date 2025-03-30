# %% Global Data Preparation
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt
from prophet import Prophet
import pickle
import os

# %%
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


# %% Complete Updated Code: Prophet with External Regressor (Log Transformation) & Auto-Tuning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import os
import time

# ----------------------------
# 1. Global Data Preparation
# ----------------------------
data = pd.read_csv('E:\\Capstone\\data\\EV\\processed\\ev_cleaned_data.csv')
data['transaction_date'] = pd.to_datetime(data['transaction_date'])

# Aggregate data to monthly level for EV registrations and average sale price
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


# ----------------------------
# 2. Train-Test Split
# ----------------------------
# Use last 12 months as test set
test_months = 12
train_data = df_prophet_ext.iloc[:-test_months].copy()
test_data = df_prophet_ext.iloc[-test_months:].copy()

# ----------------------------
# 3. Log Transformation of Target Variable
# ----------------------------
# Apply log transformation to stabilize variance
train_data['y'] = np.log1p(train_data['y'])

# ----------------------------
# 4. Forecast External Regressor (Sale Price) via a Dedicated Price Model
# ----------------------------
price_df = train_data[['ds', 'avg_sale_price']].rename(columns={'avg_sale_price': 'y'})
price_model = Prophet(yearly_seasonality=True)
price_model.fit(price_df)
price_future = price_model.make_future_dataframe(periods=test_months+48, freq='M')
price_forecast = price_model.predict(price_future)[['ds', 'yhat']].rename(columns={'yhat': 'avg_sale_price'})
# %%
# ----------------------------
# 5. Hyperparameter Tuning
# ----------------------------
param_grid = [
    {'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 1, 'n_changepoints': 25},
    {'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 1, 'n_changepoints': 50},
    {'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 1, 'n_changepoints': 75},
    {'seasonality_mode': 'multiplicative', 'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 5, 'n_changepoints': 25},
    {'seasonality_mode': 'multiplicative', 'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 5, 'n_changepoints': 50},
    {'seasonality_mode': 'multiplicative', 'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 5, 'n_changepoints': 75},
    {'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10, 'n_changepoints': 25},
    {'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10, 'n_changepoints': 50},
    {'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10, 'n_changepoints': 75},
]

best_params = param_grid[0].copy()
best_rmse = float('inf')

for params in param_grid:
    try:
        current_model = Prophet(**params)
        current_model.add_regressor('avg_sale_price')
        current_model.fit(train_data)
        
        df_cv = cross_validation(current_model,
                                 initial='730 days',
                                 period='180 days',
                                 horizon='365 days',
                                 parallel="processes")
        df_p = performance_metrics(df_cv)
        current_rmse = df_p['rmse'].mean()
        
        print(f"Params: {params}, RMSE: {current_rmse:.2f}")
        
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_params = params.copy()
            
    except Exception as e:
        print(f"⚠️ Failed with params {params}: {str(e)}")
        continue

print(f"\n✅ Best Parameters: {best_params}")
print(f"🏆 Validation RMSE: {best_rmse:.2f}")

# ----------------------------
# 6. Final Model Training with Best Parameters
# ----------------------------
final_params = best_params if best_params else {'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.05}
final_model = Prophet(**final_params)
final_model.add_regressor('avg_sale_price')
final_model.fit(train_data)

# ----------------------------
# 7. Forecasting
# ----------------------------

# Create a future DataFrame for (test_months + 48) months ahead
future = final_model.make_future_dataframe(periods=test_months + 48, freq='M')

# Merge the external regressor (avg_sale_price) from the price forecast
future = future.merge(price_forecast, on='ds', how='left')

# Compute the average monthly growth rate from the training data's sale price
train_data['growth_rate'] = train_data['avg_sale_price'].pct_change()
clean_growth = train_data['growth_rate'].replace([np.inf, -np.inf], np.nan).dropna().clip(lower=-0.5, upper=0.5)
avg_growth_rate = clean_growth.mean()
if pd.isna(avg_growth_rate) or abs(avg_growth_rate) > 0.5:
    avg_growth_rate = 0.02
print("\nAverage monthly growth rate for sale price:", avg_growth_rate)

# Get the last observed sale price from the training data
last_price = train_data['avg_sale_price'].iloc[-1]

# Check for any missing future sale price values and fill them using exponential growth
na_future = future[future['avg_sale_price'].isna()]
if not na_future.empty:
    first_future_idx = na_future.index[0]
    for i in range(first_future_idx, len(future)):
        months_ahead = i - first_future_idx + 1
        future.loc[future.index[i], 'avg_sale_price'] = last_price * ((1 + avg_growth_rate) ** months_ahead)
else:
    print("No missing future sale price values to project.")

# Generate the forecast using the final model
forecast = final_model.predict(future)

# ----------------------------
# 8. Back-Transformation & Evaluation
# ----------------------------
# Back-transform predictions to original scale
forecast['yhat_orig'] = np.expm1(forecast['yhat'])
forecast['yhat_lower_orig'] = np.expm1(forecast['yhat_lower'])
forecast['yhat_upper_orig'] = np.expm1(forecast['yhat_upper'])

# Evaluate on test set (using original scale values from df_prophet_ext)
test_eval = pd.merge_asof(test_data.sort_values('ds'),
                          forecast[['ds', 'yhat_orig']].sort_values('ds'),
                          on='ds',
                          tolerance=pd.Timedelta('30 days'))
print("Number of matched test rows with 30-day tolerance:", len(test_eval))
if test_eval.empty:
    print("No matching test dates found in the forecast. Check date ranges!")
else:
    metrics={
        'RMSE': np.sqrt(mean_squared_error(test_data['y'], test_eval['yhat_orig'])),
        'MAE': mean_absolute_error(test_data['y'], test_eval['yhat_orig']),
        'MAPE': np.mean(np.abs((test_data['y'] - test_eval['yhat_orig']) / test_data['y']))
    }

print("\nTest Set Metrics:")
print("RMSE: {:.2f}".format(metrics['RMSE']))
print("MAE: {:.2f}".format(metrics['MAE']))
print("MAPE: {:.2%}".format(metrics['MAPE']))

# ----------------------------
# 9. Enhanced Visualization
# ----------------------------
plt.figure(figsize=(16,8))
plt.plot(df_prophet_ext['ds'], df_prophet_ext['y'], label='Historical EV Registrations', color='blue', marker='o')
plt.plot(test_data['ds'], test_data['y'], label='Actual Test Values', color='green', marker='x')
plt.plot(forecast['ds'], forecast['yhat_orig'], label=f'Forecast (RMSE: {metrics["RMSE"]:.2f})', color='orange', linestyle='--')
plt.fill_between(forecast['ds'], forecast['yhat_lower_orig'], forecast['yhat_upper_orig'], color='orange', alpha=0.1)
plt.title("EV Registrations Forecast (2010-2028)", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Monthly Registrations", fontsize=12)
plt.xlim(pd.to_datetime('2010-01-01'), pd.to_datetime('2028-12-31'))
plt.xticks(pd.date_range(start='2010-01-01', end='2028-12-31', freq='2Y'))
plt.axvline(x=test_data['ds'].min(), color='red', linestyle=':', label='Test Period Start')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------
# 10. Save the Final Model
# ----------------------------
model_save_path = r"E:\Capstone\models\best_model_Prophet_ext_log.pkl"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
with open(model_save_path, "wb") as f:
    pickle.dump(final_model, f)
print("\nModel saved at:", model_save_path)



# %% King County Forecasting with Auto-Tuning, Log Transformation, and External Regressor

king_data = data[data['county'] == 'King'].copy()
king_data['transaction_date'] = pd.to_datetime(king_data['transaction_date'])
king_monthly = king_data.groupby(king_data['transaction_date'].dt.to_period('M')).agg({
    'dol_vehicle_id': 'count',
    'sale_price': 'mean'
}).reset_index()
king_monthly['transaction_date'] = king_monthly['transaction_date'].dt.to_timestamp()

df_king = king_monthly.rename(columns={
    'transaction_date': 'ds',
    'dol_vehicle_id': 'y',
    'sale_price': 'avg_sale_price'
}).sort_values('ds')
# ----------------------------
# 1. Train-Test Split
# ----------------------------
test_months = 12
train_king = df_king.iloc[:-test_months].copy()
test_king = df_king.iloc[-test_months:].copy()

# ----------------------------
# 2. Log Transformation of Target Variable (for training)
# ----------------------------
train_king['y'] = np.log1p(train_king['y'])

# ----------------------------
# 3. Hyperparameter Tuning
# ----------------------------
param_grid = [
    {'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 1, 'n_changepoints': 25},
    {'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 1, 'n_changepoints': 50},
    {'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 1, 'n_changepoints': 75},
    {'seasonality_mode': 'multiplicative', 'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 5, 'n_changepoints': 25},
    {'seasonality_mode': 'multiplicative', 'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 5, 'n_changepoints': 50},
    {'seasonality_mode': 'multiplicative', 'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 5, 'n_changepoints': 75},
    {'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10, 'n_changepoints': 25},
    {'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10, 'n_changepoints': 50},
    {'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10, 'n_changepoints': 75},
]

best_params = param_grid[0].copy()
best_rmse = float('inf')

for params in param_grid:
    try:
        current_model = Prophet(**params)
        current_model.add_regressor('avg_sale_price')
        current_model.fit(train_king)
        
        df_cv = cross_validation(current_model,
                                 initial='730 days',
                                 period='180 days',
                                 horizon='365 days',
                                 parallel="processes")
        df_p = performance_metrics(df_cv)
        current_rmse = df_p['rmse'].mean()
        print(f"Params: {params}, RMSE: {current_rmse:.2f}")
        
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_params = params.copy()
            
    except Exception as e:
        print(f"⚠️ Failed with params {params}: {str(e)}")
        continue

print(f"\n✅ Best Parameters: {best_params}")
print(f"🏆 Validation RMSE: {best_rmse:.2f}")

# ----------------------------
# 4. Final Model Training with Best Parameters
# ----------------------------
final_model = Prophet(**best_params)
final_model.add_regressor('avg_sale_price')
final_model.fit(train_king)

# ----------------------------
# 5. Forecast External Regressor (Sale Price)
# ----------------------------
# For King County, we forecast the sale price using the training data.
price_df = train_king[['ds', 'avg_sale_price']].rename(columns={'avg_sale_price': 'y'})
price_model = Prophet(yearly_seasonality=True)
price_model.fit(price_df)
price_future = price_model.make_future_dataframe(periods=test_months+48, freq='M')
price_forecast = price_model.predict(price_future)[['ds', 'yhat']].rename(columns={'yhat': 'avg_sale_price'})

# ----------------------------
# 6. Forecasting
# ----------------------------
# Create a future DataFrame for (test_months + 48) months ahead
future = final_model.make_future_dataframe(periods=test_months + 48, freq='M')
# Merge the external regressor (avg_sale_price) from the price forecast
future = future.merge(price_forecast, on='ds', how='left')

# Compute the average monthly growth rate from training data's sale price
train_king['growth_rate'] = train_king['avg_sale_price'].pct_change()
clean_growth = train_king['growth_rate'].replace([np.inf, -np.inf], np.nan).dropna().clip(lower=-0.5, upper=0.5)
avg_growth_rate = clean_growth.mean()
if pd.isna(avg_growth_rate) or abs(avg_growth_rate) > 0.5:
    avg_growth_rate = 0.02
print("\nAverage monthly growth rate for sale price (King):", avg_growth_rate)

last_price = train_king['avg_sale_price'].iloc[-1]
# Check for missing future sale price values and fill them using exponential growth
na_future = future[future['avg_sale_price'].isna()]
if not na_future.empty:
    first_future_idx = na_future.index[0]
    for i in range(first_future_idx, len(future)):
        months_ahead = i - first_future_idx + 1
        future.loc[future.index[i], 'avg_sale_price'] = last_price * ((1 + avg_growth_rate) ** months_ahead)
else:
    print("No missing future sale price values to project.")

forecast_king = final_model.predict(future)

# ----------------------------
# 7. Back-Transformation & Evaluation
# ----------------------------
# Back-transform predictions to original scale
forecast_king['yhat_orig'] = np.expm1(forecast_king['yhat'])
forecast_king['yhat_lower_orig'] = np.expm1(forecast_king['yhat_lower'])
forecast_king['yhat_upper_orig'] = np.expm1(forecast_king['yhat_upper'])

# Evaluate on test set using merge_asof with 30-day tolerance
test_eval = pd.merge_asof(test_king.sort_values('ds'),
                          forecast_king[['ds', 'yhat_orig']].sort_values('ds'),
                          on='ds',
                          tolerance=pd.Timedelta('30 days'))
print("Number of matched test rows with 30-day tolerance:", len(test_eval))
if test_eval.empty:
    print("No matching test dates found in forecast. Check date ranges!")
else:
    metrics_king = {
        'RMSE': np.sqrt(mean_squared_error(test_king['y'], test_eval['yhat_orig'])),
        'MAE': mean_absolute_error(test_king['y'], test_eval['yhat_orig']),
        'MAPE': np.mean(np.abs((test_king['y'] - test_eval['yhat_orig']) / test_king['y']))

    }
    print("\nKing County - Prophet with External Regressor Model:")
    print("Test RMSE:", metrics_king['RMSE'])
    print("Test MAE:", metrics_king['MAE'])
    print("Test MAPE:", metrics_king['MAPE'])

# ----------------------------
# 8. Visualization
# ----------------------------
plt.figure(figsize=(12,6))
fig_king = final_model.plot(forecast_king)
plt.xlim(pd.Timestamp('2023-01-01'), pd.Timestamp('2029-01-01'))
plt.title("King County: Forecast with External Regressor Until 2028")
plt.xlabel("Date")
plt.ylabel("EV Registrations (Original Scale)")
plt.show()

fig2_king = final_model.plot_components(forecast_king)
plt.show()

# ----------------------------
# 9. Save the Final Model for King County
# ----------------------------
model_save_path_king = r"E:\Capstone\models\best_model_Prophet_ext_King.pkl"
os.makedirs(os.path.dirname(model_save_path_king), exist_ok=True)
with open(model_save_path_king, "wb") as f:
    pickle.dump(final_model, f)
print("King County Model saved at:", model_save_path_king)



# %%
# %% Automated County-Level Forecasting for Top 10 Counties

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import time

# ----------------------------
# Global Data Preparation (if not already loaded)
# ----------------------------
data = pd.read_csv('E:\\Capstone\\data\\EV\\processed\\ev_cleaned_data.csv')
data['transaction_date'] = pd.to_datetime(data['transaction_date'])

# Global aggregation for external regressor model: monthly EV registrations and average sale price
monthly_agg = data.groupby(data['transaction_date'].dt.to_period('M')).agg({
    'dol_vehicle_id': 'count',
    'sale_price': 'mean'
}).reset_index()
monthly_agg['transaction_date'] = monthly_agg['transaction_date'].dt.to_timestamp()
df_global_ext = monthly_agg.rename(columns={
    'transaction_date': 'ds',
    'dol_vehicle_id': 'y',
    'sale_price': 'avg_sale_price'
}).sort_values('ds')

# ----------------------------
# Define function for county-level forecasting
# ----------------------------
def forecast_county(county_name, data, forecast_periods=48, test_months=12, 
                    models_dir=r"E:\Capstone\models", viz_dir=r"E:\Capstone\visualizations"):
    """
    Processes data for a given county, performs hyperparameter tuning using cross-validation,
    trains a final Prophet model with external regressor (sale price) on log-transformed data,
    generates forecasts up to 2028, evaluates on the test set, creates and saves visualizations,
    and saves the county-specific model.
    
    Returns:
      forecast_df (pd.DataFrame): The complete forecast DataFrame.
      metrics (dict): Evaluation metrics (RMSE, MAE, MAPE).
    """
    # Filter data for the county and keep only necessary columns
    cols_to_keep = ['transaction_date', 'county', 'sale_price', 'dol_vehicle_id']
    county_data = data[data['county'] == county_name][cols_to_keep].copy()
    county_data['transaction_date'] = pd.to_datetime(county_data['transaction_date'])
    
    # Aggregate monthly data
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
    
    # Train-Test Split: last test_months as test
    train_size = len(df_county) - test_months
    train_df = df_county.iloc[:train_size].copy()
    test_df = df_county.iloc[train_size:].copy()
    
    # Log transformation of target variable on training data
    train_df['y'] = np.log1p(train_df['y'])
    
    # ----------------------------
    # Hyperparameter Tuning
    # ----------------------------
    param_grid = [
        {'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 1, 'n_changepoints': 25},
        {'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 1, 'n_changepoints': 50},
        {'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 1, 'n_changepoints': 75},
        {'seasonality_mode': 'multiplicative', 'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 5, 'n_changepoints': 25},
        {'seasonality_mode': 'multiplicative', 'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 5, 'n_changepoints': 50},
        {'seasonality_mode': 'multiplicative', 'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 5, 'n_changepoints': 75},
        {'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10, 'n_changepoints': 25},
        {'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10, 'n_changepoints': 50},
        {'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10, 'n_changepoints': 75},
    ]
    
    best_params = param_grid[0].copy()
    best_rmse = float('inf')
    
    for params in param_grid:
        try:
            current_model = Prophet(**params)
            current_model.add_regressor('avg_sale_price')
            current_model.fit(train_df)
            
            df_cv = cross_validation(current_model,
                                     initial='730 days',
                                     period='180 days',
                                     horizon='365 days',
                                     parallel="processes")
            df_p = performance_metrics(df_cv)
            current_rmse = df_p['rmse'].mean()
            print(f"Params: {params}, RMSE: {current_rmse:.2f}")
            
            if current_rmse < best_rmse:
                best_rmse = current_rmse
                best_params = params.copy()
        except Exception as e:
            print(f"⚠️ Failed with params {params}: {str(e)}")
            continue
    
    print(f"\n✅ Best Parameters for {county_name}: {best_params}")
    print(f"🏆 Validation RMSE: {best_rmse:.2f}")
    
    # ----------------------------
    # Final Model Training with Best Parameters
    # ----------------------------
    final_model = Prophet(**best_params)
    final_model.add_regressor('avg_sale_price')
    final_model.fit(train_df)
    
    # ----------------------------
    # Forecast External Regressor (Sale Price) for County
    # ----------------------------
    price_df = train_df[['ds', 'avg_sale_price']].rename(columns={'avg_sale_price': 'y'})
    price_model = Prophet(yearly_seasonality=True)
    price_model.fit(price_df)
    price_future = price_model.make_future_dataframe(periods=test_months+forecast_periods, freq='M')
    price_forecast = price_model.predict(price_future)[['ds', 'yhat']].rename(columns={'yhat': 'avg_sale_price'})
    
    # ----------------------------
    # Forecasting
    # ----------------------------
    total_steps = test_months + forecast_periods
    future_df = final_model.make_future_dataframe(periods=total_steps, freq='M')
    # Merge the external regressor forecast into future_df
    future_df = future_df.merge(price_forecast, on='ds', how='left')
    
    # Compute average monthly growth rate from training data
    train_df['growth_rate'] = train_df['avg_sale_price'].pct_change()
    clean_growth = train_df['growth_rate'].replace([np.inf, -np.inf], np.nan).dropna().clip(lower=-0.5, upper=0.5)
    avg_growth_rate = clean_growth.mean()
    if pd.isna(avg_growth_rate) or abs(avg_growth_rate) > 0.5:
        avg_growth_rate = 0.02
    print(f"\n{county_name} - Average monthly growth rate for sale price:", avg_growth_rate)
    
    last_price = train_df['avg_sale_price'].iloc[-1]
    na_future = future_df[future_df['avg_sale_price'].isna()]
    if not na_future.empty:
        first_future_idx = na_future.index[0]
        for i in range(first_future_idx, len(future_df)):
            months_ahead = i - first_future_idx + 1
            future_df.loc[future_df.index[i], 'avg_sale_price'] = last_price * ((1 + avg_growth_rate) ** months_ahead)
    else:
        print("No missing future sale price values to project.")
    
    forecast_df = final_model.predict(future_df)
    forecast_df['yhat_orig'] = np.expm1(forecast_df['yhat'])
    forecast_df['yhat_lower_orig'] = np.expm1(forecast_df['yhat_lower'])
    forecast_df['yhat_upper_orig'] = np.expm1(forecast_df['yhat_upper'])
    
    # ----------------------------
    # Evaluation on Test Set
    # ----------------------------
    test_eval = pd.merge_asof(test_df.sort_values('ds'),
                              forecast_df[['ds', 'yhat_orig']].sort_values('ds'),
                              on='ds',
                              tolerance=pd.Timedelta('30 days'))
    print(f"Number of matched test rows for {county_name} with 30-day tolerance:", len(test_eval))
    metrics = {}
    if not test_eval.empty:
        metrics['RMSE'] = np.sqrt(mean_squared_error(test_df['y'], test_eval['yhat_orig']))
        metrics['MAE'] = mean_absolute_error(test_df['y'], test_eval['yhat_orig'])
        metrics['MAPE'] = np.mean(np.abs((test_df['y'] - test_eval['yhat_orig']) / test_df['y']))
        print(f"\n{county_name} - Prophet with External Regressor Model:")
        print("Test RMSE:", metrics['RMSE'])
        print("Test MAE:", metrics['MAE'])
        print("Test MAPE:", metrics['MAPE'])
    else:
        print(f"No matching forecast dates found for test data in {county_name}.")
    
    # ----------------------------
    # Visualization and Saving Visuals
    # ----------------------------
    # Create a folder for visualizations for this county
    county_viz_dir = os.path.join(viz_dir, county_name)
    os.makedirs(county_viz_dir, exist_ok=True)
    
    # Forecast Plot
    plt.figure(figsize=(12,6))
    fig = final_model.plot(forecast_df)
    plt.xlim(pd.Timestamp('2023-01-01'), pd.Timestamp('2029-01-01'))
    plt.title(f"{county_name} County: Forecast with External Regressor Until 2028")
    plt.xlabel("Date")
    plt.ylabel("EV Registrations (Original Scale)")
    plt.savefig(os.path.join(county_viz_dir, f"{county_name}_forecast.png"))
    plt.show()
    
    # Components Plot
    fig2 = final_model.plot_components(forecast_df)
    plt.savefig(os.path.join(county_viz_dir, f"{county_name}_components.png"))
    plt.show()
    
    # ----------------------------
    # Save the County-Specific Model
    # ----------------------------
    model_save_path = os.path.join(models_dir, f"best_model_Prophet_ext_{county_name}.pkl")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, "wb") as f:
        pickle.dump(final_model, f)
    print(f"{county_name} Model saved at:", model_save_path)
    
    return forecast_df, metrics

# ----------------------------
# Run Forecasting for Specified Counties
# ----------------------------
counties = ["King", "Snohomish", "Pierce", "Clark", "Kitsap",
            "Thurston", "Spokane", "Whatcom", "Benton", "Skagit"]

county_forecasts = {}
county_metrics = {}

for county in counties:
    print(f"\nProcessing forecast for {county} County:")
    fc_df, met = forecast_county(county, data, forecast_periods=48, test_months=12, 
                                 models_dir=r"E:\Capstone\models", viz_dir=visualizations_folder)
    county_forecasts[county] = fc_df
    county_metrics[county] = met

for county, met in county_metrics.items():
    print(f"\n{county} County Metrics:")
    print(met)

# %%
