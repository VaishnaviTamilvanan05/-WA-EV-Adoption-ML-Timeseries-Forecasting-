# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --------------------------------------
# Load and Prepare Global Data
# --------------------------------------
data = pd.read_csv(r"E:\Capstone\data\EV\processed\ev_cleaned_data.csv")
data['transaction_date'] = pd.to_datetime(data['transaction_date'])

# Monthly aggregation for external regressor (avg_sale_price)
monthly_agg = data.groupby(data['transaction_date'].dt.to_period('M')).agg({
    'dol_vehicle_id': 'count',
    'sale_price': 'mean'
}).reset_index()
monthly_agg['transaction_date'] = monthly_agg['transaction_date'].dt.to_timestamp(how='end')

df_global_ext = monthly_agg.rename(columns={
    'transaction_date': 'ds',
    'dol_vehicle_id': 'y',
    'sale_price': 'avg_sale_price'
}).sort_values('ds')

# Counties to Process
counties = ["king", "Snohomish", "Pierce", "Clark", "Kitsap", "Thurston", "Spokane", "Whatcom", "Benton", "Skagit"]

# Directories
models_dir = r"E:\Capstone\models"
viz_dir = r"E:\Capstone\visualizations"
output_dir = r"E:\Capstone\ev_count"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(viz_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# --------------------------------------
# County-Level Forecast Function
# --------------------------------------
def forecast_county(county_name, data, forecast_periods=48, test_months=12):
    cols_to_keep = ['transaction_date', 'county', 'sale_price', 'dol_vehicle_id']
    county_data = data[data['county'].str.lower() == county_name.lower()][cols_to_keep].copy()
    county_data['transaction_date'] = pd.to_datetime(county_data['transaction_date'])

    # Aggregate to monthly level
    county_monthly = county_data.groupby(county_data['transaction_date'].dt.to_period('M')).agg({
        'dol_vehicle_id': 'count',
        'sale_price': 'mean'
    }).reset_index()
    county_monthly['transaction_date'] = county_monthly['transaction_date'].dt.to_timestamp(how='end')

    df_county = county_monthly.rename(columns={
        'transaction_date': 'ds',
        'dol_vehicle_id': 'y',
        'sale_price': 'avg_sale_price'
    }).sort_values('ds')

    # Train-test split
    train_size = len(df_county) - test_months
    train_df = df_county.iloc[:train_size]
    test_df = df_county.iloc[train_size:]

    # Hyperparameter tuning
    param_grid = [
        {'seasonality_mode': 'additive', 'changepoint_prior_scale': cps, 'seasonality_prior_scale': sps, 'n_changepoints': ncp}
        for cps in [0.01, 0.05, 0.1]
        for sps in [1, 5, 10]
        for ncp in [25, 50, 75]
    ]

    best_params = None
    best_rmse = float('inf')

    for params in param_grid:
        try:
            m = Prophet(**params)
            m.add_regressor('avg_sale_price')
            m.fit(train_df)
            
            df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='365 days', parallel="processes")
            df_p = performance_metrics(df_cv)
            current_rmse = df_p['rmse'].mean()

            if current_rmse < best_rmse:
                best_rmse = current_rmse
                best_params = params.copy()
        except Exception as e:
            continue

    print(f"\n✅ Best Parameters for {county_name}: {best_params}")

    # Final model training
    final_model = Prophet(**best_params)
    final_model.add_regressor('avg_sale_price')
    final_model.fit(train_df)

    # Sale price forecasting
    price_model = Prophet(yearly_seasonality=True)
    price_model.fit(train_df[['ds', 'avg_sale_price']].rename(columns={'avg_sale_price': 'y'}))
    price_future = price_model.make_future_dataframe(periods=test_months + forecast_periods, freq='M')
    price_forecast = price_model.predict(price_future)[['ds', 'yhat']].rename(columns={'yhat': 'avg_sale_price'})

    # Future dataframe for EV forecast
    future_df = final_model.make_future_dataframe(periods=test_months + forecast_periods, freq='M')
    future_df = future_df.merge(price_forecast, on='ds', how='left')

    # Fill missing sale_price values if needed
    avg_growth_rate = train_df['avg_sale_price'].pct_change().replace([np.inf, -np.inf], np.nan).dropna().clip(-0.5, 0.5).mean()
    avg_growth_rate = 0.02 if pd.isna(avg_growth_rate) else avg_growth_rate
    last_price = train_df['avg_sale_price'].iloc[-1]
    future_df['avg_sale_price'] = future_df['avg_sale_price'].fillna(method='ffill').fillna(last_price)

    # Final forecast
    forecast_df = final_model.predict(future_df)
    forecast_df['yhat_orig'] = forecast_df['yhat']

    # Test set evaluation
    test_eval = pd.merge_asof(test_df.sort_values('ds'), forecast_df[['ds', 'yhat_orig']].sort_values('ds'), on='ds', tolerance=pd.Timedelta('30 days'))
    metrics = {}
    if not test_eval.empty:
        metrics['RMSE'] = np.sqrt(mean_squared_error(test_df['y'], test_eval['yhat_orig']))
        metrics['MAE'] = mean_absolute_error(test_df['y'], test_eval['yhat_orig'])
        metrics['MAPE'] = np.mean(np.abs((test_df['y'] - test_eval['yhat_orig']) / test_df['y']))

    # Save model
    model_path = os.path.join(models_dir, f"best_model_Prophet_ext_{county_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)


    os.makedirs(os.path.join(viz_dir, county_name), exist_ok=True)
    final_model.plot(forecast_df)
    plt.title(f"{county_name} County EV Forecast")
    plt.savefig(os.path.join(viz_dir, county_name, f"{county_name}_forecast.png"))
    plt.close()

    final_model.plot_components(forecast_df)
    plt.savefig(os.path.join(viz_dir, county_name, f"{county_name}_components.png"))
    plt.close()

    return forecast_df, metrics

# --------------------------------------
# Forecast Generation for All Counties
# --------------------------------------

county_forecasts = {}
county_metrics = {}

for county in counties:
    print(f"\n🚀 Forecasting {county} County...")
    fc_df, met = forecast_county(county, data)
    county_forecasts[county] = fc_df
    county_metrics[county] = met

# final metrics
for county, met in county_metrics.items():
    print(f"\n📊 {county} Metrics:")
    print(met)

# -------------------------------
#  County Forecasts (2025-2028)
# -------------------------------

def extract_county_forecast(county):
    model_path = os.path.join(models_dir, f"best_model_Prophet_ext_{county}.pkl")
    if not os.path.exists(model_path):
        print(f"❌ Model not found for {county}")
        return None

    with open(model_path, "rb") as f:
        county_model = pickle.load(f)

    future_dates = pd.date_range(start="2025-01-31", end="2028-12-31", freq='M')
    future = pd.DataFrame({'ds': future_dates})
    last_price = data[data['county'].str.lower() == county.lower()]['sale_price'].dropna().iloc[-1]
    future['avg_sale_price'] = last_price

    forecast = county_model.predict(future)
    forecast['yhat_orig'] = forecast['yhat']

    forecast_table = forecast[['ds', 'yhat_orig']].copy()
    forecast_table.columns = ['Date', 'Forecast']
    forecast_table['Date'] = forecast_table['Date'].dt.strftime('%Y-%m-%d')

    output_file = os.path.join(output_dir, f"{county}_EV_count_2025_2028.csv")
    forecast_table.to_csv(output_file, index=False)
    print(f"✅ Saved forecast: {output_file}")

    return forecast_table


for county in counties:
    print(f"\n Extracting forecast for {county}...")
    extract_county_forecast(county)


