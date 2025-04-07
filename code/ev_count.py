# %%
import pickle
import pandas as pd
import numpy as np
import os
from prophet import Prophet

# %%
# ----------------------------
# 1. Load the Saved Aggregated Model
# ----------------------------
model_path = r"E:\Capstone\models\best_model_Prophet_ext.pkl"
with open(model_path, "rb") as f:
    agg_model = pickle.load(f)

# ----------------------------
# 2. Load Global Aggregated Data (if not already available)
# ----------------------------
data = pd.read_csv('E:\\Capstone\\data\\EV\\processed\\ev_cleaned_data.csv')
data['transaction_date'] = pd.to_datetime(data['transaction_date'])

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
# 3. Create Future DataFrame for Forecasting
# ----------------------------
# Forecast 48 months ahead (assumes training ended in December 2024)
future = agg_model.make_future_dataframe(periods=48, freq='M')

# Instead of merging an external forecast, fill the "avg_sale_price" column with a constant.
# For example, we take the last observed average sale price from our historical data.
last_avg_sale_price = df_prophet_ext[df_prophet_ext['ds'] <= '2024-12-31']['avg_sale_price'].iloc[-1]
future['avg_sale_price'] = last_avg_sale_price

# Debug: Check that "avg_sale_price" now exists in future DataFrame
print("Future DF columns:", future.columns)
print(future.head())

# ----------------------------
# 4. Generate Forecast
# ----------------------------
forecast = agg_model.predict(future)

# ----------------------------
# 5. Filter Forecast for January 2025 to December 2028
# ----------------------------
forecast_period = forecast[(forecast['ds'] >= '2025-01-01') & (forecast['ds'] <= '2028-12-31')]
forecast_table = forecast_period[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
forecast_table.columns = ['Date', 'Forecast', 'Forecast_Lower', 'Forecast_Upper']

# Optionally, round and convert the predictions to integers for clarity
forecast_table['Forecast'] = forecast_table['Forecast'].round(0).astype(int)
forecast_table['Forecast_Lower'] = forecast_table['Forecast_Lower'].round(0).astype(int)
forecast_table['Forecast_Upper'] = forecast_table['Forecast_Upper'].round(0).astype(int)

# ----------------------------
# 6. Save the Forecast Table as CSV in the "ev_count" Folder
# ----------------------------
output_dir = r"E:\Capstone\ev_count"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "aggregated_EV_count_2025_2028.csv")
forecast_table.to_csv(output_file, index=False)

print("Monthly EV count forecast CSV saved at:", output_file)

# %%
import os
import pandas as pd
import numpy as np
from prophet import Prophet
import pickle

# ----------------------------
# Global Data Preparation (for extracting last observed avg_sale_price per county)
# ----------------------------
data = pd.read_csv(r"E:\Capstone\data\EV\processed\ev_cleaned_data.csv")
data['transaction_date'] = pd.to_datetime(data['transaction_date'])

# Aggregate data to monthly level and use month-end dates
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

# ----------------------------
# List of Counties and Directories
# ----------------------------
counties = ["King", "Snohomish", "Pierce", "Clark", "Kitsap",
            "Thurston", "Spokane", "Whatcom", "Benton", "Skagit"]

models_dir = r"E:\Capstone\models"
output_dir = r"E:\Capstone\ev_count"
os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# Function to extract monthly EV count forecasts from a saved county-level model
# ----------------------------
def extract_county_forecast(county):
    # Construct the file path for the saved county model
    model_filename = f"best_model_Prophet_ext_{county}.pkl"
    model_path = os.path.join(models_dir, model_filename)
    
    # Load the saved county-level model
    with open(model_path, "rb") as f:
        county_model = pickle.load(f)
    
    # Extract the county-specific historical aggregated data from df_global_ext
    df_county = df_global_ext[df_global_ext['ds'] <= '2024-12-31']
    df_county = df_county[df_county['county'] == county] if 'county' in df_global_ext.columns else df_county
    # If the global aggregated data does not have a "county" column, you may need to re-aggregate by county.
    # Alternatively, filter the original data for the county and aggregate:
    if df_county.empty:
        county_data = data[data['county'] == county].copy()
        county_data['transaction_date'] = pd.to_datetime(county_data['transaction_date'])
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
    
    # Get the last observed average sale price (assumed for training until 2024-12-31)
    last_avg_sale_price = df_county[df_county['ds'] <= '2024-12-31']['avg_sale_price'].iloc[-1]
    
    # ----------------------------
    # Create Future DataFrame for Forecasting
    # ----------------------------
    # Forecast 48 months ahead (assuming training ended on 2024-12-31, this covers Jan 2025 to Dec 2028)
    future = county_model.make_future_dataframe(periods=48, freq='M')
    
    # Fill the required external regressor with the last observed value
    future['avg_sale_price'] = last_avg_sale_price
    
    # ----------------------------
    # Generate Forecast and Back-Transform
    # ----------------------------
    forecast = county_model.predict(future)
    # Back-transform predictions (since models were trained on log-transformed data)
    forecast['yhat_orig'] = np.expm1(forecast['yhat'])
    forecast['yhat_lower_orig'] = np.expm1(forecast['yhat_lower'])
    forecast['yhat_upper_orig'] = np.expm1(forecast['yhat_upper'])
    
    # ----------------------------
    # Filter Forecast for January 2025 to December 2028
    # ----------------------------
    forecast_period = forecast[(forecast['ds'] >= '2025-01-01') & (forecast['ds'] <= '2028-12-31')]
    forecast_table = forecast_period[['ds', 'yhat_orig', 'yhat_lower_orig', 'yhat_upper_orig']].copy()
    forecast_table.columns = ['Date', 'Forecast', 'Forecast_Lower', 'Forecast_Upper']

    forecast_table['Date'] = pd.to_datetime(forecast_table['Date']).dt.strftime('%Y-%m-%d')
    
    # Optionally, round the values and convert to integers
    forecast_table['Forecast'] = forecast_table['Forecast'].round(0).astype(int)
    forecast_table['Forecast_Lower'] = forecast_table['Forecast_Lower'].round(0).astype(int)
    forecast_table['Forecast_Upper'] = forecast_table['Forecast_Upper'].round(0).astype(int)
    
    # Save the forecast table as CSV
    output_file = os.path.join(output_dir, f"{county}_EV_count_2025_2028.csv")
    forecast_table.to_csv(output_file, index=False)
    print(f"{county} forecast CSV saved at: {output_file}")
    
    return forecast_table

# ----------------------------
# Generate Forecasts for All Counties
# ----------------------------
all_forecasts = {}
for county in counties:
    print(f"\nProcessing forecast extraction for {county} County:")
    all_forecasts[county] = extract_county_forecast(county)


# %%
