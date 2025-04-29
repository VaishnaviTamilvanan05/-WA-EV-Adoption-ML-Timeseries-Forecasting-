# %%
import pickle
import pandas as pd
import numpy as np
import os
from prophet import Prophet

# %%
model_path = r"E:\Capstone\models\best_model_Prophet_ext.pkl"
with open(model_path, "rb") as f:
    agg_model = pickle.load(f)

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

future = agg_model.make_future_dataframe(periods=48, freq='M')

last_avg_sale_price = df_prophet_ext[df_prophet_ext['ds'] <= '2024-12-31']['avg_sale_price'].iloc[-1]
future['avg_sale_price'] = last_avg_sale_price


print("Future DF columns:", future.columns)
print(future.head())

forecast = agg_model.predict(future)

forecast_period = forecast[(forecast['ds'] >= '2025-01-01') & (forecast['ds'] <= '2028-12-31')]
forecast_table = forecast_period[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
forecast_table.columns = ['Date', 'Forecast', 'Forecast_Lower', 'Forecast_Upper']

forecast_table['Forecast'] = forecast_table['Forecast'].round(0).astype(int)
forecast_table['Forecast_Lower'] = forecast_table['Forecast_Lower'].round(0).astype(int)
forecast_table['Forecast_Upper'] = forecast_table['Forecast_Upper'].round(0).astype(int)

output_dir = r"E:\Capstone\ev_count"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "aggregated_EV_count_2025_2028.csv")
forecast_table.to_csv(output_file, index=False)

print("Monthly EV count forecast CSV saved at:", output_file)
