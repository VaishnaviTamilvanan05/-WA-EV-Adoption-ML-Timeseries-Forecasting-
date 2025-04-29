# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from pysal.explore import esda
from pysal.lib import weights
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib as mpl
from shapely.geometry import Point
import folium
from folium.plugins import MarkerCluster

# Paths
EV_DATA_PATH = r"E:/Capstone/data/EV/processed/ev_cleaned_data.csv"
CHARGING_DATA_PATH = r"E:/Capstone/data/Charging/processed/WA_charging_cleaned_with_ports.csv"
COUNTY_SHP_PATH = r"E:/Capstone/data/tl_2024_us_county/tl_2024_us_county.shp"
FORECAST_PATH = r"E:/Capstone/outputs/top_county_ev_station_predictions_2025_2028.csv"

# Load Data
ev_data = pd.read_csv(EV_DATA_PATH)
ev_data['transaction_date'] = pd.to_datetime(ev_data['transaction_date'])
ev_data['county'] = ev_data['county'].str.lower().str.strip()
charging_data = pd.read_csv(CHARGING_DATA_PATH)
charging_data['county'] = charging_data['county'].str.lower().str.strip()
counties = gpd.read_file(COUNTY_SHP_PATH)
counties['county'] = counties['NAME'].str.lower().str.strip()
counties_wa = counties[counties['STATEFP'] == '53'].copy()
forecast_df = pd.read_csv(FORECAST_PATH)
forecast_df['county'] = forecast_df['county'].str.lower().str.strip()

# %%
# Hypothesis Testing

# Charging infrastructure impact
ev_county = ev_data.groupby('county').size().reset_index(name='ev_count')
charging_county = charging_data.groupby('county').size().reset_index(name='charging_count')
merged = pd.merge(ev_county, charging_county, on='county', how='left').fillna(0)
low_group = merged[merged['charging_count'] == 0]['ev_count']
high_group = merged[merged['charging_count'] > 0]['ev_count']
t_stat, p_val = ttest_ind(high_group, low_group, equal_var=False)
print("Charging vs No Charging t-test:", t_stat, p_val)

# Temporal impact
ev_data['year_month'] = ev_data['transaction_date'].dt.to_period('M')
monthly_counts = ev_data.groupby('year_month').size().reset_index(name='ev_count')
monthly_counts['date'] = pd.to_datetime(monthly_counts['year_month'].astype(str))
before = monthly_counts[monthly_counts['date'] < '2020-01-01']['ev_count']
after = monthly_counts[monthly_counts['date'] >= '2020-01-01']['ev_count']
t_stat, p_val = ttest_ind(before, after, equal_var=False)
print("Temporal T-test:", t_stat, p_val)

# Urban vs Rural EV Adoption
urban_counties = ["king", "pierce", "snohomish", "spokane", "clark", "benton", "franklin", "yakima", "douglas"]
ev_data['urban_rural'] = ev_data['county'].apply(lambda x: 'urban' if x in urban_counties else 'rural')
urban = ev_data[ev_data['urban_rural'] == 'urban'].groupby('year_month').size().reset_index(name='ev_count')
rural = ev_data[ev_data['urban_rural'] == 'rural'].groupby('year_month').size().reset_index(name='ev_count')
t_stat, p_val = ttest_ind(urban['ev_count'], rural['ev_count'], equal_var=False)
print("Urban vs Rural t-test:", t_stat, p_val)

# Moran's I
ev_count = ev_data.groupby('county').size().reset_index(name='ev_count')
gdf = counties_wa.merge(ev_count, on='county', how='left').fillna(0)
w = weights.Queen.from_dataframe(gdf)
w.transform = 'r'
moran = esda.Moran(gdf['ev_count'], w)
print("Moran's I:", moran.I, "p-value:", moran.p_sim)

# Sale price impact
ev_data['sale_price'] = pd.to_numeric(ev_data['sale_price'], errors='coerce')
sale_prices = ev_data.groupby('county')['sale_price'].mean().reset_index(name='avg_sale_price')
sale_merge = pd.merge(sale_prices, charging_county, on='county', how='left').fillna(0)
no_charging = sale_merge[sale_merge['charging_count'] == 0]['avg_sale_price']
with_charging = sale_merge[sale_merge['charging_count'] > 0]['avg_sale_price']
t_stat, p_val = ttest_ind(with_charging, no_charging, equal_var=False)
print("Sale price t-test:", t_stat, p_val)

# %%
# Regression Analysis

# ARIMA on monthly counts
monthly_counts = ev_data.groupby(ev_data['transaction_date'].dt.to_period('M')).size()
monthly_counts.index = pd.to_datetime(monthly_counts.index.astype(str))
model = sm.tsa.ARIMA(monthly_counts, order=(1,1,1))
results = model.fit()
print(results.summary())

# OLS: EV counts ~ charging stations
ev_agg = ev_data.groupby('county').size().reset_index(name='ev_count')
charging_agg = charging_data.groupby('county').size().reset_index(name='charging_count')
merged_agg = pd.merge(ev_agg, charging_agg, on='county', how='left').fillna(0)
X = sm.add_constant(merged_agg['charging_count'])
y = merged_agg['ev_count']
ols_model = sm.OLS(y, X).fit()
print(ols_model.summary())

# Extended Multivariate Regression
ev_features = ev_data.groupby('county').agg(
    ev_count=('vin__1_10_', 'count'),
    avg_model_year=('model_year', 'mean'),
    avg_electric_range=('electric_range', 'mean'),
    avg_sale_price=('sale_price', 'mean')
).reset_index()
merged_ext = pd.merge(ev_features, charging_county, on='county', how='left').fillna(0)
merged_ext['log_ev_count'] = np.log1p(merged_ext['ev_count'])
merged_ext['log_charging'] = np.log1p(merged_ext['charging_count'])
merged_ext['log_sale_price'] = np.log1p(merged_ext['avg_sale_price'])
scaler = StandardScaler()
merged_ext[['avg_model_year', 'avg_electric_range', 'log_charging', 'log_sale_price']] = scaler.fit_transform(
    merged_ext[['avg_model_year', 'avg_electric_range', 'log_charging', 'log_sale_price']]
)
formula = "log_ev_count ~ log_charging + avg_model_year + avg_electric_range + log_sale_price"
extended_model = smf.ols(formula, data=merged_ext).fit()
print(extended_model.summary())

# %%
# Geospatial Pre: Current State (2024 EVs and Charging Stations)

# EV Registrations 2024
ev_data['year'] = ev_data['transaction_date'].dt.year
ev_2024 = ev_data[ev_data['year'] <= 2024].groupby('county').size().reset_index(name='ev_count')
choropleth_2024 = counties_wa.merge(ev_2024, on='county', how='left').fillna(0)
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
choropleth_2024.plot(column='ev_count', cmap='Blues', legend=True, ax=ax, edgecolor='black')
ax.set_title("Total EV Registrations by County (up to 2024)")
plt.axis('off')
plt.show()

# Charging Stations 2024
stations = charging_data.groupby('county').size().reset_index(name='charging_count')
choropleth_stations = counties_wa.merge(stations, on='county', how='left').fillna(0)
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
choropleth_stations.plot(column='charging_count', cmap='Purples', legend=True, ax=ax, edgecolor='black')
ax.set_title("Charging Stations by County (as of 2024)")
plt.axis('off')
plt.show()

# %%
# Post-EDA: Forecasted EVs 2025-2028 and Predicted Stations

# Forecasted EV Growth 2025-2028
forecast_agg = forecast_df.groupby('county')['ev_count'].sum().reset_index()
choropleth_forecast = counties_wa.merge(forecast_agg, on='county', how='left').fillna(0)
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
choropleth_forecast.plot(column='ev_count', cmap='Oranges', legend=True, ax=ax, edgecolor='black')
ax.set_title("Forecasted EV Registrations by County (2025-2028)")
plt.axis('off')
plt.show()

# Predicted New Charging Stations 2025-2028
predicted_agg = forecast_df.groupby('county')['predicted_stations'].sum().reset_index()
choropleth_predicted = counties_wa.merge(predicted_agg, on='county', how='left').fillna(0)
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
choropleth_predicted.plot(column='predicted_stations', cmap='Reds', legend=True, ax=ax, edgecolor='black')
ax.set_title("Predicted New Charging Stations (2025-2028)")
plt.axis('off')
plt.show()
