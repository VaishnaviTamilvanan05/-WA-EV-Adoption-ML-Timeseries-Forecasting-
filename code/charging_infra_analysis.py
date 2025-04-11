# %%
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import folium
from folium.plugins import MarkerCluster

ev_data = pd.read_csv(r"E:\Capstone\data\EV\processed\ev_cleaned_data.csv")
charging_data = pd.read_csv(r"E:\Capstone\data\Charging\processed\WA_charging_cleaned_with_ports.csv")
county_gdf = gpd.read_file(r"E:\Capstone\data\tl_2024_us_county")

# %%

top_counties = ["king", "Snohomish", "Pierce", "Clark", "Kitsap",
                "Thurston", "Spokane", "Whatcom", "Benton", "Skagit"]

# Load EV registration data


# Filter for Top 10 Counties
ev_data = ev_data[ev_data['county'].isin(top_counties)]

# Count EVs by county
ev_county = ev_data.groupby('county').size().reset_index(name='ev_count')

total_evs = len(ev_data)


# Load Charging Station data


total_stations = len(charging_data)


# Make sure column names are clean (optional safety step)
charging_data.columns = charging_data.columns.str.strip()

# Filter charging stations for Top 10 Counties
charging_data = charging_data[charging_data['county'].isin(top_counties)]

# Count stations by county
station_county = charging_data.groupby('county').size().reset_index(name='station_count')

# Merge and calculate EV-to-Station Ratio
merged = pd.merge(ev_county, station_county, on='county', how='outer').fillna(0)
merged['ev_to_station_ratio'] = merged['ev_count'] / merged['station_count']
merged = merged.sort_values('ev_to_station_ratio', ascending=False)

statewide_ratio = total_evs / total_stations
print(f"Statewide EV-to-Station Ratio: {round(statewide_ratio):,} EVs per station")

print(round(merged))
# merged.to_csv("top10_ev_to_station_ratio.csv", index=False)

# %%
merged_sorted = merged.sort_values('ev_to_station_ratio', ascending=False)

# Plot
plt.figure(figsize=(12, 6))
plt.bar(merged_sorted['county'], merged_sorted['ev_to_station_ratio'], color='skyblue')
plt.axhline(y=482, color='red', linestyle='--', label='Statewide Avg (482)')
plt.title('EV-to-Charging Station Ratio by County')
plt.xlabel('County')
plt.ylabel('EVs per Charging Station')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
# %%
ev_data = ev_data[ev_data['county'].isin(top_counties)]
ev_county = ev_data.groupby('county').size().reset_index(name='ev_count')


charging_data.columns = charging_data.columns.str.strip()
charging_data = charging_data[charging_data['county'].isin(top_counties)]

ports_county = charging_data.groupby('county')[['l2_ports', 'dcfc_ports']].sum().reset_index()

merged = pd.merge(ev_county, ports_county, on='county', how='outer').fillna(0)

# Apply DOE/NREL policy-based capacity
merged['l2_capacity'] = merged['l2_ports'] * 20
merged['dcfc_capacity'] = merged['dcfc_ports'] * 100
merged['total_capacity'] = merged['l2_capacity'] + merged['dcfc_capacity']

# Compare with actual EV count
merged['deficit_vs_policy'] = merged['ev_count'] - merged['total_capacity']
merged['status'] = merged['deficit_vs_policy'].apply(lambda x: 'Surplus' if x <= 0 else 'Deficit')

# Optional: Round for cleaner display
cols_to_int = ['ev_count', 'l2_ports', 'dcfc_ports', 'l2_capacity', 'dcfc_capacity', 'total_capacity', 'deficit_vs_policy']
merged[cols_to_int] = merged[cols_to_int].round(0).astype(int)

# Display
print(merged[['county', 'ev_count', 'l2_ports', 'dcfc_ports', 'total_capacity', 'deficit_vs_policy', 'status']])

merged.to_csv(r"E:\Capstone\outputs\ev_charging_policy_deficit_analysis.csv", index=False)


# %%
ev_data['postal_code'] = ev_data['postal_code'].astype(str).str.zfill(5)
charging_data['ZipCode'] = charging_data['ZipCode'].astype(str).str.zfill(5)

# Count EVs per ZIP
ev_by_zip = ev_data.groupby('postal_code').size().reset_index(name='ev_count')

# Count charging stations per ZIP
stations_by_zip = charging_data.groupby('ZipCode').size().reset_index(name='station_count')

# Merge both
zip_merged = pd.merge(ev_by_zip, stations_by_zip, left_on='postal_code', right_on='ZipCode', how='outer').fillna(0)
zip_merged['ev_count'] = zip_merged['ev_count'].astype(int)
zip_merged['station_count'] = zip_merged['station_count'].astype(int)

# EV-to-Station Ratio
zip_merged['ev_to_station_ratio'] = (zip_merged['ev_count'] / zip_merged['station_count']).replace([float('inf'), float('nan')], 0).round(2)

# ZIPs with EVs but 0 stations
no_station_zips = zip_merged[(zip_merged['ev_count'] > 0) & (zip_merged['station_count'] == 0)]

# Save outputs
zip_merged.to_csv(r"E:\Capstone\outputs\zip_ev_station_ratio.csv", index=False)
no_station_zips.to_csv(r"E:\Capstone\outputs\zip_with_ev_no_station.csv", index=False)

print("✅ ZIP-level analysis complete and saved.")


# %%

ev_data = pd.read_csv(r"E:\Capstone\data\EV\processed\ev_cleaned_data.csv")
ev_data['transaction_date'] = pd.to_datetime(ev_data['transaction_date'], errors='coerce')
ev_data = ev_data.dropna(subset=['transaction_date'])
ev_data['year'] = ev_data['transaction_date'].dt.year

# Historical total up to 2024
hist_total = ev_data[ev_data['year'] <= 2024].shape[0]

# Load forecasted EV data
forecast_df = pd.read_csv(r"E:\Capstone\ev_count\aggregated_EV_count_2025_2028.csv", parse_dates=['Date'])
forecast_df['year'] = forecast_df['Date'].dt.year

# Group forecast by year
forecast_by_year = forecast_df.groupby('year')['Forecast'].sum().reset_index()
forecast_by_year = forecast_by_year[forecast_by_year['year'].between(2025, 2028)]

# Initialize variables
rows = []
cumulative = hist_total

# Build growth table year by year
for _, row in forecast_by_year.iterrows():
    year = row['year']
    added = row['Forecast']
    pct_increase = (added / cumulative) * 100 if cumulative else 0
    rows.append({
        'Upto Previous Year Total': int(cumulative),
        'Current Year': year,
        'New Vehicles Added': int(added),
        '% Increase': round(pct_increase, 2)
    })
    cumulative += added

# Final DataFrame
growth_df = pd.DataFrame(rows)

# Save & display
growth_df.to_csv(r"E:\Capstone\outputs\statewide_ev_growth_precise_format.csv", index=False)
print(growth_df)
print("\n✅ Saved: statewide_ev_growth_precise_format.csv")


# %%
# Final total EVs at end of 2028
final_total = cumulative

# Overall growth vs 2024 baseline
overall_growth_pct = ((final_total - hist_total) / hist_total) * 100

print(f"\n📈 Overall EV increase from end of 2024 to end of 2028: {final_total - hist_total:,} vehicles")
print(f"🔼 Percentage Increase: {overall_growth_pct:.2f}%")


# %%
ev_data_2024 = ev_data[ev_data['year'] <= 2024]

all_results = []

# Loop through counties
for county in top_counties:
    # Historical count up to 2024
    county_hist_total = ev_data_2024[ev_data_2024['county'].str.lower() == county.lower()].shape[0]
    
    # Load forecast file
    forecast_path = fr"E:\Capstone\ev_count\{county}_EV_count_2025_2028.csv"
    if not os.path.exists(forecast_path):
        print(f"⚠️ Missing forecast file for: {county}")
        continue

    forecast_df = pd.read_csv(forecast_path, parse_dates=['Date'])
    forecast_df['year'] = forecast_df['Date'].dt.year
    forecast_by_year = forecast_df.groupby('year')['Forecast'].sum().reset_index()
    
    cumulative = county_hist_total
    for _, row in forecast_by_year.iterrows():
        year = row['year']
        added = row['Forecast']
        pct_increase = (added / cumulative) * 100 if cumulative else 0
        all_results.append({
            'County': county,
            'Upto Previous Year Total': int(cumulative),
            'Current Year': year,
            'New Vehicles Added': int(added),
            '% Increase': round(pct_increase, 2)
        })
        cumulative += added

    # Overall summary row
    total_forecast = forecast_by_year['Forecast'].sum()
    overall_pct = (total_forecast / county_hist_total * 100) if county_hist_total else 0
    all_results.append({
        'County': county,
        'Upto Previous Year Total': county_hist_total,
        'Current Year': '2025–2028',
        'New Vehicles Added': int(total_forecast),
        '% Increase': round(overall_pct, 2)
    })

# Create result DataFrame
county_growth_df = pd.DataFrame(all_results)

print(county_growth_df)

# Save
output_path = r"E:\Capstone\outputs\county_ev_growth_comparison.csv"
county_growth_df.to_csv(output_path, index=False)
print(f"\n✅ County-wise EV growth analysis saved to:\n{output_path}")
# %%
# -------------------------------------------------------------------------------------#

# %%
import pandas as pd

# Load EV registration data
ev_data = pd.read_csv("E:/Capstone/data/EV/processed/ev_cleaned_data.csv", parse_dates=['transaction_date'])
ev_data = ev_data.dropna(subset=['transaction_date'])
ev_data['year'] = ev_data['transaction_date'].dt.year
ev_data = ev_data[(ev_data['year'] >= 2017) & (ev_data['year'] <= 2024)]

# Aggregate by county and year
ev_by_year = ev_data.groupby(['county', 'year']).size().reset_index(name='ev_count')

# Fill missing combinations
years = list(range(2017, 2025))
counties = ev_by_year['county'].unique()
full_index = pd.MultiIndex.from_product([counties, years], names=["county", "year"]).to_frame(index=False)

train_df = pd.merge(full_index, ev_by_year, on=["county", "year"], how="left").fillna(0)
train_df['ev_count'] = train_df['ev_count'].astype(int)

# EV growth & cumulative
train_df['ev_growth'] = train_df.groupby('county')['ev_count'].pct_change().fillna(0)
train_df['ev_cumulative'] = train_df.groupby('county')['ev_count'].cumsum()


# %%
charging_data = pd.read_csv("E:/Capstone/data/Charging/processed/WA_charging_cleaned_with_ports.csv")
stations_2024 = charging_data.groupby('county').size().reset_index(name='existing_stations')

# Merge and simulate stations_added
train_df = pd.merge(train_df, stations_2024, on='county', how='left').fillna(0)
train_df['existing_stations'] = train_df['existing_stations'].astype(int)

def simulate_station_growth(group):
    total_stations = group['existing_stations'].iloc[-1]
    total_ev = group['ev_cumulative'].iloc[-1]
    if total_stations == 0 or total_ev == 0:
        group['stations_added'] = 0
    else:
        group['stations_needed'] = (group['ev_cumulative'] / total_ev) * total_stations
        group['stations_added'] = group['stations_needed'].diff().fillna(group['stations_needed'])
    return group

train_df = train_df.groupby('county').apply(simulate_station_growth).reset_index(drop=True)
train_df['stations_added'] = train_df['stations_added'].clip(lower=0).round().astype(int)


# %%

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

features = ['year', 'ev_count', 'ev_growth', 'ev_cumulative']
X = train_df[features]
y = train_df['stations_added']

# Clean X
X = X.replace([float('inf'), float('-inf')], pd.NA)
X = X.fillna(0)

# Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Baseline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

baseline = LinearRegression().fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

# Random Forest
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Evaluation
print("\n📊 Model Comparison:")
print(f"[Linear Regression] RMSE: {mean_squared_error(y_test, baseline_pred, squared=False):.2f}, R²: {r2_score(y_test, baseline_pred):.2f}")
print(f"[Random Forest]     RMSE: {mean_squared_error(y_test, rf_pred, squared=False):.2f}, R²: {r2_score(y_test, rf_pred):.2f}")


# %%

forecast_df = pd.read_csv("E:/Capstone/outputs/county_ev_growth_comparison.csv")
forecast_df = forecast_df[forecast_df['Current Year'].astype(str).str.isnumeric()]
forecast_df['year'] = forecast_df['Current Year'].astype(int)
forecast_df.rename(columns={
    'County': 'county',
    'New Vehicles Added': 'ev_count',
    'Upto Previous Year Total': 'ev_cumulative',
    '% Increase': 'ev_growth'
}, inplace=True)

forecast_df = forecast_df[['county', 'year', 'ev_count', 'ev_cumulative', 'ev_growth']]
forecast_df['ev_growth'] = forecast_df['ev_growth'] / 100  # convert from % to decimal

# Predict stations
X_forecast = forecast_df[features]
forecast_df['predicted_stations_added'] = rf_model.predict(X_forecast).round().astype(int)


# %%

# Carry over 2024 baseline
forecast_df['existing_stations'] = 0
for county in forecast_df['county'].unique():
    base = train_df[train_df['county'] == county]['existing_stations'].max()
    mask = forecast_df['county'] == county
    forecast_df.loc[mask, 'existing_stations'] = forecast_df.loc[mask, 'predicted_stations_added'].cumsum() + base

# Save county-wise results
forecast_df.to_csv("E:/Capstone/outputs/top_county_ev_station_predictions_2025_2028.csv", index=False)

# Create statewide summary
state_summary = forecast_df.groupby('year').agg({
    'ev_count': 'sum',
    'predicted_stations_added': 'sum',
    'existing_stations': 'sum'
}).reset_index()

state_summary.to_csv("E:/Capstone/outputs/statewide_ev_station_predictions_2025_2028.csv", index=False)
print("\n✅ ML Forecasts Saved:")
print("🔹 top_county_ev_station_predictions_2025_2028.csv")
print("🔹 statewide_ev_station_predictions_2025_2028.csv")


# %%
