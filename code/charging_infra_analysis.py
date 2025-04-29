# %%
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import folium
from folium.plugins import MarkerCluster
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

ev_data = pd.read_csv(r"E:\Capstone\data\EV\processed\ev_cleaned_data.csv")
charging_data = pd.read_csv(r"E:\Capstone\data\Charging\processed\WA_charging_cleaned_with_ports.csv")
county_gdf = gpd.read_file(r"E:\Capstone\data\tl_2024_us_county")
top_counties = ['king', 'snohomish', 'pierce', 'clark', 'kitsap',
                'thurston', 'spokane', 'whatcom', 'benton', 'skagit']

# %%

top_counties = ["king", "Snohomish", "Pierce", "Clark", "Kitsap",
                "Thurston", "Spokane", "Whatcom", "Benton", "Skagit"]


# Filter for Top 10 Counties
ev_data = ev_data[ev_data['county'].isin(top_counties)]

# Count EVs by county
ev_county = ev_data.groupby('county').size().reset_index(name='ev_count')

total_evs = len(ev_data)

total_stations = len(charging_data)

charging_data.columns = charging_data.columns.str.strip()

# Filter for Top 10 Counties
charging_data = charging_data[charging_data['county'].isin(top_counties)]

# Count stations by county
station_county = charging_data.groupby('county').size().reset_index(name='station_count')

# EV-to-Station Ratio
merged = pd.merge(ev_county, station_county, on='county', how='outer').fillna(0)
merged['ev_to_station_ratio'] = merged['ev_count'] / merged['station_count']
merged = merged.sort_values('ev_to_station_ratio', ascending=False)

statewide_ratio = total_evs / total_stations
print(f"Statewide EV-to-Station Ratio: {round(statewide_ratio):,} EVs per station")

print(round(merged))
merged.to_csv("top10_ev_to_station_ratio.csv", index=False)

merged_sorted = merged.sort_values('ev_to_station_ratio', ascending=False)

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

# DOE/NREL policy-based capacity
merged['l2_capacity'] = merged['l2_ports'] * 20
merged['dcfc_capacity'] = merged['dcfc_ports'] * 100
merged['total_capacity'] = merged['l2_capacity'] + merged['dcfc_capacity']

# Compare with actual EV count
merged['deficit_vs_policy'] = merged['ev_count'] - merged['total_capacity']
merged['status'] = merged['deficit_vs_policy'].apply(lambda x: 'Surplus' if x <= 0 else 'Deficit')

cols_to_int = ['ev_count', 'l2_ports', 'dcfc_ports', 'l2_capacity', 'dcfc_capacity', 'total_capacity', 'deficit_vs_policy']
merged[cols_to_int] = merged[cols_to_int].round(0).astype(int)

# Display
print(merged[['county', 'ev_count', 'l2_ports', 'dcfc_ports', 'total_capacity', 'deficit_vs_policy', 'status']])

merged.to_csv(r"E:\Capstone\outputs\ev_charging_policy_deficit_analysis.csv", index=False)
# %%
ev_data['postal_code'] = ev_data['postal_code'].astype(str).str.zfill(5)
charging_data['ZipCode'] = charging_data['ZipCode'].astype(str).str.zfill(5)

# EVs per ZIP
ev_by_zip = ev_data.groupby('postal_code').size().reset_index(name='ev_count')

# charging stations per ZIP
stations_by_zip = charging_data.groupby('ZipCode').size().reset_index(name='station_count')

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

print("ZIP-level analysis complete and saved.")


# %%

ev_data = pd.read_csv(r"E:\Capstone\data\EV\processed\ev_cleaned_data.csv")
ev_data['transaction_date'] = pd.to_datetime(ev_data['transaction_date'], errors='coerce')
ev_data = ev_data.dropna(subset=['transaction_date'])
ev_data['year'] = ev_data['transaction_date'].dt.year

# Historical total up to 2024
hist_total = ev_data[ev_data['year'] <= 2024].shape[0]

# forecasted EV data
forecast_df = pd.read_csv(r"E:\Capstone\ev_count\aggregated_EV_count_2025_2028.csv", parse_dates=['Date'])
forecast_df['year'] = forecast_df['Date'].dt.year

# forecast by year
forecast_by_year = forecast_df.groupby('year')['Forecast'].sum().reset_index()
forecast_by_year = forecast_by_year[forecast_by_year['year'].between(2025, 2028)]

rows = []
cumulative = hist_total

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

growth_df = pd.DataFrame(rows)

# Save & display
growth_df.to_csv(r"E:\Capstone\outputs\statewide_ev_growth_precise_format.csv", index=False)
print(growth_df)
print("\n Saved: statewide_ev_growth_precise_format.csv")


# %%
# Final total EVs at end of 2028
final_total = cumulative

# Overall growth vs 2024 baseline
overall_growth_pct = ((final_total - hist_total) / hist_total) * 100

print(f"\n Overall EV increase from end of 2024 to end of 2028: {final_total - hist_total:,} vehicles")
print(f"Percentage Increase: {overall_growth_pct:.2f}%")

# %%
ev_data_2024 = ev_data[ev_data['year'] <= 2024]

all_results = []

for county in top_counties:
    
    county_hist_total = ev_data_2024[ev_data_2024['county'].str.lower() == county.lower()].shape[0]
    
    forecast_path = fr"E:\Capstone\ev_count\{county}_EV_count_2025_2028.csv"
    if not os.path.exists(forecast_path):
        print(f"Missing forecast file for: {county}")
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

    total_forecast = forecast_by_year['Forecast'].sum()
    overall_pct = (total_forecast / county_hist_total * 100) if county_hist_total else 0
    all_results.append({
        'County': county,
        'Upto Previous Year Total': county_hist_total,
        'Current Year': '2025–2028',
        'New Vehicles Added': int(total_forecast),
        '% Increase': round(overall_pct, 2)
    })

county_growth_df = pd.DataFrame(all_results)

print(county_growth_df)

output_path = r"E:\Capstone\outputs\county_ev_growth_comparison.csv"
county_growth_df.to_csv(output_path, index=False)
print(f"\n County-wise EV growth analysis saved to:\n{output_path}")


# %%

ev_data = pd.read_csv("E:/Capstone/data/EV/processed/ev_cleaned_data.csv", parse_dates=['transaction_date'])
ev_data = ev_data.dropna(subset=['transaction_date'])
ev_data['year'] = ev_data['transaction_date'].dt.year
ev_data = ev_data[(ev_data['year'] >= 2017) & (ev_data['year'] <= 2024)]

ev_by_year = ev_data.groupby(['county', 'year']).size().reset_index(name='ev_count')

years = list(range(2017, 2025))
counties = ev_by_year['county'].unique()
full_index = pd.MultiIndex.from_product([counties, years], names=["county", "year"]).to_frame(index=False)

train_df = pd.merge(full_index, ev_by_year, on=["county", "year"], how="left").fillna(0)
train_df['ev_count'] = train_df['ev_count'].astype(int)

train_df['ev_growth'] = train_df.groupby('county')['ev_count'].pct_change().fillna(0)
train_df['ev_cumulative'] = train_df.groupby('county')['ev_count'].cumsum()


# %%
charging_data = pd.read_csv("E:/Capstone/data/Charging/processed/WA_charging_cleaned_with_ports.csv")
stations_2024 = charging_data.groupby('county').size().reset_index(name='existing_stations')

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
# === 1. ML Models on Historical Data ===
features = ['year', 'ev_count', 'ev_growth', 'ev_cumulative']
X = train_df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
y = train_df['stations_added']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression().fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_rmse = mean_squared_error(y_test, lr_pred, squared=False)
lr_r2 = r2_score(y_test, lr_pred)

# Ridge Regression (with GridSearchCV)
ridge_grid = GridSearchCV(Ridge(), {'alpha': [0.01, 0.1, 1, 10, 100]},
                          scoring='neg_root_mean_squared_error', cv=5)
ridge_grid.fit(X_train, y_train)
ridge_best = ridge_grid.best_estimator_
ridge_pred = ridge_best.predict(X_test)
ridge_rmse = mean_squared_error(y_test, ridge_pred, squared=False)
ridge_r2 = r2_score(y_test, ridge_pred)

# Lasso Regression
lasso_grid = GridSearchCV(Lasso(max_iter=10000), {'alpha': [0.01, 0.1, 1, 10, 100]},
                          scoring='neg_root_mean_squared_error', cv=5)
lasso_grid.fit(X_train, y_train)
lasso_best = lasso_grid.best_estimator_
lasso_pred = lasso_best.predict(X_test)
lasso_rmse = mean_squared_error(y_test, lasso_pred, squared=False)
lasso_r2 = r2_score(y_test, lasso_pred)

# Random Forest (Default)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_rmse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

# Tuned Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}
search = RandomizedSearchCV(RandomForestRegressor(random_state=42),
                            param_distributions=param_grid,
                            n_iter=20, cv=5, scoring='neg_root_mean_squared_error',
                            random_state=42, n_jobs=-1)
search.fit(X_train, y_train)
rf_tuned = search.best_estimator_
rf_tuned_pred = rf_tuned.predict(X_test)
rf_tuned_rmse = mean_squared_error(y_test, rf_tuned_pred, squared=False)
rf_tuned_r2 = r2_score(y_test, rf_tuned_pred)

# === 2. Model Comparison & Best Model Selection ===
results = {
    "Linear Regression": (lr, lr_rmse, lr_r2),
    f"Ridge (α={ridge_grid.best_params_['alpha']})": (ridge_best, ridge_rmse, ridge_r2),
    f"Lasso (α={lasso_grid.best_params_['alpha']})": (lasso_best, lasso_rmse, lasso_r2),
    "Random Forest (Default)": (rf, rf_rmse, rf_r2),
    "Random Forest (Tuned)": (rf_tuned, rf_tuned_rmse, rf_tuned_r2)
}

print("\n FINAL MODEL COMPARISON:")
print(f"{'Model':<35}{'RMSE':>10}{'R² Score':>15}")
for name, (model, rmse, r2) in results.items():
    print(f"{name:<35}{rmse:>10.2f}{r2:>15.2f}")

# Get best model (lowest RMSE)
best_model_name, (best_model, best_rmse, best_r2) = min(results.items(), key=lambda x: x[1][1])
print(f"\n Best Model: {best_model_name} with RMSE = {best_rmse:.2f}, R² = {best_r2:.2f}")

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
forecast_df['ev_growth'] = forecast_df['ev_growth'] / 100
forecast_df['county'] = forecast_df['county'].str.strip().str.lower()
forecast_df = forecast_df.sort_values(by=['county', 'year'])

# === 2. Predict Station Needs ===
features = ['year', 'ev_count', 'ev_growth', 'ev_cumulative']
X_forecast = forecast_df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
forecast_df['predicted_stations'] = best_model.predict(X_forecast).round().astype(int)

# === 3. Existing + Cumulative Stations ===

train_df['county'] = train_df['county'].str.strip().str.lower()
forecast_df['county'] = forecast_df['county'].str.strip().str.lower()

forecast_df['existing_stations'] = 0
forecast_df['cumulative_stations'] = 0

for county in forecast_df['county'].unique():
    mask = forecast_df['county'] == county

    if county in train_df['county'].values:
        base = train_df[train_df['county'] == county]['existing_stations'].max()
        if pd.isna(base): base = 0
    else:
        base = 0

    sub_df = forecast_df.loc[mask].copy()
    existing_list = []
    cumulative_list = []
    cumulative = 0

    for _, row in sub_df.iterrows():
        existing = base + cumulative
        cumulative += row['predicted_stations']
        existing_list.append(existing)
        cumulative_list.append(existing + row['predicted_stations'])

    forecast_df.loc[mask, 'existing_stations'] = existing_list
    forecast_df.loc[mask, 'cumulative_stations'] = cumulative_list


forecast_df['county'] = pd.Categorical(forecast_df['county'], categories=top_counties, ordered=True)
forecast_df = forecast_df.sort_values(by=['county', 'year'])

ordered_columns = [
    'county', 'year', 'ev_count', 'ev_cumulative', 'ev_growth',
    'existing_stations', 'predicted_stations', 'cumulative_stations'
]
forecast_df = forecast_df[ordered_columns]

# === 5. Save County-Level Forecast ===
forecast_df.to_csv("E:/Capstone/outputs/top_county_ev_station_predictions_2025_2028.csv", index=False)

# === 6. Create and Save Statewide Summary ===
state_summary = forecast_df.groupby('year').agg({
    'ev_count': 'sum',
    'predicted_stations': 'sum',
    'existing_stations': 'sum',
    'cumulative_stations': 'sum'
}).reset_index()

state_summary.to_csv("E:/Capstone/outputs/statewide_ev_station_predictions_2025_2028.csv", index=False)

# === 7. Done ===
print("\n ML Forecasts Saved:")
print(" top_county_ev_station_predictions_2025_2028.csv")
print(" statewide_ev_station_predictions_2025_2028.csv")


# %%

ev_data['year'] = ev_data['transaction_date'].dt.year
ev_data = ev_data[ev_data['year'] <= 2024]

# predicted county-level stations (2025–2028)
county_forecast = pd.read_csv("E:/Capstone/outputs/top_county_ev_station_predictions_2025_2028.csv")

# Get list of top counties
top_counties = county_forecast['county'].unique().tolist()

ev_data['county'] = ev_data['county'].str.strip().str.lower()
charging_data['county'] = charging_data['county'].str.strip().str.lower()
county_forecast['county'] = county_forecast['county'].str.strip().str.lower()
top_counties = county_forecast['county'].unique().tolist()

# EVs by ZIP and county
zip_ev = ev_data.groupby(['county', 'postal_code'])['year'].count().reset_index()
zip_ev.columns = ['county', 'zip', 'ev_count_2024']

# existing stations by ZIP and county
zip_stations = charging_data.groupby(['county', 'ZipCode']).size().reset_index(name='existing_stations')
zip_stations.columns = ['county', 'zip', 'existing_stations']

zip_base = pd.merge(zip_ev, zip_stations, on=['county', 'zip'], how='outer').fillna(0)
zip_base['ev_count_2024'] = zip_base['ev_count_2024'].astype(int)
zip_base['existing_stations'] = zip_base['existing_stations'].astype(int)

# Filter to top counties only
zip_base = zip_base[zip_base['county'].isin(top_counties)].copy()

# === PRIORITY SCORE ===

zip_base['priority_score'] = zip_base['ev_count_2024'] / (zip_base['existing_stations'] + 1)
zip_base['priority_score_norm'] = zip_base.groupby('county')['priority_score'].transform(
    lambda x: x / x.sum()
)

# ===  ALLOCATE PREDICTED COUNTY STATIONS TO ZIPs ===

# ZIPs by year using county forecast
years = [2025, 2026, 2027, 2028]
expanded_rows = []

for _, row in county_forecast.iterrows():
    county = row['county']
    year = row['year']
    total_stations = row['predicted_stations']
    
    zip_rows = zip_base[zip_base['county'] == county].copy()
    zip_rows['year'] = year
    zip_rows['predicted_stations'] = (zip_rows['priority_score_norm'] * total_stations).round().astype(int)
    expanded_rows.append(zip_rows)

zip_allocated = pd.concat(expanded_rows, ignore_index=True)

output_cols = [
    'county', 'zip', 'year',
    'ev_count_2024', 'existing_stations',
    'priority_score', 'priority_score_norm',
    'predicted_stations'
]

zip_allocated = zip_allocated[output_cols]
output_path = "E:/Capstone/outputs/zip_level_ev_station_predictions_2025_2028.csv"
zip_allocated.to_csv(output_path, index=False)

print(f" ZIP-level allocation saved to: {output_path}")


# %%

