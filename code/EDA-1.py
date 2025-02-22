#%%
print("""EDA Summary:

#######################################
# EV Adoption & Charging Infrastructure Analysis
# Forecasting EV Growth & Charging Demand (2025-2028)
#######################################

## Exploratory Data Analysis (EDA):
- EV Adoption Trend (2010-2024) - Trend & Growth Rate
- EV Adoption Trend by Vehicle Type - Category Comparison
- Top 10 Counties with Highest EV Registrations - Geographical Distribution
- Top 10 EV Manufacturers by Registration - Manufacturer Analysis
- EV Sale Price Trend Over Time - Pricing Impact on Adoption
- Mileage Distribution: New vs. Used EVs - Comparative Analysis
- Top 10 Cities with Highest Charging Demand - Infrastructure Readiness

## 📈 Statistical & Forecasting Analysis:
- Outlier Detection: High-Demand Charging Stations
- Correlation Analysis: EV Growth vs. Charging Infrastructure
- PCA: Feature Importance in EV Adoption Trends
- Time Series Analysis: Seasonal Decomposition of EV Registrations
- SARIMA Forecast (2025-2028): Predicting EV Growth & Charging Demand
""")


#%%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import zscore
from scipy.stats import pearsonr
from tabulate import tabulate 
import geopandas as gpd

# Load the datasets

ev_registration_data_path = "/Users/vaishnavitamilvanan/Downloads/Capstone Project/-WA-EV-Adoption-ML-Timeseries-Forecasting-/data/ev_cleaned_data.csv"
charging_data_path = "/Users/vaishnavitamilvanan/Downloads/Capstone Project/-WA-EV-Adoption-ML-Timeseries-Forecasting-/data/WA_charging_cleaned_data.csv"
ev_df = pd.read_csv(ev_registration_data_path)
charging_df = pd.read_csv(charging_data_path)


#%%

####################################
#EV Adoption Trend Analysis
####################################


# Convert transaction_date to datetime format
ev_df['transaction_date'] = pd.to_datetime(ev_df['transaction_date'], errors='coerce')

# Extract year from the transaction date and ensure it's an integer
ev_df['year'] = ev_df['transaction_date'].dt.year.astype('Int64')

# Aggregate EV registrations by year
ev_trend_yearly = ev_df.groupby('year').size().reset_index(name='ev_count')


ev_trend_yearly['year'] = ev_trend_yearly['year'].astype(int)

# Aggregate average sale price per year (ensuring integer year format)
price_trend = ev_df.groupby('year')['sale_price'].mean().reset_index()
price_trend['year'] = price_trend['year'].astype(int)

# Print EV Adoption Trend in a formatted table
print("\n EV Adoption Trend (2010-2024):")
print(tabulate(ev_trend_yearly, headers=["Year", "EV Count", "Growth Rate (%)"], tablefmt="pretty", showindex=False))


# Plot EV Adoption Trend
plt.figure(figsize=(12, 6))
plt.plot(ev_trend_yearly['year'], ev_trend_yearly['ev_count'], marker='o', linestyle='-', color='b', linewidth=2, label="EV Registrations")

# Add labels to key data points
for i, txt in enumerate(ev_trend_yearly['ev_count']):
    plt.annotate(txt, (ev_trend_yearly['year'][i], ev_trend_yearly['ev_count'][i]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=9)

plt.xticks(ev_trend_yearly['year'], rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("Year", fontsize=12, fontweight='bold')
plt.ylabel("Number of EV Registrations", fontsize=12, fontweight='bold')
plt.title("EV Adoption Trend (2010-2024)", fontsize=14, fontweight='bold', color='darkblue')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Soft gridlines

# Show  plot
plt.show()


#%%
##############################################
# EV Adoption Trend Analysis by Vehicle Type
##############################################


# Filter vehicle type registrations over time
ev_type_trend = ev_df.groupby(['year', 'clean_alternative_fuel_vehicle_type']).size().unstack()

# Calculate year-over-year growth rate for each type
ev_type_trend_growth = ev_type_trend.pct_change(fill_method=None) * 100  # Fixing deprecated warning

# Print  Adoption Trend in a formatted table
print("\n EV Adoption Trend by Vehicle Type:")
print(tabulate(ev_type_trend, headers='keys', tablefmt='pretty', showindex=True))

# Plot Adoption Trends Over Time
plt.figure(figsize=(12, 6))
for col in ev_type_trend.columns:
    plt.plot(ev_type_trend.index, ev_type_trend[col], marker='o', linestyle='-', label=col)

plt.xlabel("Year", fontsize=12, fontweight='bold')
plt.ylabel("Number of EV Registrations", fontsize=12, fontweight='bold')
plt.title("EV Adoption Trend by Vehicle Type", fontsize=14, fontweight='bold', color='darkblue')
plt.legend(title="EV Type")
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Soft gridlines

# Show plot
plt.show()


#%%
####################################
# Distribution of EVs by County & City
####################################

# Aggregate EV registrations by county and city
ev_distribution = ev_df.groupby(['county', 'city']).size().reset_index(name='ev_count')

# Sort by highest EV registrations
ev_distribution_sorted = ev_distribution.sort_values(by='ev_count', ascending=False)

# Add an index column
ev_distribution_sorted.reset_index(drop=True, inplace=True)
ev_distribution_sorted.index += 1  # Start index from 1

# Print EV distribution table using tabulate with an index column
print("\n Distribution of EVs by County & City:")
print(tabulate(ev_distribution_sorted.head(20), headers=['Index', 'County', 'City', 'EV Count'], tablefmt='pretty', showindex=True))

# Get top 10 counties with highest EV registrations
top_counties = ev_df.groupby('county').size().reset_index(name='ev_count').sort_values(by='ev_count', ascending=False).head(10)

# Use a distinct color for each county (Seaborn's "deep" palette)
palette = sns.color_palette("deep", n_colors=len(top_counties))

# Create bar chart with distinct colors for each county
plt.figure(figsize=(16, 8))
bars = plt.barh(top_counties['county'], top_counties['ev_count'], color=palette)

# Add total count labels on bars
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', 
             va='center', ha='left', fontsize=10, fontweight='bold', color='black')

plt.xlabel("Number of EV Registrations", fontsize=12, fontweight='bold')
plt.ylabel("County", fontsize=12, fontweight='bold')
plt.title("Top 10 Counties with Highest EV Registrations", fontsize=14, fontweight='bold', color='darkblue')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Show plot
plt.show()

#%%
####################################
# EV Model & Manufacturer Trends
####################################


# Aggregate EV registrations by manufacturer (make) and model
ev_make_model = ev_df.groupby(['make', 'model']).size().reset_index(name='ev_count')

# Sort by highest EV registrations
ev_make_model_sorted = ev_make_model.sort_values(by='ev_count', ascending=False)

# Add an index column
ev_make_model_sorted.reset_index(drop=True, inplace=True)
ev_make_model_sorted.index += 1  # Start index from 1

# Print EV model and manufacturer distribution table
print("\n EV Model & Manufacturer Trends:")
print(tabulate(ev_make_model_sorted.head(20), headers=['Index', 'Make', 'Model', 'EV Count'], tablefmt='pretty', showindex=True))

# Get top 10 manufacturers with highest EV registrations
top_makes = ev_df.groupby('make').size().reset_index(name='ev_count').sort_values(by='ev_count', ascending=False).head(10)

# Use a distinct color for each manufacturer (Seaborn's "muted" palette)
palette = sns.color_palette("muted", n_colors=len(top_makes))

# Create bar chart with distinct colors for each manufacturer
plt.figure(figsize=(16, 8))
bars = plt.barh(top_makes['make'], top_makes['ev_count'], color=palette)

# Add total count labels on bars
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', 
             va='center', ha='left', fontsize=10, fontweight='bold', color='black')

plt.xlabel("Number of EV Registrations", fontsize=12, fontweight='bold')
plt.ylabel("Manufacturer", fontsize=12, fontweight='bold')
plt.title("Top 10 EV Manufacturers by Registration", fontsize=14, fontweight='bold', color='darkblue')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Show plot
plt.show()




#%%
####################################
# Price & Incentive Impact on EV Adoption
####################################
# Ensure transaction_date is in datetime format
ev_df['transaction_date'] = pd.to_datetime(ev_df['transaction_date'], errors='coerce')

# Extract year from transaction_date and ensure it's an integer
ev_df['year'] = ev_df['transaction_date'].dt.year.astype('Int64')

# Filter out zero or null sale prices
df_filtered = ev_df[ev_df['sale_price'] > 0].copy()

# Aggregate average sale price per year
price_trend = df_filtered.groupby('year', as_index=False)['sale_price'].mean()

# Convert year column to integer (handling potential NaNs safely)
price_trend['year'] = price_trend['year'].astype('Int64')

# Round sale price values for better readability
price_trend['sale_price'] = price_trend['sale_price'].round(2)

# Print average sale price trend using tabulate
print("\n EV Sale Price Trend Over Time:")
print(tabulate(price_trend, headers=['Year', 'Average Sale Price ($)'], tablefmt='pretty', showindex=False))

# Plot sale price trend using a line chart with markers
plt.figure(figsize=(12, 6))
plt.plot(price_trend['year'], price_trend['sale_price'], marker='o', linestyle='-', color='crimson', linewidth=2)

# Add labels to key data points
for i, txt in enumerate(price_trend['sale_price']):
    plt.annotate(f"${txt:,.0f}", (price_trend['year'][i], price_trend['sale_price'][i]), 
                 textcoords="offset points", xytext=(0, 5), ha='center', fontsize=10, fontweight='bold')

plt.xlabel("Year", fontsize=12, fontweight='bold')
plt.ylabel("Average Sale Price ($)", fontsize=12, fontweight='bold')
plt.title("EV Sale Price Trend Over Time", fontsize=14, fontweight='bold', color='darkred')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show plot
plt.show()


#%%
####################################
# New vs Used EV Price Comparison
####################################

# Aggregate average price by new vs. used status
price_comparison = df_filtered.groupby('new_or_used_vehicle')['sale_price'].mean().reset_index()
price_comparison['sale_price'] = price_comparison['sale_price'].round(2)  # Round to 2 decimals

# Print new vs. used price comparison table
print("\n Average Sale Price of New vs. Used EVs:")
print(tabulate(price_comparison, headers=['Vehicle Type', 'Average Sale Price ($)'], tablefmt='pretty', showindex=False))





#%%
####################################
# EV Usage Patterns & Mileage Analysis
####################################



# Filter out missing or zero odometer readings
df_filtered = ev_df[ev_df['odometer_reading'] > 0].copy()

# Categorize New vs. Used Vehicles
mileage_comparison = df_filtered.groupby('new_or_used_vehicle')['odometer_reading'].describe()

# Print Mileage Statistics Table for New vs. Used Vehicles
print("\n Mileage Analysis: New vs. Used EVs")
print(tabulate(mileage_comparison, headers="keys", tablefmt="pretty", showindex=True))

# Plot Mileage Distribution for New vs. Used EVs
plt.figure(figsize=(12, 6))
sns.boxplot(x='new_or_used_vehicle', y='odometer_reading', data=df_filtered, palette="muted")

plt.xlabel("Vehicle Type", fontsize=12, fontweight='bold')
plt.ylabel("Odometer Reading (Miles)", fontsize=12, fontweight='bold')
plt.title("Mileage Distribution: New vs. Used EVs", fontsize=14, fontweight='bold', color='darkblue')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show plot
plt.show()



#%%
####################################
# Predicting Future Charging Demand for High-Mileage Vehicles
####################################


# Ensure transaction_date is in datetime format
ev_df['transaction_date'] = pd.to_datetime(ev_df['transaction_date'], errors='coerce')

# Extract year from transaction_date and ensure it's an integer
ev_df['year'] = ev_df['transaction_date'].dt.year.astype('Int64')

# Filter out zero or null odometer readings
df_filtered = ev_df[ev_df['odometer_reading'] > 0].copy()

# Define high-mileage threshold (e.g., top 25% percentile)
high_mileage_threshold = np.percentile(df_filtered['odometer_reading'], 75)

# Identify high-mileage vehicles (potential frequent chargers)
df_filtered['high_mileage'] = df_filtered['odometer_reading'] >= high_mileage_threshold

# Aggregate by city & county for charging demand hotspots
charging_demand_hotspots = (
    df_filtered[df_filtered['high_mileage']]
    .groupby(['county', 'city'])
    .size()
    .reset_index(name='high_mileage_ev_count')
)

# Sort by highest demand areas
charging_demand_hotspots = charging_demand_hotspots.sort_values(by='high_mileage_ev_count', ascending=False)

# Reset index properly to avoid duplicate or misaligned indexing
charging_demand_hotspots.reset_index(drop=True, inplace=True)
charging_demand_hotspots.index += 1  # Start index from 1 for readability

# Print High Mileage EV Demand Hotspots Table
print("\n Predicted High-Charging Demand Areas (Top 20)")
print(tabulate(charging_demand_hotspots.head(20), headers=['Index', 'County', 'City', 'High Mileage EVs'], tablefmt='pretty', showindex=True))

# Plot Top 10 Cities with Highest Charging Demand
top_cities = charging_demand_hotspots.head(10)

plt.figure(figsize=(14, 6))
bars = plt.barh(top_cities['city'], top_cities['high_mileage_ev_count'], color='cornflowerblue')

# Add labels to bars
for bar in bars:
    plt.text(
        bar.get_width(), bar.get_y() + bar.get_height() / 2,
        f'{int(bar.get_width())}',
        va='center', ha='left', fontsize=10, fontweight='bold', color='black'
    )

plt.xlabel("Number of High Mileage EVs", fontsize=12, fontweight='bold')
plt.ylabel("City", fontsize=12, fontweight='bold')
plt.title("Top 10 Cities with Highest Charging Demand (Based on High Mileage EVs)", fontsize=14, fontweight='bold', color='darkblue')

plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Show plot
plt.show()


# %%
####################################
# #Statistical analysis
####################################


#%%
####################################
# #Descriptive and Inferential Statistics - Evaluating EV-to-Charging Station Ratio to Identify High-Risk Counties
####################################

# Convert transaction_date to datetime
ev_df['transaction_date'] = pd.to_datetime(ev_df['transaction_date'])

# Aggregate EV counts by county
ev_county = ev_df.groupby('county').size().reset_index(name='EV_count')

# Aggregate Charging Station counts by county
charging_county = charging_df.groupby('county').size().reset_index(name='Station_count')

# Merge both datasets on county
infra_readiness = pd.merge(ev_county, charging_county, on='county', how='left')

# Compute EVs per Charging Station Ratio
infra_readiness['EVs_per_Station'] = infra_readiness['EV_count'] / infra_readiness['Station_count']
infra_readiness = infra_readiness.sort_values(by='EVs_per_Station', ascending=False)

# Filter only the top 10 high-risk counties for better visualization
top_10_risk_counties = infra_readiness.head(10).reset_index()  # Reset index to include in table

# 📊 Improved Visualization: Bar Chart for Top 10 Counties
plt.figure(figsize=(12, 6))
sns.barplot(data=top_10_risk_counties, x='county', y='EVs_per_Station', palette='coolwarm')
plt.xticks(rotation=45)
plt.xlabel("County")
plt.ylabel("EVs per Charging Station")
plt.title("Top 10 High-Risk Counties: EVs per Charging Station")
plt.grid()
plt.show()


#%%
####################################
# Outlier Detection in Charging Demand Intensity (EV-to-Station Ratio)
####################################


# Select the top 10 counties with the highest EV-to-Station ratio
top_10_high_demand = infra_readiness.sort_values(by='EVs_per_Station', ascending=False).head(10).reset_index(drop=True)

# 📊 Improved Boxplot for Outlier Detection
plt.figure(figsize=(8, 5))
sns.boxplot(y=infra_readiness['EVs_per_Station'], color="royalblue")
plt.title("Boxplot of Charging Demand Intensity (EVs per Station)")
plt.ylabel("EVs per Charging Station")
plt.grid()
plt.show()

# 
table = tabulate(
    top_10_high_demand[['county', 'EV_count', 'Station_count', 'EVs_per_Station']], 
    headers=["County", "EV Count", "Charging Stations", "EVs per Station"], 
    tablefmt="grid", 
    showindex="always"
)

# Print the high-risk counties in tabular format
print("\n🔹 High-Risk Counties (Charging Demand Intensity):\n")
print(table)


#%%
####################################
# #Correlation analysis
####################################

# Convert transaction_date to datetime for time-based analysis
ev_df["transaction_date"] = pd.to_datetime(ev_df["transaction_date"], errors="coerce")

# Selecting relevant numerical variables for correlation analysis
ev_corr_features = ["model_year", "electric_range", "odometer_reading", "sale_price", "postal_code"]

# Convert non-numeric values to NaN
for col in ev_corr_features:
    ev_df[col] = pd.to_numeric(ev_df[col], errors="coerce")


# Drop rows with NaN values (optional but ensures clean correlation calculations)
ev_df_cleaned = ev_df[ev_corr_features].dropna()

# Pearson Correlation Matrix (linear relationships)
ev_corr_matrix = ev_df_cleaned.corr(method="pearson")

# Spearman Correlation Matrix (monotonic relationships)
ev_spearman_corr = ev_df_cleaned.corr(method="spearman")

# Plotting Pearson Correlation Heatmap for EV Data
plt.figure(figsize=(8, 6))
sns.heatmap(ev_corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Pearson Correlation Matrix - EV Data")
plt.show()

# Plotting Spearman Correlation Heatmap for EV Data
plt.figure(figsize=(8, 6))
sns.heatmap(ev_spearman_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Spearman Correlation Matrix - EV Data")
plt.show()




#%%
####################################
# #Correlation analysis - Relationship Between Charging Station Density & EV Adoption
####################################


# 1️⃣ Aggregate EV Adoption by County
ev_county_counts = ev_df.groupby('county').size().reset_index(name='ev_count')

# 2️⃣ Aggregate Charging Stations by County
charging_county_counts = charging_df.groupby('county').size().reset_index(name='charging_station_count')

# 3️⃣ Merge EV and Charging Data on County
merged_data = pd.merge(ev_county_counts, charging_county_counts, on='county', how='inner')

# 4️⃣ Calculate Pearson Correlation
correlation, p_value = pearsonr(merged_data['ev_count'], merged_data['charging_station_count'])
print(f"Pearson Correlation: {correlation:.4f}, p-value: {p_value:.4f}")

# 5️⃣ Scatter Plot to Show Relationship
plt.figure(figsize=(10,6))
sns.scatterplot(x=merged_data['charging_station_count'], y=merged_data['ev_count'])
plt.xlabel("Number of Charging Stations")
plt.ylabel("EV Adoption Count")
plt.title("Charging Station Density vs. EV Adoption")
plt.grid()
plt.show()

# 6️⃣ Heatmap of Correlation
plt.figure(figsize=(6,5))
sns.heatmap(merged_data[['ev_count', 'charging_station_count']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between EV Adoption & Charging Stations")
plt.show()









#%%
####################################
# #Time Series analysis
####################################


# Convert transaction_date to datetime
ev_df['transaction_date'] = pd.to_datetime(ev_df['transaction_date'])

# Aggregate by Month to Identify Trends
ev_df['year_month'] = ev_df['transaction_date'].dt.to_period('M')
ev_monthly = ev_df.groupby('year_month').size()

# Convert to time series format
ev_monthly.index = ev_monthly.index.to_timestamp()

# Plot the overall trend of EV adoption
plt.figure(figsize=(12, 6))
plt.plot(ev_monthly, marker='o', linestyle='-')
plt.xlabel("Year-Month")
plt.ylabel("Number of EV Transactions")
plt.title("EV Adoption Trends Over Time")
plt.grid()
plt.show()

# Rolling Mean for Smoothing (7-month window)
plt.figure(figsize=(12, 6))
plt.plot(ev_monthly, label="Original")
plt.plot(ev_monthly.rolling(window=7).mean(), label="7-Month Rolling Mean", linestyle="dashed", color="red")
plt.xlabel("Year-Month")
plt.ylabel("Number of EV Transactions")
plt.title("EV Adoption Trends with Rolling Mean")
plt.legend()
plt.grid()
plt.show()

# ADF Test for Stationarity
adf_result = adfuller(ev_monthly)
print("ADF Test Statistic:", adf_result[0])
print("p-value:", adf_result[1])
if adf_result[1] < 0.05:
    print("The time series is stationary.")
else:
    print("The time series is NOT stationary (may need differencing).")

# Seasonal Decomposition
decomposition = seasonal_decompose(ev_monthly, model='additive', period=12)
fig = decomposition.plot()
plt.show()  





# %%
####################################
# #PCA analysis
####################################

# Selecting numerical features for PCA
ev_numeric_features = ev_df[['model_year', 'electric_range', 'odometer_reading', 'sale_price', 'postal_code']]
charging_numeric_features = charging_df[['Latitude', 'Longitude', 'ZipCode']]

# Merging the datasets based on the county or postal code if needed
merged_data = pd.merge(ev_numeric_features, charging_numeric_features, left_on='postal_code', right_on='ZipCode', how='left')

# Drop any remaining categorical columns if necessary
merged_data = merged_data.drop(columns=['postal_code', 'ZipCode'])

# Handle missing values (if any)
merged_data = merged_data.dropna()

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(merged_data)

# Apply PCA
pca = PCA(n_components=merged_data.shape[1])  # Keep all components initially
principal_components = pca.fit_transform(scaled_data)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Plot Scree Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio), marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Principal Components')
plt.grid()
plt.show()

# Choosing the optimal number of components (e.g., 95% explained variance)
cumulative_variance = np.cumsum(explained_variance_ratio)
optimal_components = np.argmax(cumulative_variance >= 0.95) + 1

print(f"Optimal number of principal components to retain 95% variance: {optimal_components}")
#%%
# Transform data with the optimal number of components
pca_final = PCA(n_components=optimal_components)
reduced_data = pca_final.fit_transform(scaled_data)

# Convert to DataFrame
pca_df = pd.DataFrame(reduced_data, columns=[f'PC{i+1}' for i in range(optimal_components)])

# Display first few rows of PCA-transformed data
print(pca_df.head())


# %%
####################################
# #SARIMA Forecasting
####################################

from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import pandas as pd

# Recompute monthly EV registrations
ev_df['transaction_date'] = pd.to_datetime(ev_df['transaction_date'])
ev_monthly = ev_df.groupby(ev_df['transaction_date'].dt.to_period('M')).size()
ev_monthly.index = ev_monthly.index.to_timestamp()

# Define SARIMA model parameters (p, d, q) x (P, D, Q, s)
sarima_order = (2, 1, 2)  # ARIMA component (p, d, q)
seasonal_order = (1, 1, 1, 12)  # Seasonal component (P, D, Q, s) for yearly seasonality

# Fit SARIMA model
sarima_model = SARIMAX(ev_monthly, order=sarima_order, seasonal_order=seasonal_order)
sarima_fit = sarima_model.fit(disp=False)

# Forecast until 2028 (48 months ahead)
forecast_steps = 48
sarima_forecast = sarima_fit.forecast(steps=forecast_steps)

# Generate date range for forecasted values
forecast_dates = pd.date_range(start=ev_monthly.index[-1], periods=forecast_steps+1, freq='M')[1:]

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(ev_monthly, label="Historical Data", color="blue")
plt.plot(forecast_dates, sarima_forecast, label="SARIMA Forecast (2025-2028)", linestyle="dashed", color="red")
plt.xlabel("Year-Month")
plt.ylabel("EV Registrations")
plt.title("EV Adoption Forecast using SARIMA (2025-2028)")
plt.legend()
plt.grid()
plt.show()

# %%
