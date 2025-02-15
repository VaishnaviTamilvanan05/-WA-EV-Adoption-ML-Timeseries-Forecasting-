#%%
print("""EDA Summary:
Visualizations:
- EV Adoption Trend (2010-2024) - Trend Analysis
- EV Adoption Trend by Vehicle Type - Category Comparison
- Top 10 Counties with Highest EV Registrations - Geographical Distribution
- Top 10 EV Manufacturers by Registration - Manufacturer Analysis
- EV Sale Price Trend Over Time - Price Trends
- Mileage Distribution: New vs. Used EVs - Comparative Analysis
- Fleet vs. Individual EV Usage: Mileage Analysis - Usage Patterns
- Top 10 Cities with Highest Charging Demand (Based on High Mileage EVs) - Charging Infrastructure""")


#%%
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns
from tabulate import tabulate 
import numpy as np

# Load dataset
file_path = "/Users/vaishnavitamilvanan/Downloads/Capstone Project/-WA-EV-Adoption-ML-Timeseries-Forecasting-/data/ev_cleaned_data.csv"  
df = pd.read_csv(file_path, low_memory=False)


#%%

####################################
#EV Adoption Trend Analysis
####################################


# Convert transaction_date to datetime format
df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')

# Extract year from the transaction date and ensure it's an integer
df['year'] = df['transaction_date'].dt.year.astype('Int64')

# Aggregate EV registrations by year
ev_trend_yearly = df.groupby('year').size().reset_index(name='ev_count')


ev_trend_yearly['year'] = ev_trend_yearly['year'].astype(int)

# Aggregate average sale price per year (ensuring integer year format)
price_trend = df.groupby('year')['sale_price'].mean().reset_index()
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
ev_type_trend = df.groupby(['year', 'clean_alternative_fuel_vehicle_type']).size().unstack()

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
ev_distribution = df.groupby(['county', 'city']).size().reset_index(name='ev_count')

# Sort by highest EV registrations
ev_distribution_sorted = ev_distribution.sort_values(by='ev_count', ascending=False)

# Add an index column
ev_distribution_sorted.reset_index(drop=True, inplace=True)
ev_distribution_sorted.index += 1  # Start index from 1

# Print EV distribution table using tabulate with an index column
print("\n Distribution of EVs by County & City:")
print(tabulate(ev_distribution_sorted.head(20), headers=['Index', 'County', 'City', 'EV Count'], tablefmt='pretty', showindex=True))

# Get top 10 counties with highest EV registrations
top_counties = df.groupby('county').size().reset_index(name='ev_count').sort_values(by='ev_count', ascending=False).head(10)

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
ev_make_model = df.groupby(['make', 'model']).size().reset_index(name='ev_count')

# Sort by highest EV registrations
ev_make_model_sorted = ev_make_model.sort_values(by='ev_count', ascending=False)

# Add an index column
ev_make_model_sorted.reset_index(drop=True, inplace=True)
ev_make_model_sorted.index += 1  # Start index from 1

# Print EV model and manufacturer distribution table
print("\n EV Model & Manufacturer Trends:")
print(tabulate(ev_make_model_sorted.head(20), headers=['Index', 'Make', 'Model', 'EV Count'], tablefmt='pretty', showindex=True))

# Get top 10 manufacturers with highest EV registrations
top_makes = df.groupby('make').size().reset_index(name='ev_count').sort_values(by='ev_count', ascending=False).head(10)

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
df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')

# Extract year from transaction_date and ensure it's an integer
df['year'] = df['transaction_date'].dt.year.astype('Int64')

# Filter out zero or null sale prices
df_filtered = df[df['sale_price'] > 0].copy()

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
df_filtered = df[df['odometer_reading'] > 0].copy()

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
# Fleet vs. Individual Ownership Patterns
####################################

# Aggregate mileage by primary use type
fleet_vs_individual = df_filtered.groupby('primary_use')['odometer_reading'].describe()

# Print Fleet vs. Individual Ownership Mileage Table
print("\n Fleet vs. Individual EV Usage Analysis")
print(tabulate(fleet_vs_individual, headers="keys", tablefmt="pretty", showindex=True))

# Plot Fleet vs. Individual Odometer Readings
plt.figure(figsize=(12, 6))
sns.boxplot(x='primary_use', y='odometer_reading', data=df_filtered, palette="coolwarm")

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.xlabel("Primary Use", fontsize=12, fontweight='bold')
plt.ylabel("Odometer Reading (Miles)", fontsize=12, fontweight='bold')
plt.title("Fleet vs. Individual EV Usage: Mileage Analysis", fontsize=14, fontweight='bold', color='darkblue')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show plot
plt.show()

#%%
####################################
# Predicting Future Charging Demand for High-Mileage Vehicles
####################################


# Ensure transaction_date is in datetime format
df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')

# Extract year from transaction_date and ensure it's an integer
df['year'] = df['transaction_date'].dt.year.astype('Int64')

# Filter out zero or null odometer readings
df_filtered = df[df['odometer_reading'] > 0].copy()

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
