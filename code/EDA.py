# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('E:\\Capstone\\data\\EV\\processed\\ev_cleaned_data.csv')

# %%
county_counts = data["county"].value_counts().head(10)  

# Plot bar chart
plt.figure(figsize=(12,6))
sns.barplot(x=county_counts.index, y=county_counts.values, palette="viridis")
plt.xticks(rotation=45)
plt.title("Top 10 Counties with Highest EV Registrations")
plt.xlabel("County")
plt.ylabel("Number of Registrations")
plt.show()
# %%
county_counts = data["county"].value_counts().head(10)

# Plot pie chart
plt.figure(figsize=(8,8))
plt.pie(county_counts.values, labels=county_counts.index, autopct='%1.1f%%', 
        startangle=140, colors=sns.color_palette("viridis", len(county_counts)))

plt.title("Share of EV Registrations Among Top 10 Counties")
plt.show()
# %%
data["transaction_date"] = pd.to_datetime(data["transaction_date"])

# Extract top 10 counties
top_10_counties = data["county"].value_counts().head(10).index

# Filter data for top 10 counties
filtered_data = data[data["county"].isin(top_10_counties)]

# Group by month and county to count registrations
trend_data = filtered_data.groupby([filtered_data["transaction_date"].dt.to_period("M"), "county"]).size().unstack()

# Plot stacked area chart
plt.figure(figsize=(12,6))
trend_data.plot(kind="area", stacked=True, colormap="viridis", alpha=0.8, figsize=(12,6))
plt.title("EV Registration Trends Over Time in Top 10 Counties")
plt.xlabel("Time (Month-Year)")
plt.ylabel("Number of Registrations")
plt.legend(title="County", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
# %%
data["transaction_date"] = pd.to_datetime(data["transaction_date"])

# Extract the month from the transaction date
data["month"] = data["transaction_date"].dt.month

# Plot boxplot to show variation in EV registrations across months
plt.figure(figsize=(12,6))
sns.boxplot(x=data["month"], y=data.groupby("month")["county"].transform("count"), palette="viridis")

plt.title("Variation in EV Registrations Across Different Months")
plt.xlabel("Month")
plt.ylabel("Number of Registrations")
plt.xticks(ticks=range(0, 12), labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
plt.show()
# %%
data["transaction_date"] = pd.to_datetime(data["transaction_date"])

# Extract year and month
data["year"] = data["transaction_date"].dt.year
data["month"] = data["transaction_date"].dt.month

# Create a pivot table for heatmap
heatmap_data = data.pivot_table(index="year", columns="month", values="county", aggfunc="count")

# Plot heatmap
plt.figure(figsize=(12,6))
sns.heatmap(heatmap_data, cmap="viridis", annot=True, fmt=".0f", linewidths=0.5)

plt.title("Monthly vs. Yearly EV Registration Density")
plt.xlabel("Month")
plt.ylabel("Year")
plt.xticks(ticks=range(12), labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
plt.show()
# %%
data["transaction_date"] = pd.to_datetime(data["transaction_date"])
data["year"] = data["transaction_date"].dt.year
data_filtered = data[(data["year"] >= 2010) & (data["year"] <= 2024)]
yearly_trend = data_filtered.groupby("year").size()

plt.figure(figsize=(12,6))
plt.plot(yearly_trend.index, yearly_trend.values, marker="o", linestyle="-", color="b")
plt.title("EV Adoption Trend (2010-2024)")
plt.xlabel("Year")
plt.ylabel("Number of EV Registrations")
plt.grid(True)
plt.show()
# %%
data["transaction_date"] = pd.to_datetime(data["transaction_date"])
data["year"] = data["transaction_date"].dt.year
data_filtered = data[(data["year"] >= 2010) & (data["year"] <= 2024)]

vehicle_type_counts = data_filtered.groupby(["year", "clean_alternative_fuel_vehicle_type"]).size().unstack()

plt.figure(figsize=(12,6))
vehicle_type_counts.plot(kind="bar", stacked=True, colormap="viridis", figsize=(12,6))
plt.title("EV Adoption Trend by Vehicle Type (2010-2024)")
plt.xlabel("Year")
plt.ylabel("Number of Registrations")
plt.legend(title="Vehicle Type")
plt.show()
# %%

city_charging_demand = data.groupby("city")["electric_range"].mean().sort_values(ascending=False).head(10)

# Plot bar chart for top 10 cities with high charging demand
plt.figure(figsize=(12,6))
sns.barplot(x=city_charging_demand.index, y=city_charging_demand.values, palette="viridis")
plt.xticks(rotation=45)
plt.title("Top 10 Cities with Highest Charging Demand")
plt.xlabel("City")
plt.ylabel("Average Electric Range")
plt.show()

# %%
from pandas.plotting import autocorrelation_plot

data["transaction_date"] = pd.to_datetime(data["transaction_date"])

# Group data by month to get total EV registrations over time
monthly_registrations = data.set_index("transaction_date").resample("M").size()

# Plot autocorrelation to identify seasonal patterns
plt.figure(figsize=(12,6))
autocorrelation_plot(monthly_registrations)
plt.title("Autocorrelation Plot of EV Registrations")
plt.show()

# %%

