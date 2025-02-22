# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('E:\\Capstone\\data\\EV\\processed\\ev_cleaned_data.csv')
df=pd.read_csv("E:\\Capstone\\data\\charging\\processed\\WA_charging_cleaned_data.csv")
# %%
county_counts = data["county"].value_counts().head(10)  

plt.figure(figsize=(12,6))
sns.barplot(x=county_counts.index, y=county_counts.values, palette="viridis")
plt.xticks(rotation=45)
plt.title("Top 10 Counties with Highest EV Registrations")
plt.xlabel("County")
plt.ylabel("Number of Registrations")
plt.show()

# %%
data["transaction_date"] = pd.to_datetime(data["transaction_date"])

top_10_counties = data["county"].value_counts().head(10).index

filtered_data = data[data["county"].isin(top_10_counties)]

trend_data = filtered_data.groupby([filtered_data["transaction_date"].dt.to_period("M"), "county"]).size().unstack()

plt.figure(figsize=(12,6))
trend_data.plot(kind="area", stacked=True, colormap="viridis", alpha=0.8, figsize=(12,6))
plt.title("EV Registration Trends Over Time in Top 10 Counties")
plt.xlabel("Time (Month-Year)")
plt.ylabel("Number of Registrations")
plt.legend(title="County", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# %%
data["transaction_date"] = pd.to_datetime(data["transaction_date"])

data["year"] = data["transaction_date"].dt.year
data["month"] = data["transaction_date"].dt.month

heatmap_data = data.pivot_table(index="year", columns="month", values="county", aggfunc="count")

plt.figure(figsize=(12,6))
sns.heatmap(heatmap_data, cmap="viridis", annot=True, fmt=".0f", linewidths=0.5)

plt.title("Monthly vs. Yearly EV Registration Density")
plt.xlabel("Month")
plt.ylabel("Year")
plt.xticks(ticks=range(12), labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
plt.show()
# %%
import matplotlib.pyplot as plt

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
ev_county_counts = data.groupby("county").size().reset_index(name="ev_registrations")
charging_county_counts = df.groupby("county").size().reset_index(name="charging_stations")

# Merge datasets
county_data = pd.merge(ev_county_counts, charging_county_counts, on="county", how="left").fillna(0)
county_data["chargers_per_ev"] = county_data["charging_stations"] / county_data["ev_registrations"]

# Sort data by EV registrations
county_data = county_data.sort_values(by="ev_registrations", ascending=False).head(10)

# Plot Bubble Chart
plt.figure(figsize=(12,6))
sns.scatterplot(
    x=county_data["county"],
    y=county_data["charging_stations"],
    size=county_data["ev_registrations"],
    hue=county_data["chargers_per_ev"],
    palette="coolwarm",
    sizes=(100, 1000),
    edgecolor="black",
    alpha=0.7
)

plt.title("Charging Station Availability vs. EV Registrations (Top 10 Counties)")
plt.xlabel("County")
plt.ylabel("Number of Charging Stations")
plt.xticks(rotation=45)
plt.legend(title="Chargers per EV Ratio", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.show()


# %%
data.columns
# %%
import statsmodels.api as sm

# Prepare the data for Linear Regression
ev_counts = data.groupby("year").size().reset_index(name="ev_registrations")

# Define independent (X) and dependent (Y) variables
X = ev_counts["year"]
y = ev_counts["ev_registrations"]

# Add constant for regression
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Get regression results
slope = model.params[1]
p_value = model.pvalues[1]

# Interpretation of Results
if p_value < 0.05:
    interpretation = "The p-value is statistically significant (p < 0.05), indicating a positive trend in EV registrations over time."
else:
    interpretation = "The p-value is not statistically significant (p > 0.05), meaning that EV registrations do not show a clear increasing trend over time."

# Display Regression Results
print(model.summary())
print(f"Interpretation: {interpretation}")

# Plot EV registrations with the regression line
plt.figure(figsize=(12,6))
sns.scatterplot(x=ev_counts["year"], y=ev_counts["ev_registrations"], color="blue", label="EV Registrations")
sns.lineplot(x=ev_counts["year"], y=model.predict(X), color="red", label="Regression Line")

plt.title("Linear Regression Trend of EV Registrations Over Time")
plt.xlabel("Year")
plt.ylabel("Number of EV Registrations")
plt.legend()
plt.grid(True)
plt.show()
# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of charging stations across counties
plt.figure(figsize=(12,6))
sns.barplot(x=df["county"].value_counts().index[:10], y=df["county"].value_counts().values[:10], palette="Blues_r")
plt.title("Top 10 Counties with Most Charging Stations")
plt.xlabel("County")
plt.ylabel("Number of Charging Stations")
plt.xticks(rotation=45)
plt.show()

# %%
plt.figure(figsize=(12,6))
sns.barplot(x=df["ZipCode"].value_counts().index[:10], y=df["ZipCode"].value_counts().values[:10], palette="Greens_r")
plt.title("Top 10 ZIP Codes with Most Charging Stations")
plt.xlabel("Zip Code")
plt.ylabel("Number of Charging Stations")
plt.xticks(rotation=45)
plt.show()

# %%
from scipy.stats import spearmanr
ev_county_counts = data.groupby("county").size().reset_index(name="ev_registrations")

# Aggregate charging stations per county
charging_county_counts = df.groupby("county").size().reset_index(name="charging_stations")

# Merge EV and charging station data per county
county_data = pd.merge(ev_county_counts, charging_county_counts, on="county", how="inner")

# Spearman's Correlation Test for EV registrations vs. Charging stations (county level)
correlation, p_value = spearmanr(county_data["ev_registrations"], county_data["charging_stations"])

print(f"Spearman Correlation (County Level): {correlation:.4f}")
print(f"P-Value: {p_value:.4f}")
# %%
ev_county_counts = data.groupby("county").size().reset_index(name="ev_registrations")

# Aggregate charging stations per county
charging_county_counts = df.groupby("county").size().reset_index(name="charging_stations")

# Merge EV and charging station data per county
county_data = pd.merge(ev_county_counts, charging_county_counts, on="county", how="inner")

# Scatter plot of EV registrations vs. charging stations per county
plt.figure(figsize=(12,6))
sns.scatterplot(x=county_data["ev_registrations"], y=county_data["charging_stations"], color="b")
sns.regplot(x=county_data["ev_registrations"], y=county_data["charging_stations"], scatter=False, color="r")

plt.title("Correlation Between EV Registrations and Charging Stations (County Level)")
plt.xlabel("EV Registrations")
plt.ylabel("Charging Stations")
plt.grid(True)
plt.show()
# %%
