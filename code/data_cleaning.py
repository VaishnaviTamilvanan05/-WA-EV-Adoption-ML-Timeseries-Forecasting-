# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# %%
data = pd.read_csv('E:\\Capstone\\data\\EV\\raw\\ev_raw_data.csv')

# Standardizing column names
data.columns = data.columns.str.lower().str.strip().str.replace(r"[^a-zA-Z0-9]", "_", regex=True)

# %%
# irrelevant or redundant columns
drop_cols = [
    'electric_vehicle_fee_paid',
    'transportation_electrification_fee_paid',
    'hybrid_vehicle_electrification_fee_paid',
    'legislative_district',
    '2019_hb_2042__clean_alternative_fuel_vehicle__cafv__eligibility',
    'meets_2019_hb_2042_electric_range_requirement',
    'meets_2019_hb_2042_sale_date_requirement',
    'meets_2019_hb_2042_sale_price_value_requirement',
    '2019_hb_2042__battery_range_requirement',
    '2019_hb_2042__purchase_date_requirement',
    '2019_hb_2042__sale_price_value_requirement',
    '2020_geoid'
]
data.drop(drop_cols, axis=1, inplace=True)

# %%

data['transaction_date'] = pd.to_datetime(data['transaction_date'])
data['m/y'] = data['transaction_date'].dt.strftime("%m-%Y")
data.set_index('transaction_date', inplace=True)

# %%
data.drop(['sale_date', 'base_msrp', 'year'], axis=1, inplace=True)

# %%
# Handling missing and duplicate data
data[['postal_code', 'county', 'city']] = data[['postal_code', 'county', 'city']].fillna('Unknown')
data['state'].fillna('WA', inplace=True)
data.dropna(subset=['electric_range', 'electric_utility'], inplace=True)
data.drop_duplicates(inplace=True)

# %%
data.to_csv("E:\\Capstone\\data\\EV\\processed\\ev_cleaned_data.csv", index=True)

# %%
print("Whitespace issues in string columns:\n", 
      data.select_dtypes(include=['object']).apply(lambda x: x.str.contains(r'^\s|\s$', regex=True).sum()))

# Check for duplicated column names
print(f"Number of duplicated column names: {data.columns.duplicated().sum()}")

# %%
# Detecting outliers using IQR and Z-score methods
outlier_columns = ["electric_range", "odometer_reading", "sale_price", "model_year"]
outliers_dict = {}
zscore_outliers = {}

for col in outlier_columns:
    # IQR Method
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_dict[col] = data[(data[col] < lower_bound) | (data[col] > upper_bound)].shape[0]

    # Z-score Method
    data["z_score"] = np.abs(zscore(data[col]))
    zscore_outliers[col] = data[data["z_score"] > 3].shape[0]

# %%
# Visualizing outlier counts
outliers_df = pd.DataFrame({
    "IQR Outliers": outliers_dict,
    "Z-score Outliers": zscore_outliers
})

outliers_df.plot(kind="bar", colormap="viridis", figsize=(10, 5))
plt.title("Outlier Counts in Selected Numerical Variables")
plt.xlabel("Feature")
plt.ylabel("Number of Outliers")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.tight_layout()
plt.show()

print("\nOutlier Summary:\n", outliers_df)

