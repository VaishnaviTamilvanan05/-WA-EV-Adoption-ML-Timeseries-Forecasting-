# %%
import pandas as pd
import numpy as np


# %%

data=pd.read_csv('E:\\Capstone\\data\\EV\\raw\\ev_raw_data.csv')
data.columns

# %%

data.info()

# %%
data.head()
# %%
data.columns = data.columns.str.lower().str.strip().str.replace(r"[^a-zA-Z0-9]", "_", regex=True)

# Print new column names to verify
print("Updated column names:", data.columns)
# %%
# Data Preprocessing

# dropping unwanted columns
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

data.head()

# %%
data['transaction_date'] = pd.to_datetime(data['transaction_date'])

data['m/y'] = data['transaction_date'].dt.strftime("%m-%Y")

data.set_index('transaction_date', inplace=True)

# %% 
data.columns

# %%
# dropping redundant cols

red_cols=['sale_date','base_msrp','year']

data.drop(red_cols, axis=1, inplace=True)

# %%

print("Checking for null values:\n", data.isnull().sum())

duplicate_count = data.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_count}")
# %%
data[['postal_code', 'county', 'city']] = data[['postal_code', 'county', 'city']].fillna('Unknown')
data['state'].fillna('WA', inplace=True)
data.dropna(subset=['electric_range', 'electric_utility'], inplace=True)
data.drop_duplicates(inplace=True)

# %%
data['county'].value_counts()
# %%
data['model'].unique()

# %%
len(data['model'].unique())


# %%
data.to_csv("E:\\Capstone\\data\\EV\\processed\\ev_cleaned_data.csv", index=True)

# %%
print(data.select_dtypes(include=['object']).apply(lambda x: x.str.contains(r'^\s|\s$', regex=True).sum()))

# %%
print(data.columns.duplicated().sum())  # Should return 0

# %%
data.head()
# %%
