# %%
import pandas as pd
import json
import ast
import pgeocode
# %%
df=pd.read_csv('E:\\Capstone\\data\\charging\\washington_filtered_ev_stations.csv')
# data=pd.read_csv("E:\\Capstone\\data\\EV\\processed\\ev_cleaned_data.csv")

# %%
df.head()
# %%
df.columns
# %%
# dropping unwanted
columns_to_drop = [
    'IsRecentlyVerified', 'UUID', 'DataProviderID', 'OperatorID', 'NumberOfPoints',
    'GeneralComments', 'StatusTypeID', 'DateLastStatusUpdate', 'DataQualityLevel',
    'DateCreated', 'DatePlanned', 'OperatorsReference', 'MetadataValues', 'DateLastConfirmed', 
    'SubmissionStatusTypeID','UsageCost','DataProvidersReference','Connections','UsageTypeID'
]

df.drop(columns=columns_to_drop, inplace=True)
# %%


def get_field(info, field):
    # If info is a string, convert it to a dictionary using ast.literal_eval
    if isinstance(info, str):
        try:
            info = ast.literal_eval(info)
        except (ValueError, SyntaxError):
            return None
    # If info is a dictionary, return the desired field's value
    if isinstance(info, dict):
        return info.get(field, None)
    return None

# Create new columns by applying the helper function for each field
df['AddressLine1'] = df['AddressInfo'].apply(lambda x: get_field(x, 'AddressLine1'))
df['Town'] = df['AddressInfo'].apply(lambda x: get_field(x, 'Town'))
df['Latitude'] = df['AddressInfo'].apply(lambda x: get_field(x, 'Latitude'))
df['Longitude'] = df['AddressInfo'].apply(lambda x: get_field(x, 'Longitude'))

# %%
df.drop(columns='AddressInfo',inplace=True)
# %%
df['ZipCode'] = df['ZipCode'].astype(int)
# %%
df.head()
# %%
null_counts = df.isnull().sum()
print("Missing values in each column:")
print(null_counts)

duplicate_count = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_count}")

# %%
nomi = pgeocode.Nominatim('us')

def get_county(zip_code):
    # Ensure the ZIP code is a 5-digit string
    zip_code = str(zip_code).strip().zfill(5)
    result = nomi.query_postal_code(zip_code)
    # The returned result is a pandas Series; county data might be in 'county_name'
    # Check the output of result to confirm the column name.
    return result.county_name if hasattr(result, 'county_name') else None

# Apply the function to create a new 'county' column in df
df['county'] = df['ZipCode'].apply(get_county)

# %%
for col in df.select_dtypes(include=['number']).columns:
    df[col] = df[col].fillna(0).clip(lower=0) 
             
# %%
df.to_csv("E:\\Capstone\\data\\charging\\processed\\WA_charging_cleaned_data.csv", index=True)

df.head()
# %%

