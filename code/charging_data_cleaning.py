# %%
import pandas as pd
import json
import ast
# %%
df=pd.read_csv('E:\\Capstone\\data\\charging\\washington_filtered_ev_stations.csv')

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
df.to_csv("E:\\Capstone\\data\\charging\\processed\\WA_charging_cleaned_data.csv", index=True)
