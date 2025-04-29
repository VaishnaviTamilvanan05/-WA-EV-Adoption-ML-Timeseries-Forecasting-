
import pandas as pd
import ast
import pgeocode
import os


df = pd.read_csv(r'E:\Capstone\data\charging\washington_filtered_ev_stations.csv')

cols_to_drop = [
    'IsRecentlyVerified','UUID','DataProviderID','OperatorID','NumberOfPoints',
    'GeneralComments','StatusTypeID','DateLastStatusUpdate','DataQualityLevel',
    'DateCreated','DatePlanned','OperatorsReference','MetadataValues',
    'DateLastConfirmed','SubmissionStatusTypeID','UsageCost','DataProvidersReference',
    'UsageTypeID'
]
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

def parse_connections(info):
    if pd.isna(info):
        return []
    if isinstance(info, str):
        try:
            return ast.literal_eval(info)
        except (ValueError, SyntaxError):
            return []
    if isinstance(info, list):
        return info
    return []

df['total_ports'] = df['Connections'].apply(
    lambda conns: sum(item.get('Quantity', 0) for item in parse_connections(conns))
)
df['l2_ports'] = df['Connections'].apply(
    lambda conns: sum(item.get('Quantity', 0)
                      for item in parse_connections(conns)
                      if item.get('LevelID') == 2)
)
df['dcfc_ports'] = df['Connections'].apply(
    lambda conns: sum(item.get('Quantity', 0)
                      for item in parse_connections(conns)
                      if item.get('LevelID') == 3)
)

df.drop(columns=['Connections'], inplace=True)

addr_col = None
for candidate in ('AddressInfo','Address Info'):
    if candidate in df.columns:
        addr_col = candidate
        break

if addr_col:
    def get_field(info, field):
        if pd.isna(info):
            return None
        if isinstance(info, str):
            try:
                info = ast.literal_eval(info)
            except:
                return None
        if isinstance(info, dict):
            return info.get(field)
        return None

    df['AddressLine1'] = df[addr_col].apply(lambda x: get_field(x, 'AddressLine1'))
    df['Town']         = df[addr_col].apply(lambda x: get_field(x, 'Town'))
    df['Latitude']     = df[addr_col].apply(lambda x: get_field(x, 'Latitude'))
    df['Longitude']    = df[addr_col].apply(lambda x: get_field(x, 'Longitude'))

    df['Latitude']  = pd.to_numeric(df['Latitude'],  errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

    df.drop(columns=[addr_col], errors='ignore', inplace=True)
else:
    df['AddressLine1'] = None
    df['Town']         = None
    df['Latitude']     = pd.NA
    df['Longitude']    = pd.NA

df['ZipCode'] = pd.to_numeric(df['ZipCode'], errors='coerce').fillna(0).astype(int)
nomi = pgeocode.Nominatim('us')
df['county'] = df['ZipCode'].apply(
    lambda z: nomi.query_postal_code(str(z).zfill(5)).county_name
)

for col in df.select_dtypes(include='number').columns:
    if col in ['Latitude', 'Longitude']:
        continue
    df[col] = df[col].fillna(0).clip(lower=0)

out_dir = r'E:\Capstone\data\charging\processed'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'WA_charging_cleaned_with_ports.csv')
df.to_csv(out_path, index=False)
print(f"Saved cleaned charging data with port counts to:\n   {out_path}")
