import json
import requests
import time
import pandas as pd
import os
from datetime import date

# Define a function to open/load JSON files
def get_keys(path):
    """Function accesses and returns the JSON file at the specified path."""
    with open(path) as f:
        return json.load(f)

# Storing API keys in variables
keys = get_keys("/Users/berke/.secret/socrata_api_project_5.json")
api_key_socrata = keys['api_key']
app_token_socrata = keys['app_token']
api_key_secret_socrata = keys['api_key_secret']

# Defining API request headers and parameters
headers = {
    'X-App-Token': app_token_socrata,
    'username': api_key_socrata,
    'password': api_key_secret_socrata
}
params = {'$limit': '50000', '$offset': None}
offset = list(range(0, 500000, 50000))

# Requesting data from API and parsing results to a dictionary
dfs = {}
for number in offset:
    params['$offset'] = str(number)
    r = requests.get('https://data.wa.gov/resource/rpr4-cgyd.json?', 
                     headers=headers, params=params)
    dfs[f'df_{number}'] = pd.DataFrame.from_records(r.json())
    time.sleep(1)

# Saving all pages as one CSV file
today = date.today().strftime("%m-%d-%Y")
path = 'E:/Capstone/data'
os.makedirs(path, exist_ok=True)
output_file = os.path.join(path, 'api_data.csv.gz')

df_final = pd.concat(dfs.values(), axis=0)
df_final.to_csv(output_file, index=False, compression='gzip')

print(f"CSV file saved as {output_file}")
