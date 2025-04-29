import requests
import pandas as pd
import os

#API key
api_key = "CRpk42HcJ8xC4cAbdf32gZ0KmtgKYQUoKb1OhmGa"


url = "https://developer.nrel.gov/api/alt-fuel-stations/v1.json"


params = {
    "api_key": api_key,   
    "fuel_type": "ELEC",  
    "state": "WA",        
    "limit": "all"        
}


response = requests.get(url, params=params)


if response.status_code == 200:
    data = response.json()
    df_chargers = pd.DataFrame(data['fuel_stations']) 
    
    output_dir = r"E:\Capstone\data\Charging"  
    os.makedirs(output_dir, exist_ok=True)  

    
    output_file = os.path.join(output_dir, "charging_stations.csv")

    
    df_chargers.to_csv(output_file, index=False)

    print(f"Data fetched successfully and saved to: {output_file}")
else:
    print("Error:", response.status_code, response.text)
    
# %%

# Open Charge Map API Key
API_KEY = "653c773a-9bb2-47ff-8964-8547c020cbbd"

# Open Charge Map API URL
URL = "https://api.openchargemap.io/v3/poi/"

params = {
    "key": API_KEY,
    "countrycode": "US",
    "stateorprovince": "Washington",
    "maxresults": 100000,  
    "compact": True,
    "verbose": False
}


save_directory = r"E:\Capstone\data\Charging"
csv_filename = "washington_filtered_ev_stations.csv"
full_path = os.path.join(save_directory, csv_filename)

try:
    # API request
    response = requests.get(URL, params=params)

    if response.status_code == 200:
        data = response.json()

        if data:
           
            df = pd.DataFrame(data)

            df["ZipCode"] = df["AddressInfo"].apply(lambda x: x.get("Postcode", None) if isinstance(x, dict) else None)

            df["ZipCode"] = pd.to_numeric(df["ZipCode"], errors='coerce')

            df_filtered = df[(df["ZipCode"] >= 98001) & (df["ZipCode"] <= 99403)]

            os.makedirs(save_directory, exist_ok=True)

            df_filtered.to_csv(full_path, index=False)

            print(f" Data successfully filtered and saved to '{full_path}'")
        else:
            print(" No data found for Washington State.")
    else:
        print(f" API Error: {response.status_code}, {response.text}")

except Exception as e:
    print(f" Error fetching data: {str(e)}")



