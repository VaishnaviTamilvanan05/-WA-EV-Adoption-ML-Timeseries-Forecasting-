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
import requests
import pandas as pd
import os

# Open Charge Map API Key
API_KEY = "653c773a-9bb2-47ff-8964-8547c020cbbd"

# Open Charge Map API URL
URL = "https://api.openchargemap.io/v3/poi/"

# API Parameters to fetch data for Washington
params = {
    "key": API_KEY,
    "countrycode": "US",
    "stateorprovince": "Washington",
    "maxresults": 100000,  # Fetch as many as possible
    "compact": True,
    "verbose": False
}

# Define the directory path to save the file
save_directory = r"E:\Capstone\data\Charging"
csv_filename = "washington_filtered_ev_stations.csv"
full_path = os.path.join(save_directory, csv_filename)

try:
    # Make API request
    response = requests.get(URL, params=params)

    if response.status_code == 200:
        data = response.json()

        if data:
            # Convert JSON response to DataFrame
            df = pd.DataFrame(data)

            # Extract ZIP codes (some might be missing)
            df["ZipCode"] = df["AddressInfo"].apply(lambda x: x.get("Postcode", None) if isinstance(x, dict) else None)

            # Convert ZIP to numeric (force errors to NaN)
            df["ZipCode"] = pd.to_numeric(df["ZipCode"], errors='coerce')

            # Filter zip codes in the range 98001–99403
            df_filtered = df[(df["ZipCode"] >= 98001) & (df["ZipCode"] <= 99403)]

            # Ensure directory exists
            os.makedirs(save_directory, exist_ok=True)

            # Save filtered data to CSV
            df_filtered.to_csv(full_path, index=False)

            print(f"✅ Data successfully filtered and saved to '{full_path}'")
        else:
            print("⚠️ No data found for Washington State.")
    else:
        print(f"❌ API Error: {response.status_code}, {response.text}")

except Exception as e:
    print(f"❌ Error fetching data: {str(e)}")



# %%

import pandas as pd

df = pd.read_csv("E:\\Capstone\\data\\Charging\\washington_filtered_ev_stations.csv")
print(df.columns)


# %%
