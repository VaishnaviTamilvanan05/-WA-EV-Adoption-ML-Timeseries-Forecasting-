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
