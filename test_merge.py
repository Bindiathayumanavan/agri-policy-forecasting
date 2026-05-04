import pandas as pd

try:
    df_crop = pd.read_csv('data/crop_production.csv', nrows=1000)
    print("Crop Production cols:", df_crop.columns.tolist())
    
    df_rain = pd.read_csv('data/rainfall.csv', nrows=1000)
    print("Rainfall cols:", df_rain.columns.tolist())
    
    df_temp = pd.read_csv('data/temperature.csv', nrows=1000)
    print("Temperature cols:", df_temp.columns.tolist())
    
    df_soil = pd.read_csv('data/soil_nutrients.csv', nrows=1000)
    print("Soil cols:", df_soil.columns.tolist())
    
    df_fert = pd.read_csv('data/fertilizer.csv', nrows=1000)
    print("Fertilizer cols:", df_fert.columns.tolist())
    
    df_prices = pd.read_csv('data/crop_prices.csv', nrows=1000)
    print("Prices cols:", df_prices.columns.tolist())
except Exception as e:
    print("Error:", e)
