import pandas as pd
import numpy as np

# load data
df_crop = pd.read_csv('data/crop_production.csv')
df_rain = pd.read_csv('data/rainfall.csv')
df_temp = pd.read_csv('data/temperature.csv')
df_soil = pd.read_csv('data/soil_nutrients.csv')
df_fert = pd.read_csv('data/fertilizer.csv')
df_prices = pd.read_csv('data/crop_prices.csv')

# Standardize columns
df_crop['State'] = df_crop['State_Name'].str.strip().str.title()
df_crop['Year'] = df_crop['Crop_Year']
df_crop['Crop'] = df_crop['Crop'].str.strip().str.title()
df_crop['District'] = df_crop['District_Name'].str.strip().str.title()

df_rain['State'] = df_rain['SUBDIVISION'].str.strip().str.title()
df_rain['Year'] = df_rain['YEAR']

df_temp['dt'] = pd.to_datetime(df_temp['dt'])
df_temp['Year'] = df_temp['dt'].dt.year
df_temp['State'] = df_temp['State'].str.strip().str.title()
df_temp_agg = df_temp.groupby(['State', 'Year'])['AverageTemperature'].mean().reset_index()

df_soil['District'] = df_soil['District'].str.strip().str.title()
df_soil_agg = df_soil.groupby('District')[['Nitrogen Value', 'Phosphorous value', 'Potassium value']].mean().reset_index()

df_prices['Date'] = pd.to_datetime(df_prices['Date'])
df_prices['Year'] = df_prices['Date'].dt.year
df_prices['State'] = df_prices['State'].str.strip().str.title()
df_prices['Crop'] = df_prices['Crop Type'].str.strip().str.title()
df_prices_agg = df_prices.groupby(['State', 'Year', 'Crop'])[['Price (₹/ton)', 'Fertilizer Usage (kg/hectare)']].mean().reset_index()

# merge
df_merged = pd.merge(df_crop, df_soil_agg, on='District', how='left')
df_merged = pd.merge(df_merged, df_temp_agg, on=['State', 'Year'], how='left')
df_merged = pd.merge(df_merged, df_rain[['State', 'Year', 'ANNUAL']], on=['State', 'Year'], how='left')
df_merged.rename(columns={'ANNUAL': 'Rainfall'}, inplace=True)
df_merged = pd.merge(df_merged, df_prices_agg, on=['State', 'Year', 'Crop'], how='left')

print("Merged shape:", df_merged.shape)
