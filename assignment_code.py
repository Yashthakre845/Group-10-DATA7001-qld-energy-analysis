# assignment_code.py
# Import necessary libraries
import pandas as pd
import numpy as np

# 1. DATA ACQUISITION
print("1. Loading raw data...")
file_path = "19981201 Open Electricity (4).csv"
df = pd.read_csv(file_path)

print(f"Raw data shape: {df.shape}")

# 2. INITIAL EXPLORATION
print("\n2. Initial exploration of raw data...")
print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nSummary of missing values per column:")
print(df.isnull().sum())

# 3. DATA CLEANING
print("\n3. Starting data cleaning process...")

print("-> Filtering data for years 2018 and onwards...")
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
df = df[df['date'].dt.year >= 2018]
print(f"Shape after filtering for 2018 onwards: {df.shape}")

print("-> Dropping completely empty rows...")
df.dropna(how='all', inplace=True)
print(f"Shape after dropping empty rows: {df.shape}")

print("\nMissing values after initial cleaning:")
missing_data = df.isnull().sum()
print(missing_data[missing_data > 0])

print("-> Filling missing numeric values with 0...")
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

print("Missing values after filling NaNs:")
print(df.isnull().sum().sum())

print("-> Setting 'date' column as the index...")
df.set_index('date', inplace=True)

print("-> Checking for duplicate index entries...")
print(f"Number of duplicate dates: {df.index.duplicated().sum()}")

# 4. FOCUSING ON RELEVANT COLUMNS (WITH CORRECTED COLUMN NAMES)
print("\n4. Selecting relevant columns based on project scope...")

# CORRECTED COLUMN NAMES - MATCHING EXACTLY WHAT'S IN YOUR CSV
columns_to_keep = [
    'Coal (Brown) -  GWh',      # Note: Two spaces after dash
    'Coal (Black) -  GWh',      # Note: Two spaces after dash
    'Gas (CCGT) -  GWh',        # Note: Two spaces after dash
    'Gas (Steam) -  GWh',       # Note: Two spaces after dash
    'Hydro -  GWh',             # Note: Two spaces after dash
    'Wind -  GWh',              # Note: Two spaces after dash
    'Solar (Utility) -  GWh',   # Note: Two spaces after dash
    'Solar (Rooftop) -  GWh',   # Note: Two spaces after dash
    'Battery (Discharging) -  GWh',  # Note: Two spaces after dash
    'Volume Weighted Price - AUD/MWh',
    'Emissions Intensity - kgCO₂e/MWh'
]

df_focused = df[columns_to_keep].copy()
print(f"Focused dataset shape: {df_focused.shape}")
print("\nColumns in focused dataset:")
print(df_focused.columns.tolist())

# 5. FINAL DATA ENHANCEMENTS
print("\n5. Adding calculated columns for analysis...")

generation_columns = [col for col in df_focused.columns if 'GWh' in col and 'Battery' not in col]
df_focused['Total Generation - GWh'] = df_focused[generation_columns].sum(axis=1)

df_focused['Rooftop Solar %'] = (df_focused['Solar (Rooftop) -  GWh'] / df_focused['Total Generation - GWh']) * 100

df_focused['Total Solar - GWh'] = df_focused['Solar (Utility) -  GWh'] + df_focused['Solar (Rooftop) -  GWh']
df_focused['Total Solar %'] = (df_focused['Total Solar - GWh'] / df_focused['Total Generation - GWh']) * 100

df_focused['Total Renewables - GWh'] = df_focused['Total Solar - GWh'] + df_focused['Wind -  GWh'] + df_focused['Hydro -  GWh']
df_focused['Renewables %'] = (df_focused['Total Renewables - GWh'] / df_focused['Total Generation - GWh']) * 100

# 6. EXPORT CLEANED DATA
print("\n6. Exporting cleaned and focused dataset...")
cleaned_file_path = "QLD_Energy_2018_2025_CLEAN_FOCUSED.csv"
df_focused.to_csv(cleaned_file_path)
print(f"Cleaned and focused data saved to: {cleaned_file_path}")

# 7. FINAL PREVIEW
print("\n7. Final preview of the cleaned and focused dataset:")
print(f"Final shape: {df_focused.shape}")
print("\nFirst 5 rows:")
print(df_focused.head())
print("\nDataset description:")
print(df_focused.describe())

print("\nCleaning and enhancement complete! The data is now ready for analysis.")