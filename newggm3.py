import pandas as pd
import matplotlib.pyplot as plt


# DATA LOADING AND CLEANING

try:
    df = pd.read_csv('queensland_data.csv')
except FileNotFoundError:
    print("Error: 'queensland_data.csv' not found.")
    exit()

df['date'] = pd.to_datetime(df['date'])
df = df[df['date'] >= '2018-01-01'].copy()
df.set_index('date', inplace=True)

columns_to_drop = [col for col in df.columns if 'Emissions Vol' in col or 'Market Value' in col or 'Price' in col or 'Intensity' in col or 'Curtailment' in col or 'Temperature' in col]
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

cleaned_columns = {}
for col in df.columns:
    new_name = col.lower().strip()
    new_name = new_name.replace(' - ', '_').replace(' ', '_').replace('(', '').replace(')', '')
    cleaned_columns[col] = new_name
df.rename(columns=cleaned_columns, inplace=True)

df.to_csv('cleaned_queensland_data.csv')

print("Data Cleaning Complete. Cleaned data saved to 'cleaned_queensland_data.csv'")


# DATA ANALYSIS AND VISUALIZATION


print("\nStarting Analysis and Visualization")

df = pd.read_csv('cleaned_queensland_data.csv', index_col='date', parse_dates=True)
print("\nCleaned Columns Found:", df.columns.tolist())


def find_col(df, keyword):
    try:
        return [col for col in df.columns if keyword in col][0]
    except IndexError:
        print(f"FATAL ERROR: A critical column containing '{keyword}' was not found. Exiting.")
        exit()


rooftop_solar_col = find_col(df, 'solar_rooftop')
utility_solar_col = find_col(df, 'solar_utility')
wind_col = find_col(df, 'wind')
hydro_col = find_col(df, 'hydro')
bioenergy_col = find_col(df, 'bioenergy')
coal_col = find_col(df, 'coal_black')
gas_ccgt_col = find_col(df, 'gas_ccgt')
gas_ocgt_col = find_col(df, 'gas_ocgt')
distillate_col = find_col(df, 'distillate')
imports_col = find_col(df, 'imports')
exports_col = find_col(df, 'exports')

# Analysis 1: The Rise of Renewables
renewable_sources = [hydro_col, wind_col, utility_solar_col, rooftop_solar_col, bioenergy_col]
fossil_sources = [coal_col, gas_ccgt_col, gas_ocgt_col, distillate_col]

df['total_renewable_gwh'] = df[renewable_sources].sum(axis=1)
df['total_fossil_gwh'] = df[fossil_sources].sum(axis=1)
df['total_generation_gwh'] = df['total_renewable_gwh'] + df['total_fossil_gwh']

plt.style.use('seaborn-v0_8-whitegrid')
fig1, ax1 = plt.subplots(figsize=(14, 8))
ax1.stackplot(df.index, df[rooftop_solar_col], df[utility_solar_col], df[wind_col], df[hydro_col],
              labels=['Rooftop Solar', 'Utility Solar', 'Wind', 'Hydro'],
              colors=['#FFC300', '#FFD700', '#0077B6', '#00B4D8'])
ax1.set_title('The Rise of Renewable Energy Generation in Queensland (2018-Present)', fontsize=16)
ax1.set_ylabel('Daily Electricity Generation (GWh)')
ax1.set_xlabel('Year')
ax1.legend(loc='upper left')
fig1.tight_layout()
fig1.savefig('1_rise_of_renewables.png')
print("Plot 1: '1_rise_of_renewables.png' saved successfully.")

# Analysis 2: Impact of Solar on Grid Demand
generation_cols = [col for col in df.columns if '_gwh' in col and 'charging' not in col]
df['total_demand_gwh'] = df[generation_cols].sum(axis=1) + df[imports_col] - df[exports_col]
df['total_solar_gwh'] = df[rooftop_solar_col] + df[utility_solar_col]
df['operational_demand_gwh'] = df['total_demand_gwh'] - df['total_solar_gwh']

fig2, ax2 = plt.subplots(figsize=(14, 8))
ax2.plot(df.index, df['total_demand_gwh'].rolling(window=30).mean(), label='Total Demand (30-day average)', color='grey', linestyle='--')
ax2.plot(df.index, df['operational_demand_gwh'].rolling(window=30).mean(), label='Operational Demand after Solar (30-day average)', color='blue', linewidth=2)
ax2.set_title("Impact of Solar Generation on Queensland's Grid Demand", fontsize=16)
ax2.set_ylabel('Average Daily Demand (GWh)')
ax2.set_xlabel('Year')
ax2.legend()
fig2.tight_layout()
fig2.savefig('2_solar_impact_on_demand.png')
print("Plot 2: '2_solar_impact_on_demand.png' saved successfully.")

# Analysis 3: Seasonal Consumption Patterns
monthly_avg_generation = df['total_generation_gwh'].resample('M').mean()
fig3, ax3 = plt.subplots(figsize=(14, 8))
monthly_avg_generation.plot(kind='bar', ax=ax3, color='teal')
ax3.set_title('Average Daily Electricity Generation by Month in Queensland', fontsize=16)
ax3.set_ylabel('Average Daily Generation (GWh)')
ax3.set_xlabel('Month')
ax3.set_xticklabels([d.strftime('%Y-%b') for d in monthly_avg_generation.index], rotation=45, ha='right')
fig3.tight_layout()
fig3.savefig('3_seasonal_patterns.png')
print("Plot 3: '3_seasonal_patterns.png' saved successfully.")

# Analysis 4: The Changing Energy Mix
df_yearly = df.resample('Y').sum()
df_yearly['total_generation_gwh'] = df_yearly['total_fossil_gwh'] + df_yearly['total_renewable_gwh']
df_yearly['coal_%'] = (df_yearly[coal_col] / df_yearly['total_generation_gwh']) * 100
df_yearly['gas_%'] = ((df_yearly.get(gas_ccgt_col, 0) + df_yearly.get(gas_ocgt_col, 0)) / df_yearly['total_generation_gwh']) * 100
df_yearly['solar_%'] = (df_yearly['total_solar_gwh'] / df_yearly['total_generation_gwh']) * 100
df_yearly['wind_%'] = (df_yearly[wind_col] / df_yearly['total_generation_gwh']) * 100
df_yearly['hydro_%'] = (df_yearly[hydro_col] / df_yearly['total_generation_gwh']) * 100

energy_mix = df_yearly[['coal_%', 'gas_%', 'solar_%', 'wind_%', 'hydro_%']]
fig4, ax4 = plt.subplots(figsize=(14, 8))
energy_mix.plot(kind='bar', stacked=True, ax=ax4,
                color=['#333333', '#F4A261', '#FFC300', '#0077B6', '#00B4D8'])
ax4.set_title("Queensland's Changing Annual Energy Mix", fontsize=16)
ax4.set_ylabel('Percentage of Total Annual Generation (%)')
ax4.set_xlabel('Year')
ax4.set_xticklabels([d.strftime('%Y') for d in energy_mix.index], rotation=0)
ax4.legend(title='Energy Source', bbox_to_anchor=(1.05, 1), loc='upper left')
fig4.tight_layout()
fig4.savefig('4_changing_energy_mix.png')
print("Plot 4: '4_changing_energy_mix.png' saved successfully.")
