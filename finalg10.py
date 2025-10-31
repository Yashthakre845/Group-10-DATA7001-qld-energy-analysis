import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np



print("Data Loading and Cleaning")
try:
    df = pd.read_csv('queensland_data.csv')
except FileNotFoundError:
    print("Error: 'queensland_data.csv' not found.")
    exit()

df['date'] = pd.to_datetime(df['date'])
df = df[df['date'] >= '2018-01-01'].copy()
df.set_index('date', inplace=True)
df.dropna(how='all', inplace=True)

columns_to_drop = [col for col in df.columns if 'Emissions' in col or 'Market Value' in col or 'Price' in col or 'Intensity' in col or 'Curtailment' in col or 'Temperature' in col]
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

cleaned_columns = {}
for col in df.columns:
    new_name = col.lower().strip().replace(' - ', '_').replace(' ', '_').replace('(', '').replace(')', '')
    cleaned_columns[col] = new_name
df.rename(columns=cleaned_columns, inplace=True)

df.to_csv('cleaned_queensland_data.csv')
print("Data Cleaning Complete")




print("\nAnalysis and Feature Engineering")
df = pd.read_csv('cleaned_queensland_data.csv', index_col='date', parse_dates=True)


def find_col(df, keyword, fail_if_missing=False):
    try:
        return [col for col in df.columns if keyword in col][0]
    except IndexError:
        if fail_if_missing:
            print(f"FATAL ERROR: A critical column with '{keyword}' not found. Exiting.")
            exit()
        return None

rooftop_solar_col = find_col(df, 'solar_rooftop', True)
utility_solar_col = find_col(df, 'solar_utility', True)
wind_col = find_col(df, 'wind', True)
hydro_col = find_col(df, 'hydro', True)
bioenergy_col = find_col(df, 'bioenergy_biomass', True)
coal_col = find_col(df, 'coal_black', True)
imports_col = find_col(df, 'imports', True)
exports_col = find_col(df, 'exports', True)


renewable_sources = [hydro_col, wind_col, utility_solar_col, rooftop_solar_col, bioenergy_col]
fossil_sources = [col for col in df.columns if 'coal' in col or 'gas' in col or 'distillate' in col]
generation_cols = [col for col in df.columns if '_gwh' in col and 'charging' not in col]

df['total_renewable_gwh'] = df[renewable_sources].sum(axis=1)
df['total_fossil_gwh'] = df[fossil_sources].sum(axis=1)
df['total_generation_gwh'] = df['total_renewable_gwh'] + df['total_fossil_gwh']
df['total_solar_gwh'] = df[rooftop_solar_col] + df[utility_solar_col]
df['total_demand_gwh'] = df[generation_cols].sum(axis=1) + df[imports_col] - df[exports_col]
df['operational_demand_gwh'] = df['total_demand_gwh'] - df['total_solar_gwh']
print("aggregate columns created")


#VISUALIZATIONS

plt.style.use('seaborn-v0_8-whitegrid')


fig1, ax1 = plt.subplots(figsize=(14, 8))
ax1.stackplot(df.index, df[rooftop_solar_col], df[utility_solar_col], df[wind_col], df[hydro_col],
              labels=['Rooftop Solar', 'Utility Solar', 'Wind', 'Hydro'], colors=['#FFC300', '#FFD700', '#0077B6', '#00B4D8'])
ax1.set_title('The Rise of Renewable Energy Generation in Queensland (2018-Present)', fontsize=16)
ax1.set_ylabel('Daily Electricity Generation (GWh)'); ax1.set_xlabel('Year'); ax1.legend(loc='upper left')
fig1.tight_layout(); fig1.savefig('1_rise_of_renewables.png')
print("Plot 1: '1_rise_of_renewables.png' saved.")


fig2, ax2 = plt.subplots(figsize=(14, 8))
ax2.plot(df['total_demand_gwh'].rolling(window=30).mean(), label='Total Demand (30-day average)', color='grey', linestyle='--')
ax2.plot(df['operational_demand_gwh'].rolling(window=30).mean(), label='Operational Demand after Solar (30-day average)', color='blue', linewidth=2)
ax2.set_title("Impact of Solar Generation on Queensland's Grid Demand", fontsize=16)
ax2.set_ylabel('Average Daily Demand (GWh)'); ax2.set_xlabel('Year'); ax2.legend()
fig2.tight_layout(); fig2.savefig('2_solar_impact_on_demand.png')
print("Plot 2: '2_solar_impact_on_demand.png' saved.")

monthly_avg_generation = df['total_generation_gwh'].resample('M').mean()
fig3, ax3 = plt.subplots(figsize=(14, 8))
monthly_avg_generation.plot(kind='bar', ax=ax3, color='teal')
ax3.set_title('Average Daily Electricity Generation by Month in Queensland', fontsize=16)
ax3.set_ylabel('Average Daily Generation (GWh)'); ax3.set_xlabel('Month')
ax3.set_xticklabels([d.strftime('%Y-%b') for d in monthly_avg_generation.index], rotation=45, ha='right')
fig3.tight_layout(); fig3.savefig('3_seasonal_patterns.png')
print("Plot 3: '3_seasonal_patterns.png' saved.")

df_yearly = df.resample('Y').sum()
df_yearly['total_generation_gwh'] = df_yearly['total_fossil_gwh'] + df_yearly['total_renewable_gwh']
df_yearly['coal_%'] = (df_yearly[coal_col] / df_yearly['total_generation_gwh']) * 100
df_yearly['gas_%'] = ((df_yearly[[col for col in df.columns if 'gas' in col]].sum(axis=1)) / df_yearly['total_generation_gwh']) * 100
df_yearly['solar_%'] = (df_yearly['total_solar_gwh'] / df_yearly['total_generation_gwh']) * 100
df_yearly['wind_%'] = (df_yearly[wind_col] / df_yearly['total_generation_gwh']) * 100
df_yearly['hydro_%'] = (df_yearly[hydro_col] / df_yearly['total_generation_gwh']) * 100
energy_mix = df_yearly[['coal_%', 'gas_%', 'solar_%', 'wind_%', 'hydro_%']]
fig4, ax4 = plt.subplots(figsize=(14, 8))
energy_mix.plot(kind='bar', stacked=True, ax=ax4, color=['#333333', '#F4A261', '#FFC300', '#0077B6', '#00B4D8'])
ax4.set_title("Queensland's Changing Annual Energy Mix", fontsize=16)
ax4.set_ylabel('Percentage of Total Annual Generation (%)'); ax4.set_xlabel('Year')
ax4.set_xticklabels([d.strftime('%Y') for d in energy_mix.index], rotation=0)
ax4.legend(title='Energy Source', bbox_to_anchor=(1.05, 1), loc='upper left')
fig4.tight_layout(); fig4.savefig('4_changing_energy_mix.png')
print("Plot 4: '4_changing_energy_mix.png' saved.")


# PREDICTIVE MODELING


print("\n--- Starting Predictive Modeling ---")
model_df = df[['total_demand_gwh']].dropna().copy()
model_df['year'] = model_df.index.year
model_df['month'] = model_df.index.month
model_df['day_of_year'] = model_df.index.dayofyear
model_df['day_of_week'] = model_df.index.dayofweek
X = model_df[['year', 'month', 'day_of_year', 'day_of_week']]
y = model_df['total_demand_gwh']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Findings: Feature Coefficients")
coeffs = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coeffs)


model_df['predicted_demand_gwh'] = model.predict(X)
rmse = np.sqrt(mean_squared_error(y_test, model_df.loc[y_test.index, 'predicted_demand_gwh']))
print(f"\nModel Performance on Test Set (RMSE): {rmse:.2f} GWh")


fig5, ax5 = plt.subplots(figsize=(14, 8))
ax5.plot(model_df['total_demand_gwh'], label='Actual Demand', color='grey', alpha=0.9)

ax5.plot(y_train.index, model_df.loc[y_train.index, 'predicted_demand_gwh'], label='Fitted Model (Train Set)', color='blue', linestyle='--')

ax5.plot(y_test.index, model_df.loc[y_test.index, 'predicted_demand_gwh'], label='Predicted Demand (Test Set)', color='red', linestyle='--')
ax5.set_title('Model Performance: Predicting Daily Electricity Demand', fontsize=16)
ax5.set_ylabel('Daily Demand (GWh)'); ax5.set_xlabel('Date'); ax5.legend()
fig5.tight_layout(); fig5.savefig('5_model_performance.png')
print("Plot 5: '5_model_performance.png' saved with full context.")


print("\nanalysis complete")

