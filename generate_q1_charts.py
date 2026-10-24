import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
from sklearn.linear_model import LinearRegression
import seaborn as sns
from matplotlib.colors import Normalize

sns.set_style("whitegrid")

# Load Data
df = pd.read_csv('/Users/yudhapratama/Documents/Science/Publikasi/clean-water-access/data/clean_water.csv')
df_cluster = pd.read_csv('/Users/yudhapratama/Documents/Science/Publikasi/clean-water-access/Clean-Water-Access/Hasil_KMeans_AksesAir.csv')
solid1_daerah = df_cluster[df_cluster['Cluster_KMeans'] == 0]['daerah'].tolist()

df_2020 = df[(df['year'] == 2020) & (df['daerah'].isin(solid1_daerah))]
bottom14 = df_2020.nsmallest(14, 'water')['daerah'].tolist()

with open('/Users/yudhapratama/Documents/Science/Publikasi/clean-water-access/Clean-Water-Access/Model_AirBersih_.pkl', 'rb') as file:
    model_forecast = pickle.load(file)
model_forecast.named_steps['preprocessing'].transformers_[0][1].handle_unknown = 'ignore'

def project_feature(df_daerah, feature, years_future):
    X = df_daerah['year'].values.reshape(-1, 1)
    y = df_daerah[feature].values
    reg = LinearRegression().fit(X, y)
    return reg.predict(np.array(years_future).reshape(-1, 1))

# Generate predictions for all years between 2021 and 2045 for a smooth line
forecast_years = list(range(2021, 2046)) 
all_forecasts = []

for daerah in bottom14:
    df_d = df[df['daerah'] == daerah].sort_values('year')
    last_row = df_d[df_d['year'] == 2020].iloc[0]

    for yr in forecast_years:
        row = {
            'year': yr,
            'daerah': daerah,
            'tingkat': last_row['tingkat'],
            'status': last_row['status'],
            'logpop': project_feature(df_d, 'logpop', [yr])[0],
            'logarea': float(last_row['logarea']),
            'logpad': project_feature(df_d, 'logpad', [yr])[0],
            'morbid': project_feature(df_d, 'morbid', [yr])[0],
        }
        all_forecasts.append(row)

df_forecast = pd.DataFrame(all_forecasts)

features_order = ['logpop', 'logarea', 'logpad', 'year', 'morbid', 'daerah', 'tingkat', 'status']
df_forecast['water'] = model_forecast.predict(df_forecast[features_order])
df_forecast['water'] = df_forecast['water'].clip(0, 100)

# ==========================================
# CHART 1: 4x4 Grid in English
# ==========================================
ncols = 4
nrows = math.ceil(len(bottom14) / ncols)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, nrows * 4.5))
axes = axes.flatten()

for i, daerah in enumerate(bottom14):
    ax = axes[i]
    actual_data = df[df['daerah'] == daerah].sort_values('year')[['year', 'water']]
    # For annotations, we only annotate specific milestone years
    milestones = df_forecast[(df_forecast['daerah'] == daerah) & (df_forecast['year'].isin([2025, 2030, 2040, 2045]))].sort_values('year')[['year', 'water']]
    
    last_actual = actual_data[actual_data['year'] == 2020]
    full_forecast = df_forecast[df_forecast['daerah'] == daerah].sort_values('year')[['year', 'water']]
    connected_forecast = pd.concat([last_actual, full_forecast]).reset_index(drop=True)
    
    ax.plot(actual_data['year'], actual_data['water'], color='#1f77b4', lw=2.5, label='Actual (1996-2020)')
    ax.plot(connected_forecast['year'], connected_forecast['water'], color='#d62728', lw=2.5, ls='--', label='Projected (2021-2045)')
    
    ax.scatter(milestones['year'], milestones['water'], color='#d62728', s=40, zorder=5)
    
    for _, row in milestones.iterrows():
        ax.annotate(f"{row['water']:.1f}%",
                    xy=(row['year'], row['water']),
                    xytext=(0, 6), textcoords='offset points',
                    ha='center', fontsize=8, color='#d62728')
    
    ax.axvline(2020, color='gray', ls=':', alpha=0.8)
    ax.set_title(daerah, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.set_xlim(1995, 2050)
    ax.set_ylabel('Clean Water Access (%)', fontsize=9)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8, loc='upper left')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Clean Water Access Projection: 14 Least Developed Districts', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/Users/yudhapratama/Documents/Science/Publikasi/clean-water-access/Clean-Water-Access/forecast_english_grid.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# CHART 2: Single Consolidated Chart (Q1 Grade)
# ==========================================
fig, ax = plt.subplots(figsize=(12, 8))

# Shade the forecast region
ax.axvspan(2020, 2045, color='gray', alpha=0.08, label='_nolegend_', zorder=0)
ax.axvline(2020, color='gray', ls='--', lw=1.5, alpha=0.7, zorder=1)
ax.text(2020.5, 3, 'Forecast Horizon', color='dimgray', fontsize=11, fontstyle='italic')

# Sort districts by their 2045 projected value for color mapping
final_vals = []
for daerah in bottom14:
    val_2045 = df_forecast[(df_forecast['daerah'] == daerah) & (df_forecast['year'] == 2045)]['water'].values[0]
    final_vals.append((daerah, val_2045))
final_vals.sort(key=lambda x: x[1])

cmap = plt.get_cmap('viridis')
norm = Normalize(vmin=min([x[1] for x in final_vals]), vmax=max([x[1] for x in final_vals]))

for daerah, final_val in final_vals:
    actual_data = df[df['daerah'] == daerah].sort_values('year')
    full_forecast = df_forecast[df_forecast['daerah'] == daerah].sort_values('year')
    last_actual = actual_data[actual_data['year'] == 2020]
    connected_forecast = pd.concat([last_actual, full_forecast]).reset_index(drop=True)
    
    color = cmap(norm(final_val))
    
    # Smooth line plotting
    ax.plot(actual_data['year'], actual_data['water'], color=color, lw=2, alpha=0.85, zorder=3)
    ax.plot(connected_forecast['year'], connected_forecast['water'], color=color, lw=2, ls='--', alpha=0.85, zorder=3)
    
    # Label at the end of the line
    ax.text(2045.5, final_val, f" {daerah} ({final_val:.1f}%)", color=color, fontsize=9, va='center', fontweight='bold')

ax.set_ylim(0, 105)
ax.set_xlim(1995, 2056) # Give space for text labels
ax.set_ylabel('Clean Water Access (%)', fontsize=13, fontweight='bold', labelpad=10)
ax.set_xlabel('Year', fontsize=13, fontweight='bold', labelpad=10)
ax.set_title('Clean Water Access Trajectories and Projections for the 14 Least Developed Districts', fontsize=15, fontweight='bold', pad=20)
ax.tick_params(axis='both', which='major', labelsize=11)

# Custom legend for line styles
import matplotlib.lines as mlines
actual_line = mlines.Line2D([], [], color='black', lw=2, label='Actual Data (1996-2020)')
proj_line = mlines.Line2D([], [], color='black', lw=2, ls='--', label='Projections (2021-2045)')
ax.legend(handles=[actual_line, proj_line], loc='upper left', frameon=True, fontsize=11, framealpha=0.9, edgecolor='gray')

# Clean up axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/Users/yudhapratama/Documents/Science/Publikasi/clean-water-access/Clean-Water-Access/forecast_english_single.png', dpi=300, bbox_inches='tight')
plt.close()

print("Charts generated successfully!")
