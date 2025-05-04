import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from matplotlib.patches import Rectangle
from datetime import datetime
warnings.filterwarnings('ignore')

# Setup
cache_dir = 'f1_cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# Load data from CSV files
drivers_df = pd.read_csv('drivers_2025.csv')
previous_races_df = pd.read_csv('2025_previous_races.csv', sep=',', skiprows=1, header=None)
previous_races_df.columns = ['position', 'driver', 'nationality', 'team', 'number', 'time', 'points', 'race']
miami_qualifiers_df = pd.read_csv('2025_miami_qualifiers.csv')

# Clean and process previous races data
previous_races_df = previous_races_df[previous_races_df['position'].notna()]
previous_races_df['position'] = previous_races_df['position'].astype(int)
previous_races_df['points'] = pd.to_numeric(previous_races_df['points'], errors='coerce').fillna(0)

# Process Miami qualifiers data
miami_qualifiers_df['Position'] = miami_qualifiers_df['Position'].astype(int)
miami_qualifiers_df['Driver'] = miami_qualifiers_df['Driver'].apply(lambda x: x.strip())

# Standardize team names across all data sources
team_mapping = {
    'McLaren': 'McLaren',
    'Red Bull': 'Red Bull Racing',
    'Red Bull Racing': 'Red Bull Racing',
    'Mercedes': 'Mercedes',
    'Ferrari': 'Ferrari',
    'Aston Martin': 'Aston Martin',
    'Williams': 'Williams',
    'Alpine': 'Alpine',
    'Haas': 'Haas F1 Team',
    'Haas F1 Team': 'Haas F1 Team',
    'Kick Sauber': 'Kick Sauber',
    'Stake': 'Kick Sauber',
    'Sauber': 'Kick Sauber',
    'RB': 'VCARB',
    'Racing Bulls': 'VCARB',
    'VCARB': 'VCARB'
}

# Apply team name standardization
previous_races_df['team'] = previous_races_df['team'].map(lambda x: team_mapping.get(x, x))
miami_qualifiers_df['Team'] = miami_qualifiers_df['Team'].map(lambda x: team_mapping.get(x, x))
drivers_df['Team'] = drivers_df['Team'].map(lambda x: team_mapping.get(x, x))

# Create team color mapping
team_colors = {
    'McLaren': '#FF8700',          # Orange
    'Red Bull Racing': '#0600EF',  # Dark blue
    'Ferrari': '#DC0000',          # Red
    'Mercedes': '#00D2BE',         # Turquoise
    'Aston Martin': '#006F62',     # British racing green
    'Williams': '#005AFF',         # Blue
    'Alpine': '#0090FF',           # Blue
    'Haas F1 Team': '#888888',     # Gray
    'Kick Sauber': '#900000',      # Burgundy
    'VCARB': '#2B4562'             # Navy blue
}

# Load historical Miami GP data (2022-2024)
miami_gp_data = []
for season in [2022, 2023, 2024]:
    try:
        # Find the Miami GP in each season
        season_calendar = fastf1.get_event_schedule(season)
        miami_events = season_calendar[season_calendar['EventName'].str.contains('Miami', case=False)]
        
        if not miami_events.empty:
            miami_event = miami_events.iloc[0]
            session = fastf1.get_session(season, miami_event['EventName'], 'R')
            session.load()
            
            # Get weather data if available
            weather_data = None
            try:
                weather_data = session.weather_data
                weather_data['Season'] = season
                weather_data['Circuit'] = 'Miami'
            except:
                pass
            
            results = session.results[['DriverNumber', 'Position', 'Points', 'GridPosition', 'Team']]
            results['Season'] = season
            results['Circuit'] = 'Miami'
            
            # Add weather data if available
            if weather_data is not None:
                avg_temp = weather_data['AirTemp'].mean()
                avg_humidity = weather_data['Humidity'].mean()
                results['AvgTemp'] = avg_temp
                results['AvgHumidity'] = avg_humidity
            
            miami_gp_data.append(results)
            print(f"Loaded Miami GP data for {season}")
        else:
            print(f"No Miami GP found for {season}")
            
    except Exception as e:
        print(f"Error loading Miami GP data for {season}: {e}")
        continue

# Combine historical data
if miami_gp_data:
    historical_miami_df = pd.concat(miami_gp_data)
    historical_miami_df['DriverNumber'] = historical_miami_df['DriverNumber'].astype(int)
    historical_miami_df['Position'] = pd.to_numeric(historical_miami_df['Position'], errors='coerce').fillna(25)
    historical_miami_df['GridPosition'] = pd.to_numeric(historical_miami_df['GridPosition'], errors='coerce').fillna(25)
else:
    # Create a dummy dataframe if no historical data could be loaded
    print("Warning: No historical Miami GP data could be loaded. Using dummy data.")
    historical_miami_df = pd.DataFrame({
        'DriverNumber': [],
        'Position': [],
        'Points': [],
        'GridPosition': [],
        'Team': [],
        'Season': [],
        'Circuit': []
    })

# Calculate driver form metrics from 2025 race data
driver_stats = {}
all_drivers = previous_races_df['driver'].unique()

for driver in all_drivers:
    driver_results = previous_races_df[previous_races_df['driver'] == driver]
    if not driver_results.empty:
        avg_position = driver_results['position'].mean()
        avg_points = driver_results['points'].mean()
        finishes = len(driver_results)
        team = driver_results.iloc[-1]['team']  # Get most recent team
        
        # Calculate position changes between races
        pos_changes = []
        race_positions = driver_results.sort_values('race')['position'].tolist()
        if len(race_positions) > 1:
            for i in range(1, len(race_positions)):
                pos_changes.append(race_positions[i-1] - race_positions[i])
        
        avg_pos_change = np.mean(pos_changes) if pos_changes else 0
        
        driver_stats[driver] = {
            'AvgPosition': avg_position,
            'AvgPoints': avg_points,
            'Races': finishes,
            'Team': team,
            'AvgPosChange': avg_pos_change,
            'LastRacePosition': driver_results.iloc[-1]['position'] if not driver_results.empty else 20
        }
    else:
        # Default values for drivers with no recent data
        driver_stats[driver] = {
            'AvgPosition': 15,
            'AvgPoints': 0,
            'Races': 0,
            'Team': "Unknown",
            'AvgPosChange': 0,
            'LastRacePosition': 20
        }

# Create a DataFrame from driver stats
drivers_stats_df = pd.DataFrame.from_dict(driver_stats, orient='index').reset_index()
drivers_stats_df.rename(columns={'index': 'FullName'}, inplace=True)

# Merge with driver numbers from drivers_df
drivers_stats_df = pd.merge(drivers_stats_df, drivers_df[['FullName', 'DriverNumber']], on='FullName', how='left')

# Add Miami-specific performance based on historical data
for i, driver in drivers_stats_df.iterrows():
    driver_number = driver['DriverNumber']
    
    # Get this driver's historical performance at Miami
    driver_miami_data = historical_miami_df[historical_miami_df['DriverNumber'] == driver_number]
    
    if not driver_miami_data.empty:
        # Calculate average position delta (grid to finish) at Miami
        pos_deltas = driver_miami_data['GridPosition'] - driver_miami_data['Position']
        avg_miami_delta = pos_deltas.mean()
        
        # Calculate average finish position at Miami
        avg_miami_position = driver_miami_data['Position'].mean()
        
        drivers_stats_df.at[i, 'MiamiAvgDelta'] = avg_miami_delta
        drivers_stats_df.at[i, 'MiamiAvgPosition'] = avg_miami_position
        drivers_stats_df.at[i, 'MiamiExperience'] = len(driver_miami_data)
    else:
        # No Miami data for this driver
        drivers_stats_df.at[i, 'MiamiAvgDelta'] = 0
        drivers_stats_df.at[i, 'MiamiAvgPosition'] = driver['AvgPosition']  # Use season average as fallback
        drivers_stats_df.at[i, 'MiamiExperience'] = 0

# Add qualifying data from Miami qualifiers
drivers_stats_df = pd.merge(drivers_stats_df, 
                           miami_qualifiers_df[['Driver', 'Position']].rename(columns={'Driver': 'FullName', 'Position': 'GridPosition'}),
                           on='FullName', how='left')

# Fill missing grid positions with estimated values based on recent form
missing_grid = drivers_stats_df['GridPosition'].isna()
if missing_grid.any():
    # Estimate grid position based on recent performance with some randomness
    np.random.seed(42)
    estimated_grid = drivers_stats_df[missing_grid]['LastRacePosition'] + np.random.normal(0, 2, size=sum(missing_grid))
    estimated_grid = np.clip(estimated_grid, 1, 20)
    estimated_grid_ranks = estimated_grid.rank(method='first').astype(int)
    drivers_stats_df.loc[missing_grid, 'GridPosition'] = estimated_grid_ranks

# Assign colors to each driver based on team
drivers_stats_df['Color'] = drivers_stats_df['Team'].map(team_colors)

# Prepare feature data for prediction
feature_data = []
for i, driver in drivers_stats_df.iterrows():
    driver_name = driver['FullName']
    team = driver['Team']
    grid_pos = driver['GridPosition']
    
    # Base prediction on combination of:
    # 1. Recent form (60%)
    # 2. Miami-specific performance (30%)
    # 3. Grid position (10%)
    form_component = driver['AvgPosition'] * 0.6
    
    # Use Miami-specific data if available, otherwise just use form
    if driver['MiamiExperience'] > 0:
        miami_component = driver['MiamiAvgPosition'] * 0.3
    else:
        # No Miami experience - rely more on recent form and add slight penalty for unknown track
        miami_component = driver['AvgPosition'] * 0.3 * 1.05  # 5% penalty
    
    grid_component = grid_pos * 0.1
    
    base_prediction = form_component + miami_component + grid_component
    
    # Adjust prediction based on recent position change tendency
    pos_change_adjustment = -0.2 * driver['AvgPosChange']  # If driver tends to gain positions, this is positive
    
    adjusted_prediction = base_prediction + pos_change_adjustment
    
    # Add uncertainty - higher for midfield, lower for top teams
    if driver['AvgPosition'] <= 5:
        uncertainty = 1.5  # Low uncertainty for top teams
    elif driver['AvgPosition'] <= 10:
        uncertainty = 2.0  # Medium uncertainty for midfield
    else:
        uncertainty = 2.5  # High uncertainty for backmarkers
    
    # Miami-specific adjustments
    # Miami has tight corners and long straights - good for balanced cars
    team_performance_factor = 1.0
    if team in ['Red Bull Racing', 'McLaren', 'Mercedes']:
        team_performance_factor = 0.95  # 5% advantage for top teams
    elif team in ['Ferrari', 'Aston Martin']:
        team_performance_factor = 0.98  # 2% advantage for upper midfield
    
    # Adjust prediction with team performance factor
    final_prediction = adjusted_prediction * team_performance_factor
    
    # Add to features
    feature_data.append({
        'DriverNumber': driver['DriverNumber'],
        'FullName': driver_name,
        'Team': team,
        'GridPosition': grid_pos,
        'Experience': driver['Races'],
        'MiamiExperience': driver['MiamiExperience'],
        'RecentAvgPos': driver['AvgPosition'],
        'RecentAvgPoints': driver['AvgPoints'],
        'PredictedPosition': final_prediction,
        'Uncertainty': uncertainty,
        'Color': driver['Color']
    })

# Create DataFrame with all features
df_prediction = pd.DataFrame(feature_data)

# Run simulation for race outcome
sim_count = 1000
all_results = []

for sim in range(sim_count):
    sim_results = df_prediction.copy()
    
    # Add random noise based on uncertainty
    sim_results['SimPosition'] = sim_results.apply(
        lambda x: max(1, x['PredictedPosition'] + np.random.normal(0, x['Uncertainty'])), 
        axis=1
    )
    
    # Factor in race incidents - Miami has medium chance of safety cars
    incident_mask = np.random.random(len(sim_results)) < 0.10
    sim_results.loc[incident_mask, 'SimPosition'] += np.random.randint(3, 10, size=sum(incident_mask))
    
    # Add first lap chaos factor
    first_lap_chaos = np.random.normal(0, 2.5, size=len(sim_results)) * (sim_results['GridPosition'] / 10)
    sim_results['SimPosition'] += first_lap_chaos
    
    # Sort by simulated position for this race
    race_result = sim_results.sort_values('SimPosition').reset_index(drop=True)
    race_result['SimFinish'] = race_result.index + 1
    
    all_results.append(race_result[['DriverNumber', 'FullName', 'SimFinish']])

# Aggregate results from all simulations
final_results = pd.concat(all_results)
avg_positions = final_results.groupby(['DriverNumber', 'FullName'])['SimFinish'].agg(['mean', 'std', 'min', 'max'])
avg_positions = avg_positions.reset_index()

# Join with driver and team info
final_df = pd.merge(avg_positions, df_prediction[['DriverNumber', 'FullName', 'Team', 'GridPosition', 'Color']], 
                   on=['DriverNumber', 'FullName'])

# Calculate probability of podium finish
podium_counts = {}
for sim_df in all_results:
    for driver in sim_df.loc[sim_df['SimFinish'] <= 3, 'FullName'].unique():
        podium_counts[driver] = podium_counts.get(driver, 0) + 1

# Convert to DataFrame and calculate percentages
podium_probs = pd.DataFrame(list(podium_counts.items()), columns=['FullName', 'PodiumCount'])
podium_probs['PodiumProbability'] = podium_probs['PodiumCount'] / sim_count * 100

# Merge with final results
final_df = pd.merge(final_df, podium_probs[['FullName', 'PodiumProbability']], 
                   on='FullName', how='left')
final_df['PodiumProbability'] = final_df['PodiumProbability'].fillna(0)

# Calculate points expectations
points_mapping = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

# Calculate expected points for each driver
expected_points = {}
for sim_df in all_results:
    for _, row in sim_df.iterrows():
        driver = row['FullName']
        position = row['SimFinish']
        points = points_mapping.get(position, 0)
        expected_points[driver] = expected_points.get(driver, 0) + points

# Convert to average expected points
for driver in expected_points:
    expected_points[driver] /= sim_count

# Add to final DataFrame
expected_points_df = pd.DataFrame(list(expected_points.items()), columns=['FullName', 'ExpectedPoints'])
final_df = pd.merge(final_df, expected_points_df, on='FullName', how='left')

# Sort by expected finish position
final_df = final_df.sort_values('mean').reset_index(drop=True)
final_df['ExpectedPosition'] = final_df.index + 1

# Function to add watermark to plots
def add_watermark(fig):
    fig.text(0.95, 0.05, 'powered by https://www.otto.rentals',
             fontsize=10, color='gray', alpha=0.5,
             ha='right', va='bottom', rotation=0)

# 1. 📈 Driver performance trends across the 2025 season
plt.figure(figsize=(14, 8))
for driver in final_df['FullName'].head(10):
    driver_data = previous_races_df[previous_races_df['driver'] == driver]
    if not driver_data.empty:
        color = final_df[final_df['FullName'] == driver]['Color'].values[0]
        races = driver_data['race'].unique()
        positions = [driver_data[driver_data['race'] == race]['position'].values[0] for race in races]
        plt.plot(races, positions, 'o-', label=driver, color=color, markersize=8)

plt.gca().invert_yaxis()
plt.title('2025 Season Performance Trends - Top 10 Drivers', fontsize=16)
plt.xlabel('Race', fontsize=12)
plt.ylabel('Finishing Position', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
add_watermark(plt.gcf())
plt.savefig('miami_gp_performance_trends.png', dpi=300, bbox_inches='tight')

# 2. 🏆 Predicted Top 10 Finishers
plt.figure(figsize=(12, 8))
top10 = final_df.head(10).sort_values('mean', ascending=True)

# Set up position ranges (error bars)
yerr = np.zeros((2, len(top10)))
yerr[0] = top10['mean'] - top10['min']  # bottom error
yerr[1] = top10['max'] - top10['mean']  # top error

bars = plt.barh(top10['FullName'], top10['mean'], 
                xerr=yerr,
                alpha=0.7,
                color=top10['Color'])

# Add probability percentages
for i, (_, row) in enumerate(top10.iterrows()):
    plt.text(0.5, i, f"{row['PodiumProbability']:.1f}% Pod", 
             va='center', ha='left', fontsize=10, color='black')
    plt.text(row['mean'] + 0.2, i, f"{row['ExpectedPoints']:.1f} Pts", 
             va='center', ha='left', fontsize=10, color='black')

plt.title('2025 Miami GP - Predicted Top 10 Finishers', fontsize=16)
plt.xlabel('Position', fontsize=12)
plt.ylabel('Driver', fontsize=12)
plt.xlim(0.5, 10.5)
plt.gca().invert_xaxis()
plt.grid(True, axis='x')
plt.tight_layout()
add_watermark(plt.gcf())
plt.savefig('miami_gp_top10_prediction.png', dpi=300)

# 3. 🌦️ Weather impact analysis (if historical weather data is available)
if 'AvgTemp' in historical_miami_df.columns:
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=historical_miami_df, x='AvgTemp', y='Position', hue='Team', palette=team_colors)
    plt.title('Historical Miami GP - Temperature Impact on Performance', fontsize=16)
    plt.xlabel('Average Temperature (°C)', fontsize=12)
    plt.ylabel('Finishing Position', fontsize=12)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    add_watermark(plt.gcf())
    plt.savefig('miami_gp_weather_impact.png', dpi=300, bbox_inches='tight')
else:
    print("No historical weather data available for Miami GP")

# 4. 🎯 Feature importance plot
# Prepare data for feature importance analysis
X = df_prediction[['GridPosition', 'Experience', 'MiamiExperience', 'RecentAvgPos', 'RecentAvgPoints']]
y = df_prediction['PredictedPosition']

# Train a Random Forest model to get feature importance
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_
features = X.columns

# Create feature importance plot
plt.figure(figsize=(10, 6))
plt.barh(features, importances, color='#FF8700')
plt.title('Feature Importance for Miami GP Predictions', fontsize=16)
plt.xlabel('Importance', fontsize=12)
plt.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
add_watermark(plt.gcf())
plt.savefig('miami_gp_feature_importance.png', dpi=300)

# Create a heatmap showing the distribution of predicted finishing positions
finish_distribution = pd.pivot_table(
    final_results, 
    index='FullName', 
    columns='SimFinish', 
    aggfunc='size', 
    fill_value=0
)

# Calculate percentage of simulations for each position
for col in finish_distribution.columns:
    finish_distribution[col] = finish_distribution[col] / sim_count * 100

# Get driver order based on expected position
driver_order = final_df['FullName'].tolist()
finish_distribution = finish_distribution.reindex(driver_order)

plt.figure(figsize=(14, 10))
sns.heatmap(finish_distribution, cmap='YlOrRd', annot=False, fmt='.1f', 
            linewidths=0.5, cbar_kws={'label': '% of Simulations'})

plt.title('Predicted Finishing Position Distribution - Miami GP 2025', fontsize=16)
plt.xlabel('Finishing Position', fontsize=12)
plt.ylabel('Driver', fontsize=12)
plt.tight_layout()
add_watermark(plt.gcf())
plt.savefig('miami_gp_position_heatmap.png', dpi=300)

# Show all plots
plt.show()

# Export predictions to CSV
final_df.to_csv('miami_gp_predictions.csv', index=False)

print("\nAnalysis complete! All visualizations have been saved.")