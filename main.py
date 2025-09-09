import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load datasets
practice1 = pd.read_csv('practice1.csv')
qualifying = pd.read_csv('qualifying.csv')
starting_grid = pd.read_csv('startinggrid.csv')
fastestlaps = pd.read_csv('fastestlaps.csv')
pitstop = pd.read_csv('pitstop.csv')
raceresult = pd.read_csv('raceresult.csv')

# Create 'winner' column in raceresult
raceresult['winner'] = (raceresult['pos'] == 1).astype(int)

# Merge datasets
df = raceresult.merge(practice1[['race_id', 'driver', 'pos']], on=['race_id', 'driver'], how='left')
df = df.rename(columns={'pos_x': 'race_pos', 'pos_y': 'practice1_pos'})

df = df.merge(qualifying[['race_id', 'driver', 'pos']], on=['race_id', 'driver'], how='left')
df = df.rename(columns={'pos': 'qualifying_pos'})

df = df.merge(starting_grid[['race_id', 'driver', 'grid_pos']], on=['race_id', 'driver'], how='left')

# Fastest lap feature
df['fastest_lap'] = df.apply(lambda row: 1 if (
    (row['race_id'], row['driver']) in 
    set(zip(fastestlaps['race_id'], fastestlaps['driver']))
) else 0, axis=1)

# Pit stop features
pit_counts = pitstop.groupby(['event_id', 'driver']).size().reset_index(name='pit_count')
df = df.merge(pit_counts, left_on=['race_id', 'driver'], right_on=['event_id', 'driver'], how='left')
df['pit_count'] = df['pit_count'].fillna(0)

# Drop unused columns
df_model = df[['practice1_pos', 'qualifying_pos', 'grid_pos', 'fastest_lap', 'pit_count', 'winner']].dropna()

# Split data
X = df_model.drop(columns='winner')
y = df_model['winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict Dutch GP (example)
dutch_gp = df[df['race_id'] == '2025_dutch_gp']
dutch_gp_features = dutch_gp[['practice1_pos', 'qualifying_pos', 'grid_pos', 'fastest_lap', 'pit_count']].fillna(0)
dutch_predictions = model.predict(dutch_gp_features)

# Show predicted winner
dutch_gp['predicted_winner'] = dutch_predictions
winner_row = dutch_gp[dutch_gp['predicted_winner'] == 1]
print("\n🏁 Predicted Dutch GP Winner:")
print(winner_row[['driver', 'team', 'grid_pos']])
