import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Load data
df = pd.read_csv("train.csv")

# Fill missing values
df['RoadType'] = df['RoadType'].fillna(df['RoadType'].mode()[0])
df['Weather'] = df['Weather'].fillna(df['Weather'].mode()[0])
df['Temperature'] = df['Temperature'].fillna(df['Temperature'].median())

# Create Hour and Minute
df[['Hour','Minute']] = df['timestamp'].str.split(':', expand=True)

df['Hour'] = df['Hour'].astype(int)
df['Minute'] = df['Minute'].astype(int)

# Encode categorical columns
columns = [
    'geohash',
    'day',
    'RoadType',
    'LargeVehicles',
    'Landmarks',
    'Weather'
]

encoders = {}

for col in columns:
    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(df[col])

    encoders[col] = encoder

# Features
X = df.drop(
    ['Index','demand','timestamp','Hour'],
    axis=1
)

# Target
y = df['demand']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print("R2 Score:", r2)
print("MAE:", mae)

print("\nActual vs Predicted:\n")

for i in range(10):
    print(
        "Actual:",
        round(y_test.iloc[i],4),
        "| Predicted:",
        round(predictions[i],4)
    )

importance = model.feature_importances_

for feature, score in zip(X.columns, importance):
    print(feature, round(score,4))
plt.figure(figsize=(10,5))

plt.plot(
    y_test.values[:100],
    label='Actual'
)

plt.plot(
    predictions[:100],
    label='Predicted'
)

plt.title('Actual vs Predicted Traffic Demand')

plt.xlabel('Sample Number')

plt.ylabel('Traffic Demand')

plt.legend()

plt.show()

feature_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

feature_df = feature_df.sort_values(
    by='Importance',
    ascending=False
)

plt.figure(figsize=(10,5))

plt.bar(
    feature_df['Feature'],
    feature_df['Importance']
)

plt.title("Feature Importance")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("images/feature_importance.png")

plt.show()

feature_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})


joblib.dump(
    model,
    "traffic_model.pkl"
)

print("Model Saved Successfully!")
print("\nMODEL COMPARISON")
print("-" * 40)

lr = LinearRegression()

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

lr_r2 = r2_score(y_test, lr_pred)

print("Linear Regression R2:", round(lr_r2,4))

dt = DecisionTreeRegressor(
    random_state=42
)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

dt_r2 = r2_score(y_test, dt_pred)

print("Decision Tree R2:", round(dt_r2,4))
print("Random Forest R2:", round(r2,4))

import joblib

model = joblib.load("traffic_model.pkl")

print("Model loaded successfully!")

models = ['Linear Regression', 'Decision Tree', 'Random Forest']

scores = [
    lr_r2,
    dt_r2,
    r2
]

plt.figure(figsize=(8,5))

plt.bar(
    models,
    scores
)

plt.title("Model Comparison")

plt.ylabel("R2 Score")

plt.savefig("images/model_comparison.png")

plt.close()

print("Model comparison graph saved!")

print("\nTRAFFIC INSIGHTS")
print("-" * 40)
peak_hour = df.groupby('Hour')['demand'].mean()

peak_hour = peak_hour.idxmax()

print(
    "Peak Traffic Hour:",
    f"{peak_hour}:00 - {peak_hour+1}:00"
)
road_traffic = df.groupby('RoadType')['demand'].mean()

best_road = road_traffic.idxmax()

road_name = encoders['RoadType'].inverse_transform(
    [best_road]
)[0]

print("Most Congested Road Type:", road_name)


weather_traffic = df.groupby('Weather')['demand'].mean()

best_weather = weather_traffic.idxmax()

weather_name = encoders['Weather'].inverse_transform(
    [best_weather]
)[0]

print(
    "Most Congested Weather Condition:",
    weather_name
)

with open("results.txt", "w") as f:

    f.write("MODEL COMPARISON\n")
    f.write("------------------------------\n")

    f.write(f"Linear Regression R2: {round(lr_r2,4)}\n")
    f.write(f"Decision Tree R2: {round(dt_r2,4)}\n")
    f.write(f"Random Forest R2: {round(r2,4)}\n\n")

    f.write("TRAFFIC INSIGHTS\n")
    f.write("------------------------------\n")

    f.write(
        f"Peak Traffic Hour: {peak_hour}:00 - {peak_hour+1}:00\n"
    )

    f.write(
        f"Most Congested Road Type: {road_name}\n"
    )

    f.write(
        f"Most Congested Weather Condition: {weather_name}\n"
    )

print("Results saved successfully!")