import joblib

model = joblib.load("traffic_model.pkl")

print("Model loaded successfully!")

import joblib

# Load trained model
model = joblib.load("traffic_model.pkl")

print("Traffic Demand Predictor")
print("-" * 30)

geohash = int(input("Enter geohash value: "))
day = int(input("Enter day value: "))
roadtype = int(input("Enter RoadType value: "))
lanes = int(input("Enter Number of Lanes: "))
largevehicles = int(input("Enter LargeVehicles value: "))
landmarks = int(input("Enter Landmarks value: "))
temperature = float(input("Enter Temperature: "))
weather = int(input("Enter Weather value: "))
hour = int(input("Enter Hour (0-23): "))
minute = int(input("Enter Minute (0-59): "))

sample = [[
    geohash,
    day,
    roadtype,
    lanes,
    largevehicles,
    landmarks,
    temperature,
    weather,
    hour,
    minute
]]
prediction = model.predict(sample)

demand = prediction[0]

print("\nPredicted Traffic Demand:", round(demand,4))
if demand < 0.3:
    status = "LOW TRAFFIC"

elif demand < 0.7:
    status = "MODERATE TRAFFIC"

else:
    status = "HIGH TRAFFIC"

print("Traffic Status:", status)

if demand < 0.3:
    green_time = 30

elif demand < 0.7:
    green_time = 60

else:
    green_time = 90

print("Recommended Green Signal Time:", green_time, "seconds")
if demand > 0.8:
    print("ALERT: Heavy Congestion Expected")