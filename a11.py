import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Load Dataset
cols = ["unit", "cycle", "set1", "set2", "set3"] + ["s" + str(i) for i in range(1, 22)]

df = pd.read_csv(
    "train_FD001.txt",
    sep=r"\s+",
    header=None,
    names=cols
)

print("Dataset shape:", df.shape)

# Calculate Remaining Useful Life (RUL)
df["RUL"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]

# Cap RUL at 125 cycles
df["RUL"] = df["RUL"].clip(upper=125)

# Select Sensor Inputs
inputs = [
    "s2", "s3", "s4", "s7", "s8", "s9",
    "s11", "s12", "s13", "s14", "s15",
    "s17", "s20", "s21"
]

X = df[inputs]
y = df["RUL"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate Model
print("R2 Score:", round(r2_score(y_test, y_pred), 3))
print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))

# Display Coefficients
print("\nSensor Coefficients:")
for sensor, coef in zip(inputs, model.coef_):
    print(sensor, ":", round(coef, 4))

# Predict RUL for a New Engine
new_engine = pd.DataFrame([{
    "s2": 642.5,
    "s3": 1590,
    "s4": 1408,
    "s7": 553.5,
    "s8": 2388.1,
    "s9": 9060,
    "s11": 47.5,
    "s12": 521.5,
    "s13": 2388.1,
    "s14": 8140,
    "s15": 8.44,
    "s17": 393,
    "s20": 38.8,
    "s21": 23.3
}])

predicted_rul = model.predict(new_engine)

print("\nPredicted RUL for New Engine:", round(predicted_rul[0]), "cycles")

# Plot Actual vs Predicted RUL
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, label="Predictions")

plt.plot(
    [0, 125],
    [0, 125],
    "r--",
    label="Perfect Prediction"
)

plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Remaining Useful Life Prediction (125-Cycle Cap)")
plt.legend()
plt.grid(True)
plt.show()