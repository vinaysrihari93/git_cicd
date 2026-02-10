import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv(r"D:\mlops_pipeline\data\data.csv")

X = df[["area", "bedrooms"]]
y = df["price"]

# Train model
model = LinearRegression()
model.fit(X, y)


# Save model
with open(r"D:\mlops_sample_cicid\backend\models\model.pkl", "wb") as f:
    pickle.dump(model, f)


print("✅ Linear Regression model trained and saved")