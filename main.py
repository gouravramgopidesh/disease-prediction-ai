import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("data/dataset.csv")

# Split
X = df.drop("disease", axis=1)
y = df["disease"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
pickle.dump(model, open("model/model.pkl", "wb"))

print("Model trained on correct dataset ✅")