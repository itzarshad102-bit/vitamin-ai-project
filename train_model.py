import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("dataset.csv")

# Input features
X = data[["age", "sun_exposure", "fatigue", "bone_pain", "vegetarian"]]

# Multiple outputs
y = data[["vitamin_d_deficient", "vitamin_b12_deficient", "vitamin_c_deficient"]]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "vitamin_model.pkl")

# Accuracy
accuracy = model.score(X_test, y_test)

print("Multi-vitamin model trained")
print("Accuracy:", accuracy)
print("Model saved")