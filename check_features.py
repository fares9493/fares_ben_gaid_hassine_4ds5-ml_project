import joblib

# Load the trained scaler
scaler = joblib.load("scaler.pkl")

# Print the expected number of features
print("Expected number of features:", scaler.n_features_in_)
