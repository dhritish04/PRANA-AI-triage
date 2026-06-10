import joblib

model = joblib.load("ecg_model.pkl")

print(model.feature_names_in_)