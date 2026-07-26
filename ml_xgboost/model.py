import joblib

# Load model pipeline kamu
pipeline = joblib.load(r"C:\Users\Adyty\Documents\Farid ITS\TriaGo\ml_xgboost\triage_model.joblib")

# 1. Cek nama dan urutan fitur yang diharapkan model
if hasattr(pipeline, "feature_names_in_"):
    print("=== URUTAN FITUR YANG DIBUTUHKAN MODEL ===")
    print(list(pipeline.feature_names_in_))

# 2. Cek urutan kelas label output
model = pipeline.named_steps['model']
print("\n=== URUTAN KELAS TARGET MODEL ===")
print(model.classes_)