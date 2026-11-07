import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

print("Fetching data...")
df = pd.read_csv("icu_mortality_prediction.csv", sep=None, engine='python')
for col in ["gender", "admission_type"]:
    if col in df.columns:
        df[col] = df[col].astype("category").cat.codes
df.columns = df.columns.str.strip()
target_label= df.columns[-1]
df = df.drop(columns=['patient_id'], errors='ignore')
model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth = 5,
    learning_rate=0.06,
    scale_pos_weight=3.5,
    subsample=0.8,
    cobsample_bytree=0.8,
    random_state=42,
    use_label_encoding = False
)

X = df.drop(columns=target_label)
Y = df[target_label]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model.fit(X_train, Y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
print(f"Model training complete! Accuracy: {accuracy*100:.2f}%")
print(f"Classification report: {classification_report(Y_test, predictions)}")

model_filename = "cardiosynth_model.pkl"

with open(model_filename, "wb") as file:
    pickle.dump(model, file)
print(f"Saved trained model as '{model_filename}' in your directory")