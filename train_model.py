import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import logging

logging.basicConfig(filename="train.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
np.random.seed(42)
DATA_PATH = "data/lung_cancer_new.csv"
MODEL_PATH = "models/voting_model.pkl"
SCALER_PATH = "models/scaler.pkl"
logging.info("Starting train_model.py")

if not os.path.exists(DATA_PATH):
    logging.error(f"Dataset not found at {DATA_PATH}")
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

logging.info(f"Loading dataset from {DATA_PATH}")
try:
    data = pd.read_csv(DATA_PATH)
    logging.info(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")
except Exception as e:
    logging.error(f"Error loading dataset: {e}")
    raise

expected_columns = ["age", "sex", "smoking", "persistent_cough", "fatigue", "cough_blood", "chest_pain",
                    "weight_loss", "tumor_size", "alk_phosphate", "sgot", "lung_function", "tumor_marker",
                    "histology", "class"]
if list(data.columns) != expected_columns:
    logging.error(f"Dataset columns mismatch. Expected: {expected_columns}, Got: {list(data.columns)}")
    raise ValueError("Dataset columns do not match expected features")

X = data.drop("class", axis=1)
y = data["class"]
logging.info("Data split into X and y")

scaler = StandardScaler()
numerical_cols = ["age", "tumor_size", "alk_phosphate", "sgot", "lung_function", "tumor_marker"]
try:
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    logging.info("Numerical features scaled")
except Exception as e:
    logging.error(f"Error scaling features: {e}")
    raise

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info(f"Data split: {len(X_train)} train, {len(X_test)} test")

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)
voting_clf = VotingClassifier(estimators=[("rf", rf), ("gb", gb), ("lr", lr)], voting="soft", n_jobs=-1)

try:
    voting_clf.fit(X_train, y_train)
    logging.info("Model training completed")
except Exception as e:
    logging.error(f"Error training model: {e}")
    raise

y_pred = voting_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Model accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))

try:
    os.makedirs("models", exist_ok=True)
    logging.info(f"Saving model to {MODEL_PATH}")
    joblib.dump(voting_clf, MODEL_PATH)
    file_size = os.path.getsize(MODEL_PATH) / 1024
    logging.info(f"Model saved: {MODEL_PATH}, Size: {file_size:.2f} KB")
except Exception as e:
    logging.error(f"Error saving model: {e}")
    raise

try:
    logging.info(f"Saving scaler to {SCALER_PATH}")
    joblib.dump(scaler, SCALER_PATH)
    file_size = os.path.getsize(SCALER_PATH) / 1024
    logging.info(f"Scaler saved: {SCALER_PATH}, Size: {file_size:.2f} KB")
except Exception as e:
    logging.error(f"Error saving scaler: {e}")
    raise

print(f"Model and scaler saved: {MODEL_PATH}, {SCALER_PATH}")