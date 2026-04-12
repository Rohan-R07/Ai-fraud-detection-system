import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Create model directory if it doesn't exist
if not os.path.exists("model"):
    os.makedirs("model")


def load_data():
    """Load dataset or create synthetic data for demonstration."""
    file_path = "fraud_data.csv"

    if os.path.exists(file_path):
        print(f"Loading dataset from {file_path}...")
        df = pd.read_csv(file_path)
    else:
        print("Dataset not found. Generating synthetic data for demonstration...")
        # Create synthetic data resembling Kaggle PaySim dataset
        data = {
            "step": np.random.randint(1, 100, 1000),
            "type": np.random.choice(
                ["CASH_OUT", "TRANSFER", "CASH_IN", "DEBIT", "PAYMENT"], 1000
            ),
            "amount": np.random.uniform(100, 100000, 1000),
            "isFraud": np.random.choice([0, 1], 1000, p=[0.95, 0.05]),
        }
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

    return df


def preprocess_data(df):
    """Handle categorical encoding and feature selection."""
    # Define mapping for 'type' column (consistent encoding for training and inference)
    type_map = {"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5}
    df["type"] = df["type"].map(type_map)

    # Features and Target
    features = ["step", "type", "amount"]
    X = df[features]
    y = df["isFraud"]

    return X, y, type_map


def train_model():
    # 1. Load data
    df = load_data()

    # 2. Preprocess
    X, y, type_map = preprocess_data(df)

    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Train RandomForest
    print("Training Random Forest model...")
    # model = RandomForestClassifier(
    #     n_estimators=100, class_weight="balanced", random_state=42
    # )

    model = RandomForestClassifier(class_weight="balanced")
    model.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Save model and metadata
    # We save the mapping too so the API knows how to encode the input
    model_data = {"model": model, "type_map": type_map}
    joblib.dump(model_data, "model/fraud_model.pkl")
    print("\nModel saved to model/fraud_model.pkl")


if __name__ == "__main__":
    train_model()
