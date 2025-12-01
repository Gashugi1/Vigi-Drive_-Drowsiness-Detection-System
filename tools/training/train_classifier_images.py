import argparse
import os
import pandas as pd
import joblib
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Define the feature columns used by the model
FEATURE_COLUMNS = ["ear", "mar", "eye_close", "mouth_open"]

def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Logistic Regression model for Drowsiness Detection."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="features/features_images.csv",
        help="Path to the input CSV file containing extracted features."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="models/earmar_img_logreg.joblib",
        help="Path where the trained model bundle (.joblib) will be saved."
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split (e.g., 0.2 for 20%%)."
    )
    return parser.parse_args()

def load_data(csv_path: Path) -> pd.DataFrame:
    """Loads the CSV data and performs initial checks."""
    if not csv_path.exists():
        print(f"Error: The input feature file '{csv_path}' does not exist!")
        # If the CSV is missing, we must exit and tell the user to run the extraction script again.
        print("\nPlease run the feature extraction script first: 'python training/extract_features_images.py --root /path/to/data --out features/features_images.csv'")
        sys.exit(1)
        
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    if df.empty:
        print("Error: The CSV file is empty!")
        sys.exit(1)
        
    # Check for required columns
    missing_cols = [col for col in FEATURE_COLUMNS + ["label"] if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in CSV: {missing_cols}")
        sys.exit(1)

    print(f"[Data] Loaded {len(df)} samples from {csv_path.name}")
    print("First few rows of the dataset:")
    print(df.head())
    
    return df

def train_and_save_model(df: pd.DataFrame, output_path: Path, test_size: float):
    """
    Trains the Logistic Regression model, evaluates it, and saves the resulting bundle.
    """
    
    # 1. Prepare Data
    X = df[FEATURE_COLUMNS].values
    y = df["label"].values
    
    print(f"[Data] Feature shape: {X.shape}")
    print(f"[Data] Labels shape: {y.shape}")

    # 2. Split Data
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    print(f"[Split] Training set size: {Xtr.shape[0]}")
    print(f"[Split] Test set size: {Xte.shape[0]}")

    # 3. Scale Features
    # The scaler must be fitted on the training data ONLY!
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    print("[Scale] Feature scaler trained and applied.")

    # 4. Train Model
    clf = LogisticRegression(max_iter=1000, random_state=42).fit(Xtr_s, ytr)
    print(f"[Train] Logistic Regression trained.")
    print(f"[Train] Model coefficients: {clf.coef_}")

    # 5. Evaluate Model
    y_pred = clf.predict(Xte_s)
    print("\n--- Model Classification Report ---")
    print(classification_report(yte, y_pred, digits=3))
    print("-----------------------------------")

    # 6. Save Model Bundle
    # Create the output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Bundle contains scaler (for new data), model, and feature order (metadata)
    model_bundle = {
        "scaler": scaler, 
        "model": clf, 
        "feature_order": FEATURE_COLUMNS
    }
    
    joblib.dump(model_bundle, output_path)
    print(f"\n[SUCCESS] Model saved to {output_path.resolve()}")

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Use pathlib for robust path handling
    csv_path = Path(args.csv)
    output_path = Path(args.out)
    
    df = load_data(csv_path)
    train_and_save_model(df, output_path, args.test_size)

if __name__ == "__main__":
    main()
