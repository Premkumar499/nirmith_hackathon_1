import os
import joblib
import numpy as np
import pandas as pd
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier
    USE_XGB = True
except:
    USE_XGB = False

CSV_PATH = "cleaned_dataset.csv"
K_SIZE = 4
MAX_FEATURES = 10000
CONFIDENCE_THRESHOLD = 0.80
BATCH_SIZE = 5000

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def generate_kmers(sequence, k=4):
    sequence = "".join([c for c in sequence.upper() if c in "ACGT"])
    return " ".join(sequence[i:i+k] for i in range(len(sequence) - k + 1))


def sequences_to_kmers(sequences):
    return [generate_kmers(seq, K_SIZE) for seq in sequences]


def load_data():
    logging.info("Loading dataset...")

    df = pd.read_csv(CSV_PATH, usecols=['sequence', 'phylum'])
    df = df.dropna()

    df['sequence'] = df['sequence'].astype(str).str.upper()
    df = df[df['sequence'].str.len() >= K_SIZE]

    df = df.drop_duplicates(subset="sequence")

    logging.info(f"Dataset size: {len(df)}")
    return df


def train_model():
    df = load_data()

    sequences = df['sequence'].tolist()
    labels = df['phylum'].tolist()

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    logging.info("Generating k-mers...")
    kmer_text = sequences_to_kmers(sequences)

    logging.info("TF-IDF vectorization...")
    tfidf = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        dtype=np.float32
    )
    X = tfidf.fit_transform(kmer_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    logging.info("Training model...")

    if USE_XGB:
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",  
            eval_metric='mlogloss',
            random_state=42
        )
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            random_state=42
        )

    model.fit(X_train, y_train)


    logging.info("Calibrating probabilities...")
    model = CalibratedClassifierCV(model, method='sigmoid')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    logging.info("Classification Report:\n" + classification_report(y_test, y_pred))


    logging.info("Training anomaly detector...")
    iso = IsolationForest(
        contamination=0.03,
        n_estimators=200,
        random_state=42
    )
    iso.fit(X_train)

    joblib.dump(tfidf, f"{MODEL_DIR}/tfidf.pkl")
    joblib.dump(model, f"{MODEL_DIR}/model.pkl")
    joblib.dump(iso, f"{MODEL_DIR}/iso.pkl")
    joblib.dump(le, f"{MODEL_DIR}/le.pkl")

    logging.info("Training completed and models saved.")


def predict_sequences(input_sequences):
    tfidf = joblib.load(f"{MODEL_DIR}/tfidf.pkl")
    model = joblib.load(f"{MODEL_DIR}/model.pkl")
    iso = joblib.load(f"{MODEL_DIR}/iso.pkl")
    le = joblib.load(f"{MODEL_DIR}/le.pkl")

    results = []

    logging.info("Running predictions...")

    for i in range(0, len(input_sequences), BATCH_SIZE):
        batch = input_sequences[i:i + BATCH_SIZE]

        kmer_text = sequences_to_kmers(batch)
        X = tfidf.transform(kmer_text)

        probabilities = model.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        iso_result = iso.predict(X)

        for j, seq in enumerate(batch):
            confidence = float(np.max(probabilities[j]))
            predicted_label = le.inverse_transform([predictions[j]])[0]
            is_anomaly = iso_result[j] == -1

            if confidence < CONFIDENCE_THRESHOLD:
                status = "UNKNOWN"
                predicted_label = "Unknown"
            elif is_anomaly:
                status = "KNOWN_CHECK"
            else:
                status = "KNOWN"

            results.append({
                "sequence": seq,
                "prediction": predicted_label,
                "confidence": round(confidence, 4),
                "status": status
            })

    df = pd.DataFrame(results)
    return df


def biodiversity_report(df):
    counts = df['prediction'].value_counts()
    total = len(df)

    report = []

    for label, count in counts.items():
        percentage = (count / total) * 100
        report.append({
            "phylum": label,
            "percentage": round(percentage, 2),
            "count": count
        })

    return pd.DataFrame(report)


if __name__ == "__main__":

    train_model()

    test_sequences = [
        "ATGCGTACGTAGCTAGCTAG",
        "CGTAGCTAGCTAGGCTAACG",
        "TTTTTTTTTTTTTTTTTTTT"
    ]

    results = predict_sequences(test_sequences)
    print("\nPrediction Results:\n", results)

    report = biodiversity_report(results)
    print("\nBiodiversity Report:\n", report)
