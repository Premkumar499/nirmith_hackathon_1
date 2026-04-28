import os
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

try:
    from xgboost import XGBClassifier
    USE_XGB = True
except:
    USE_XGB = False


CSV_PATH = "cleaned_dataset.csv"
K_SIZE = 4
MAX_FEATURES = 8000
CONFIDENCE_THRESHOLD = 0.80


def generate_kmers(sequence, k=4):
    sequence = "".join([c for c in sequence.upper() if c in "ACGTN"])
    return " ".join(sequence[i:i+k] for i in range(len(sequence) - k + 1))


def sequences_to_kmers(sequences):
    return [generate_kmers(seq, K_SIZE) for seq in sequences]


def load_data():
    df = pd.read_csv(CSV_PATH)

    df = df[['sequence', 'phylum']].dropna()
    df['sequence'] = df['sequence'].astype(str).str.upper()

    df = df[df['sequence'].str.len() >= K_SIZE]
    df = df.drop_duplicates(subset="sequence")

    return df


def train_model():
    df = load_data()

    sequences = df['sequence'].tolist()
    labels = df['phylum'].tolist()

    le = LabelEncoder()
    y = le.fit_transform(labels)

    kmer_text = sequences_to_kmers(sequences)

    tfidf = TfidfVectorizer(max_features=MAX_FEATURES)
    X = tfidf.fit_transform(kmer_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    if USE_XGB:
        xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
        xgb.fit(X_train, y_train)
        model = xgb
    else:
        model = rf

    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X_train)

    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(tfidf, "saved_models/tfidf.pkl")
    joblib.dump(model, "saved_models/model.pkl")
    joblib.dump(iso, "saved_models/iso.pkl")
    joblib.dump(le, "saved_models/le.pkl")

    print("Training completed and models saved.")


def predict_sequences(input_sequences):
    tfidf = joblib.load("saved_models/tfidf.pkl")
    model = joblib.load("saved_models/model.pkl")
    iso = joblib.load("saved_models/iso.pkl")
    le = joblib.load("saved_models/le.pkl")

    kmer_text = sequences_to_kmers(input_sequences)
    X = tfidf.transform(kmer_text)

    probabilities = model.predict_proba(X)
    predictions = np.argmax(probabilities, axis=1)

    iso_result = iso.predict(X)

    results = []

    for i, seq in enumerate(input_sequences):
        confidence = float(np.max(probabilities[i]))
        predicted_label = le.inverse_transform([predictions[i]])[0]
        is_anomaly = iso_result[i] == -1

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
    print(df)
    return df


def biodiversity_report(df):
    counts = df['prediction'].value_counts()
    total = len(df)

    print("\nBiodiversity Report:")
    for label, count in counts.items():
        percentage = (count / total) * 100
        print(f"{label}: {percentage:.2f}% ({count})")


if __name__ == "__main__":
    
    train_model()

    test_sequences = [
        "ATGCGTACGTAGCTAGCTAG",
        "CGTAGCTAGCTAGGCTAACG",
        "TTTTTTTTTTTTTTTTTTTT"
    ]

    results = predict_sequences(test_sequences)

    biodiversity_report(results)
