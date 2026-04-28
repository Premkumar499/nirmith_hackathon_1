import os, sys, warnings, joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

K_SIZE = 4
CONFIDENCE_THR = 0.80
VALID_BASES = set("ACGTN")

def generate_kmers(sequence: str, k: int = K_SIZE) -> str:
    seq = "".join(c for c in sequence.upper().strip() if c in VALID_BASES)
    if len(seq) < k:
        return ""
    return " ".join(seq[i:i+k] for i in range(len(seq) - k + 1))

def sequences_to_kmers(sequences, k: int = K_SIZE) -> list:
    return [generate_kmers(s, k) for s in sequences]

def predict_sequences(
    sequences: list,
    tfidf: TfidfVectorizer,
    clf,
    iso: IsolationForest,
    le: LabelEncoder,
    top_n: int = 1,
) -> pd.DataFrame:
    kmer_sents = sequences_to_kmers(sequences)
    
    # Filter out empty kmer sequences and track valid indices
    valid_indices = [i for i, kmer in enumerate(kmer_sents) if kmer.strip()]
    valid_sequences = [sequences[i] for i in valid_indices]
    valid_kmers = [kmer_sents[i] for i in valid_indices]
    
    if not valid_kmers:
        return pd.DataFrame(columns=["sequence", "full_sequence", "prediction", "confidence", 
                                     "anomaly_score", "is_anomaly", "status"])
    
    X = tfidf.transform(valid_kmers)
    proba = clf.predict_proba(X)
    iso_scores = iso.decision_function(X)
    iso_pred = iso.predict(X)
    
    sorted_idx = np.argsort(proba, axis=1)[:, ::-1]
    
    results = []
    for i, seq in enumerate(valid_sequences):
        best_class_idx = sorted_idx[i, 0]
        conf = float(proba[i, best_class_idx])
        anomaly = (iso_pred[i] == -1)
        phylum = le.inverse_transform([best_class_idx])[0]
        
        if conf >= CONFIDENCE_THR and not anomaly:
            status = "KNOWN"
        elif conf >= CONFIDENCE_THR and anomaly:
            status = "KNOWN ⚠"
        else:
            status = "UNKNOWN"
        
        row = {
            "sequence": seq[:35] + "…" if len(seq) > 35 else seq,
            "full_sequence": seq,
            "prediction": phylum,
            "confidence": round(conf, 4),
            "anomaly_score": round(float(iso_scores[i]), 4),
            "is_anomaly": anomaly,
            "status": status,
        }
        
        if top_n > 1:
            for rank in range(min(top_n, proba.shape[1])):
                idx = sorted_idx[i, rank]
                row[f"top{rank+1}_phylum"] = le.inverse_transform([idx])[0]
                row[f"top{rank+1}_confidence"] = round(float(proba[i, idx]), 4)
        
        results.append(row)
    
    return pd.DataFrame(results)

def biodiversity_report(df: pd.DataFrame, output_path: str = "biodiversity_pie.png"):
    counts = df["prediction"].value_counts()
    total = len(df)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
           startangle=140, pctdistance=0.82)
    ax.set_title("Biodiversity Distribution", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return counts

def load_models(directory: str = "saved_models"):
    tfidf = joblib.load(f"{directory}/tfidf.pkl")
    clf = joblib.load(f"{directory}/classifier.pkl")
    iso = joblib.load(f"{directory}/novelty_detector.pkl")
    le = joblib.load(f"{directory}/label_encoder.pkl")
    return tfidf, clf, iso, le
