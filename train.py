
import os, sys, argparse, warnings, joblib, time
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, IsolationForest, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    balanced_accuracy_score, f1_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight

try:
    from xgboost import XGBClassifier
    XGBOOST = True
except ImportError:
    XGBOOST = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_OK = True
except ImportError:
    SMOTE_OK = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

CSV_PATH        = "cleaned_dataset.csv" 
SEQ_COLUMN      = "sequence"             
LABEL_COLUMN    = "phylum"               
K_SIZE          = 4                     
MAX_FEATURES    = 8000                  
CONFIDENCE_THR  = 0.80                  
ANOMALY_CONTAM  = 0.05                   
TEST_SIZE       = 0.20                  
RANDOM_STATE    = 42
MODEL_DIR       = "saved_models"
N_CV_FOLDS      = 5                      



VALID_BASES = set("ACGTN")

def generate_kmers(sequence: str, k: int = K_SIZE) -> str:
    """
    Convert a DNA string into a space-separated k-mer sentence.
    k=4 example:  ATGCGT  →  'ATGC TGCG GCGT'
    Non-ACGTN characters are stripped before sliding.
    """
    seq = "".join(c for c in sequence.upper().strip() if c in VALID_BASES)
    if len(seq) < k:
        return ""
    return " ".join(seq[i:i+k] for i in range(len(seq) - k + 1))


def sequences_to_kmers(sequences, k: int = K_SIZE) -> list:
    return [generate_kmers(s, k) for s in sequences]



def _find_column(df: pd.DataFrame, candidates: list):
    """Case-insensitive flexible column finder."""
    lower_map = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_dataset(csv_path: str) -> pd.DataFrame:
    print(f"\n{'═'*60}")
    print(f"    LOADING DATASET : {csv_path}")
    print(f"{'═'*60}")

    df = pd.read_csv(csv_path)
    print(f"  Raw shape        : {df.shape}")
    print(f"  Columns          : {df.columns.tolist()}")

    seq_col   = _find_column(df, [SEQ_COLUMN, "sequence", "dna", "seq", "nucleotide"])
    label_col = _find_column(df, [LABEL_COLUMN, "phylum", "label", "class", "target"])

    if not seq_col or not label_col:
        raise ValueError(
            f"Cannot find sequence/label columns.\n"
            f"Detected: {df.columns.tolist()}\n"
            f"Set SEQ_COLUMN and LABEL_COLUMN at the top of this script."
        )

    df = df[[seq_col, label_col]].rename(columns={seq_col: "sequence", label_col: "label"})
    df.dropna(subset=["sequence", "label"], inplace=True)
    df["sequence"] = df["sequence"].astype(str).str.upper().str.strip()

    before = len(df)
    df = df[df["sequence"].str.len() >= K_SIZE]
    df.drop_duplicates(subset="sequence", inplace=True)
    after = len(df)

    print(f"  Clean shape      : {df.shape}  (dropped {before - after} short/dup rows)")
    print(f"\n  LABEL DISTRIBUTION:")
    vc = df["label"].value_counts()
    for label, count in vc.items():
        bar = "▓" * int(count / vc.max() * 30)
        print(f"    {label:<20} {bar:<32} {count:>5}")
    print(f"\n  Sequence length  : min={df['sequence'].str.len().min()}  "
          f"mean={df['sequence'].str.len().mean():.0f}  "
          f"max={df['sequence'].str.len().max()}")
    return df


def build_tfidf(k: int = K_SIZE, max_features: int = MAX_FEATURES) -> TfidfVectorizer:
    """
    TF-IDF on k-mer 'words'.
    sublinear_tf=True  →  log(TF+1), smooths high-frequency k-mers
    min_df=2           →  ignore k-mers seen only once (noise)
    """
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 1),
        max_features=max_features,
        sublinear_tf=True,
        min_df=2,
        token_pattern=r"(?u)\b\w+\b",
    )


def _make_rf(n_classes: int) -> CalibratedClassifierCV:
    base = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return CalibratedClassifierCV(base, cv=3, method="sigmoid")


def _make_xgb(n_classes: int) -> CalibratedClassifierCV:
    base = XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    return CalibratedClassifierCV(base, cv=3, method="sigmoid")


def train_classifier(X_train, y_train, n_classes: int):
    """
    Trains both Random Forest and (optionally) XGBoost, then
    builds a soft-voting ensemble if both are available.
    Returns the best single classifier (used for saving) plus the ensemble.
    """
    print(f"\n{'─'*60}")
    print(f"    TRAINING CLASSIFIERS")
    print(f"{'─'*60}")

    t0 = time.time()
    print("  → Random Forest ...")
    clf_rf = _make_rf(n_classes)
    clf_rf.fit(X_train, y_train)
    print(f"     done in {time.time()-t0:.1f}s")

    clf_xgb = None
    if XGBOOST:
        t0 = time.time()
        print("  → XGBoost ...")
        clf_xgb = _make_xgb(n_classes)
        clf_xgb.fit(X_train, y_train)
        print(f"     done in {time.time()-t0:.1f}s")

    # Pick best single model (used for saving)
    best_clf = clf_xgb if XGBOOST else clf_rf

    return clf_rf, clf_xgb, best_clf



def train_novelty_detector(X_train):
    print(f"\n    TRAINING NOVELTY DETECTOR (IsolationForest, contamination={ANOMALY_CONTAM})")
    iso = IsolationForest(
        n_estimators=200,
        contamination=ANOMALY_CONTAM,
        max_samples="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X_train)
    print("     done ✓")
    return iso


def evaluate(clf, X_test, y_test, le: LabelEncoder, clf_name: str):
    print(f"\n{'─'*60}")
    print(f"   EVALUATION — {clf_name}")
    print(f"{'─'*60}")
    y_pred = clf.predict(X_test)
    labels = le.classes_

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1_mac  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"  Balanced Accuracy : {bal_acc:.4f}")
    print(f"  F1 (macro)        : {f1_mac:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(labels)))
    fig, ax = plt.subplots(figsize=(max(10, len(labels)), max(8, len(labels) - 1)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, xticks_rotation=45, cmap="Blues")
    plt.title(f"Confusion Matrix — {clf_name}", fontsize=13, pad=12)
    plt.tight_layout()
    fname = f"confusion_matrix_{clf_name.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix → {fname}")
    return bal_acc, f1_mac



def predict_sequences(
    sequences: list,
    tfidf: TfidfVectorizer,
    clf,
    iso: IsolationForest,
    le: LabelEncoder,
    top_n: int = 1,
) -> pd.DataFrame:
    """
    Full prediction pipeline: k-mer → TF-IDF → classifier + anomaly detector.

    Decision logic:
        conf >= CONFIDENCE_THR  AND  not anomaly  →  KNOWN
        conf >= CONFIDENCE_THR  AND  anomaly       →  KNOWN ⚠ (check)
        conf <  CONFIDENCE_THR                     →  UNKNOWN
    """
    kmer_sents = sequences_to_kmers(sequences)
    
    # Filter out empty k-mer sequences and track valid indices
    valid_indices = [i for i, kmer in enumerate(kmer_sents) if kmer.strip()]
    valid_sequences = [sequences[i] for i in valid_indices]
    valid_kmers = [kmer_sents[i] for i in valid_indices]
    
    if not valid_kmers:
        print(f"  ⚠ Warning: All {len(sequences)} sequences were too short or invalid (< {K_SIZE} valid bases)")
        return pd.DataFrame(columns=["sequence", "full_sequence", "prediction", "confidence", 
                                     "anomaly_score", "is_anomaly", "status"])
    
    if len(valid_kmers) < len(sequences):
        print(f"  ⚠ Warning: {len(sequences) - len(valid_kmers)} sequences filtered out (too short or invalid)")
    
    X          = tfidf.transform(valid_kmers)
    proba      = clf.predict_proba(X)                
    iso_scores = iso.decision_function(X)              
    iso_pred   = iso.predict(X)                        

    sorted_idx = np.argsort(proba, axis=1)[:, ::-1]  

    results = []
    for i, seq in enumerate(valid_sequences):
        best_class_idx = sorted_idx[i, 0]
        conf           = float(proba[i, best_class_idx])
        anomaly        = (iso_pred[i] == -1)
        phylum         = le.inverse_transform([best_class_idx])[0]

        if conf >= CONFIDENCE_THR and not anomaly:
            status = "KNOWN"
        elif conf >= CONFIDENCE_THR and anomaly:
            status = "KNOWN ⚠"          
           
        else:
            status = "UNKNOWN"
            

        row = {
            "sequence"      : seq[:35] + "…" if len(seq) > 35 else seq,
            "full_sequence"  : seq,
            "prediction"    : phylum,
            "confidence"    : round(conf, 4),
            "anomaly_score" : round(float(iso_scores[i]), 4),
            "is_anomaly"    : anomaly,
            "status"        : status,
        }

        if top_n > 1:
            for rank in range(min(top_n, proba.shape[1])):
                idx = sorted_idx[i, rank]
                row[f"top{rank+1}_phylum"]     = le.inverse_transform([idx])[0]
                row[f"top{rank+1}_confidence"] = round(float(proba[i, idx]), 4)

        results.append(row)

    return pd.DataFrame(results)



def biodiversity_report(df: pd.DataFrame):
    print(f"\n{'═'*60}")
    print("    BIODIVERSITY REPORT")
    print(f"{'═'*60}")
    counts = df["prediction"].value_counts()
    total  = len(df)
    for phylum, count in counts.items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {phylum:<25} {bar:<35}  {pct:5.1f}%  ({count})")
    print(f"{'═'*60}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
           startangle=140, pctdistance=0.82)
    ax.set_title("Biodiversity Distribution", fontsize=14)
    plt.tight_layout()
    plt.savefig("biodiversity_pie.png", dpi=150)
    plt.close()
    print("    Biodiversity pie chart saved → biodiversity_pie.png")
    return counts



def save_models(tfidf, clf, iso, le, directory: str = MODEL_DIR):
    os.makedirs(directory, exist_ok=True)
    joblib.dump(tfidf, f"{directory}/tfidf.pkl")
    joblib.dump(clf,   f"{directory}/classifier.pkl")
    joblib.dump(iso,   f"{directory}/novelty_detector.pkl")
    joblib.dump(le,    f"{directory}/label_encoder.pkl")
    print(f"\n    Models saved → {directory}/")


def load_models(directory: str = MODEL_DIR):
    tfidf = joblib.load(f"{directory}/tfidf.pkl")
    clf   = joblib.load(f"{directory}/classifier.pkl")
    iso   = joblib.load(f"{directory}/novelty_detector.pkl")
    le    = joblib.load(f"{directory}/label_encoder.pkl")
    print(f"    Models loaded from {directory}/")
    return tfidf, clf, iso, le


def run_training():
    total_start = time.time()

    df        = load_dataset(CSV_PATH)
    sequences = df["sequence"].tolist()
    labels    = df["label"].tolist()

    le = LabelEncoder()
    y  = le.fit_transform(labels)
    n_classes = len(le.classes_)
    print(f"\n  Classes ({n_classes}): {le.classes_.tolist()}")

    print(f"\n    Generating k-mers (k={K_SIZE}) for {len(sequences):,} sequences ...")
    t0 = time.time()
    kmer_sentences = sequences_to_kmers(sequences, K_SIZE)
    print(f"     done in {time.time()-t0:.1f}s")

    sample_kmers = kmer_sentences[0].split()
    print(f"  Sample k-mers (first seq, first 8): {sample_kmers[:8]}")

    print(f"\n    Fitting TF-IDF (max_features={MAX_FEATURES}) ...")
    t0    = time.time()
    tfidf = build_tfidf(K_SIZE, MAX_FEATURES)
    X     = tfidf.fit_transform(kmer_sentences)
    print(f"     done in {time.time()-t0:.1f}s  →  feature matrix: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\n  Split: train={X_train.shape[0]:,}  test={X_test.shape[0]:,}")

    if SMOTE_OK:
        try:
            sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            print(f"    SMOTE applied → training shape: {X_train.shape}")
        except Exception as e:
            print(f"    SMOTE skipped: {e}")
    else:
        print("  ℹ  imbalanced-learn not installed, skipping SMOTE.")

    clf_rf, clf_xgb, best_clf = train_classifier(X_train, y_train, n_classes)

    rf_bal_acc, rf_f1 = evaluate(clf_rf, X_test, y_test, le, "Random Forest")
    if clf_xgb:
        xgb_bal_acc, xgb_f1 = evaluate(clf_xgb, X_test, y_test, le, "XGBoost")
        print(f"\n    COMPARISON:")
        print(f"      Random Forest  →  Balanced Acc={rf_bal_acc:.4f}  F1={rf_f1:.4f}")
        print(f"      XGBoost        →  Balanced Acc={xgb_bal_acc:.4f}  F1={xgb_f1:.4f}")
        best_name = "XGBoost" if xgb_f1 >= rf_f1 else "Random Forest"
        print(f"       Best model  : {best_name}")
    else:
        print("\n  ℹ XGBoost not installed — using Random Forest only.")

    iso = train_novelty_detector(X_train)

    save_models(tfidf, best_clf, iso, le)

    demo_seqs = df["sequence"].iloc[:15].tolist()

    print(f"\n{'═'*70}")
    print("    SEQUENCE-LEVEL PREDICTIONS  (first 15 sequences)")
    print(f"{'═'*70}")
    results = predict_sequences(demo_seqs, tfidf, best_clf, iso, le)
    display_cols = ["sequence", "prediction", "confidence", "anomaly_score", "status"]
    print(results[display_cols].to_string(index=False))

    biodiversity_report(results)

    results.to_csv("final_output.csv", index=False)
    print(f"\n    Saved → final_output.csv")
    elapsed = time.time() - total_start
    print(f"\n  ⏱   Total training time: {elapsed:.1f}s")
    print(f"{'═'*70}\n")



def run_prediction(seq_input: str, top_n: int = 1):
   
    if os.path.exists(seq_input):
        with open(seq_input) as f:
            sequences = [ln.strip() for ln in f if ln.strip()]
        print(f"\n    Loaded {len(sequences)} sequences from {seq_input}")
    else:
        sequences = [seq_input]
        print(f"\n    Using provided sequence ({len(seq_input)} bp)")

    tfidf, clf, iso, le = load_models()

    print(f"\n    Running predictions ...")
    results = predict_sequences(sequences, tfidf, clf, iso, le, top_n=top_n)

    print(f"\n{'═'*70}")
    print("  SEQUENCE PREDICTIONS")
    print(f"{'═'*70}")
    display_cols = ["sequence", "prediction", "confidence", "anomaly_score", "status"]
    if top_n > 1:
        extra = [f"top{i+1}_phylum" for i in range(top_n)]
        display_cols += extra
    print(results[display_cols].to_string(index=False))

    biodiversity_report(results)

    results.to_csv("final_output.csv", index=False)
    print(f"\n    Saved → final_output.csv")


def main():
    parser = argparse.ArgumentParser(
        description="DNA Biodiversity Analyzer — k-mer + TF-IDF + RF/XGBoost + IsolationForest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Train models on cleaned_dataset.csv and save to saved_models/"
    )
    parser.add_argument(
        "--predict", type=str, default=None,
        help="Path to file of sequences (one per line) to predict, or a raw sequence string"
    )
    parser.add_argument(
        "--top", type=int, default=1,
        help="Show top-N phylum candidates (default: 1)"
    )
    args, unknown = parser.parse_known_args()  # Ignore Jupyter's -f argument

    if args.train:
        run_training()
    elif args.predict:
        run_prediction(args.predict, top_n=args.top)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
