
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import load_models, predict_sequences, biodiversity_report
import pandas as pd

def validate_sequence(seq, min_length=50):
   
    if any(char.isdigit() for char in seq):
        return False, "Contains numbers"
    
    if len(seq) < min_length:
        return False, f"Too short (min: {min_length} bp)"
    
    return True, None

def main():
   
    test_file = "sample.csv"
    
    print(f"\n{'═'*70}")
    print(f"    Loading test sequences from: {test_file}")
    print(f"{'═'*70}")
    
    with open(test_file, 'r') as f:
        raw_sequences = [line.strip() for line in f if line.strip()]
    
   
    sequences = []
    invalid_count = 0
    for seq in raw_sequences:
        is_valid, error_msg = validate_sequence(seq)
        if is_valid:
            sequences.append(seq)
        else:
            invalid_count += 1
            print(f"    Invalid sequence (skipped): {seq[:50]}... - {error_msg}")
    
    if invalid_count > 0:
        print(f"\n    Skipped {invalid_count} invalid sequence(s)")
    
    if len(sequences) == 0:
        print(f"\n    No valid sequences to process!")
        return
    
    print(f"  ✓ Loaded {len(sequences)} valid sequences")
    print(f"  Average length: {sum(len(s) for s in sequences) / len(sequences):.0f} bp")
    

    print(f"\n   Loading trained models...")
    tfidf, clf, iso, le = load_models()
    
   
    print(f"\n   Running predictions...")
    results = predict_sequences(sequences, tfidf, clf, iso, le, top_n=3)
 
    results_sorted = results.sort_values('confidence', ascending=False)
    

    print(f"\n{'═'*70}")
    print("  TOP CONFIDENT PREDICTIONS")
    print(f"{'═'*70}")
    display_cols = ["sequence", "prediction", "confidence", "status"]
    print(results_sorted[display_cols].to_string(index=False))
    

    biodiversity_report(results, output_path="biodiversity_pie.png")
    

    # Save results - only the display columns
    output_file = "test_predictions.csv"
    results_sorted[display_cols].to_csv(output_file, index=False)
    print(f"\n  💾  Results saved to: {output_file}")
    print(f"{'═'*70}\n")

if __name__ == "__main__":
    main()
