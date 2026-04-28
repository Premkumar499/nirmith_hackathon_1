from django.shortcuts import render, redirect
from django.conf import settings
import os
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import load_models, predict_sequences, biodiversity_report

def upload_csv(request):
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']
        
        media_dir = os.path.join(settings.BASE_DIR, 'media')
        os.makedirs(media_dir, exist_ok=True)
        
        csv_path = os.path.join(media_dir, 'uploaded_data.csv')
        with open(csv_path, 'wb+') as destination:
            for chunk in csv_file.chunks():
                destination.write(chunk)
        
        try:
            # Try reading with header first
            df = pd.read_csv(csv_path)
            
            # Check if the CSV has a proper header or if the first row is actually data
            # If the first column name looks like a DNA sequence, treat it as no header
            first_col_name = str(df.columns[0])
            is_dna_sequence = len(first_col_name) > 50 and all(c in 'ACGTN' for c in first_col_name.upper().replace(' ', ''))
            
            if is_dna_sequence or len(df) == 0:
                # Re-read without header
                df = pd.read_csv(csv_path, header=None)
                seq_col = 0
            else:
                # Find the sequence column
                seq_col = None
                for col in df.columns:
                    if col.lower() in ['sequence', 'seq', 'dna', 'nucleotide']:
                        seq_col = col
                        break
                
                if seq_col is None:
                    seq_col = df.columns[0]
            
            sequences = df[seq_col].astype(str).tolist()
            
            # Remove any NaN or empty values
            sequences = [s for s in sequences if pd.notna(s) and str(s).strip()]
            
            if len(sequences) == 0:
                return render(request, 'upload.html', {
                    'error': 'No sequences found in the CSV file. Please ensure your CSV contains DNA sequences.'
                })
            
            # Validate sequences before processing
            valid_count = 0
            invalid_reasons = []
            for seq in sequences:
                clean_seq = ''.join(c for c in str(seq).upper() if c in 'ACGTN')
                if len(clean_seq) >= 4:  # K_SIZE = 4
                    valid_count += 1
                else:
                    if len(clean_seq) == 0:
                        invalid_reasons.append(f"Empty or no valid bases: {str(seq)[:50]}")
                    else:
                        invalid_reasons.append(f"Too short ({len(clean_seq)} bases): {str(seq)[:50]}")
            
            if valid_count == 0:
                error_details = "\n".join(invalid_reasons[:5])  # Show first 5 errors
                return render(request, 'upload.html', {
                    'error': f'No valid DNA sequences found. Sequences must contain at least 4 valid bases (A, C, G, T, N).\n\nFirst few issues:\n{error_details}'
                })
            
            if valid_count < len(sequences):
                print(f"Warning: {len(sequences) - valid_count} out of {len(sequences)} sequences are too short or invalid")
                for reason in invalid_reasons[:3]:
                    print(f"  - {reason}")
            
            model_path = os.path.join(settings.BASE_DIR, 'model', 'saved_models')
            
            if not os.path.exists(model_path):
                return render(request, 'upload.html', {
                    'error': f'Model files not found at {model_path}. Please train the model first using train.py'
                })
            
            try:
                tfidf, clf, iso, le = load_models(model_path)
            except Exception as model_error:
                return render(request, 'upload.html', {
                    'error': f'Error loading models: {str(model_error)}'
                })
            
            results = predict_sequences(sequences, tfidf, clf, iso, le, top_n=3)
            
            if len(results) == 0:
                return render(request, 'upload.html', {
                    'error': 'No valid DNA sequences found in the CSV file. Please ensure sequences contain only A, C, G, T, N characters and are at least 50 base pairs long.'
                })
            
            output_path = os.path.join(media_dir, 'test_predictions.csv')
            results.to_csv(output_path, index=False)
            
            chart_dest = os.path.join(media_dir, 'biodiversity_pie.png')
            biodiversity_report(results, output_path=chart_dest)
            
            return redirect('show_results')
            
        except Exception as e:
            return render(request, 'upload.html', {'error': str(e)})
    
    return render(request, 'upload.html')


def show_results(request):
    media_dir = os.path.join(settings.BASE_DIR, 'media')
    csv_path = os.path.join(media_dir, 'test_predictions.csv')
    chart_path = os.path.join(media_dir, 'biodiversity_pie.png')
    
    if not os.path.exists(csv_path):
        return redirect('upload_csv')
    
    df = pd.read_csv(csv_path)
    results = df.to_dict('records')
    
    has_chart = os.path.exists(chart_path)
    
    context = {
        'results': results,
        'has_chart': has_chart,
        'total_sequences': len(results),
    }
    
    return render(request, 'results.html', context)
