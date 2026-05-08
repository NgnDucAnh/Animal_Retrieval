import os
import io
import json
import base64
import warnings
import tempfile
import threading

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import mysql.connector

from flask import Flask, request, jsonify, render_template, send_from_directory
from scipy.signal import butter, lfilter
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from werkzeug.utils import secure_filename

from pre_processing import clean_audio_signal
from extract_audio_features import get_features

from config import BASE_DIR, ANIMAL_SOUND_DIR, SR, DB_CONFIG, ANIMAL_EMOJI
from database import get_db_connection, ensure_database_exists, get_db_stats
import indexer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload



def get_waveform_b64(y, sr, title="Waveform"):
    """Generate waveform plot and return as base64 string."""
    fig, ax = plt.subplots(figsize=(8, 2.5), facecolor='#0a0f1e')
    ax.set_facecolor('#0a0f1e')
    times = np.linspace(0, len(y) / sr, len(y))
    ax.plot(times, y, color='#6366f1', linewidth=0.8, alpha=0.9)
    ax.fill_between(times, y, alpha=0.3, color='#818cf8')
    ax.set_title(title, color='#e2e8f0', fontsize=11, pad=8)
    ax.set_xlabel("Time (s)", color='#94a3b8', fontsize=9)
    ax.set_ylabel("Amplitude", color='#94a3b8', fontsize=9)
    ax.tick_params(colors='#64748b', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e293b')
    ax.grid(True, alpha=0.15, color='#334155')
    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0a0f1e')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def get_spectrogram_b64(y, sr):
    """Generate mel spectrogram and return as base64."""
    fig, ax = plt.subplots(figsize=(8, 3), facecolor='#0a0f1e')
    ax.set_facecolor('#0a0f1e')
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel',
                                    ax=ax, cmap='magma')
    fig.colorbar(img, ax=ax, format='%+2.0f dB', label='dB')
    ax.set_title("Mel Spectrogram", color='#e2e8f0', fontsize=11, pad=8)
    ax.set_xlabel("Time (s)", color='#94a3b8', fontsize=9)
    ax.set_ylabel("Frequency (Hz)", color='#94a3b8', fontsize=9)
    ax.tick_params(colors='#64748b', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e293b')
    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0a0f1e')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def get_comparison_chart_b64(query_scaled, top_results_scaled, top_labels):
    """Generate feature comparison chart."""
    feat_labels = [f'M{i+1}' for i in range(13)] + [f'S{i+1}' for i in range(13)] + \
                  ['Cen_m', 'Cen_s', 'Rol_m', 'Rol_s', 'Bnd_m', 'Bnd_s', 'ZCR_m', 'ZCR_s']
    
    fig, ax = plt.subplots(figsize=(14, 4.5), facecolor='#0a0f1e')
    ax.set_facecolor('#0a0f1e')
    
    x = np.arange(len(feat_labels))
    ax.plot(x, query_scaled, color='#6366f1', marker='o', linestyle='-',
            linewidth=2.5, markersize=5, label='Query', zorder=5)
    
    palette = ['#f43f5e', '#10b981', '#f59e0b', '#06b6d4', '#8b5cf6']
    markers = ['s', '^', 'D', 'v', 'P']
    
    for i, (vec, lbl) in enumerate(zip(top_results_scaled, top_labels)):
        emoji = ANIMAL_EMOJI.get(lbl, "")
        color = palette[i % len(palette)]
        marker = markers[i % len(markers)]
        ax.plot(x, vec, color=color, marker=marker,
                linestyle='--', linewidth=1.5, markersize=4, alpha=0.8,
                label=f'Top {i+1}: {lbl} {emoji}')
    
    ax.set_xticks(x)
    ax.set_xticklabels(feat_labels, rotation=90, fontsize=7, color='#64748b')
    ax.tick_params(axis='y', colors='#64748b', labelsize=8)
    ax.set_title("Feature Vector Comparison: Query vs Top Results", color='#e2e8f0', fontsize=12, pad=10)
    ax.legend(loc='upper right', fontsize=8, facecolor='#1e293b',
              edgecolor='#334155', labelcolor='#e2e8f0', framealpha=0.9)
    ax.grid(True, alpha=0.12, color='#334155')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e293b')
    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0a0f1e')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# FLASK ROUTES

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def api_stats():
    return jsonify(get_db_stats())

@app.route('/api/index/start', methods=['POST'])
def api_index_start():
    if indexer.indexing_status.get("running"):
        return jsonify({"error": "Indexing already in progress"}), 400
    
    data = request.get_json(silent=True) or {}
    base_dir = data.get("base_dir", ANIMAL_SOUND_DIR)
    
    if not os.path.exists(base_dir):
        return jsonify({"error": f"Directory not found: {base_dir}"}), 400
    
    thread = threading.Thread(target=indexer.run_indexing, args=(base_dir,))
    thread.daemon = True
    thread.start()
    return jsonify({"message": "Indexing started", "base_dir": base_dir})

@app.route('/api/index/status')
def api_index_status():
    return jsonify(indexer.indexing_status)

@app.route('/api/search', methods=['POST'])
def api_search():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    top_k = int(request.form.get('top_k', 5))
    alpha = float(request.form.get('alpha', 0.5))
    
    # Save temp file
    tmp_path = None
    try:
        suffix = os.path.splitext(secure_filename(file.filename))[1] or '.wav'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        # Load and process audio
        y_raw, sr_raw = librosa.load(tmp_path, sr=SR)
        y_clean = clean_audio_signal(y_raw, SR)
        query_vector = get_features(y_clean, SR)
        
        # Generate visualizations
        waveform_raw_b64 = get_waveform_b64(y_raw, sr_raw, "Original Waveform")
        waveform_clean_b64 = get_waveform_b64(y_clean, SR, "Processed Waveform")
        spectrogram_b64 = get_spectrogram_b64(y_clean, SR)
        
        # Query database
        db = get_db_connection()
        df = pd.read_sql("SELECT * FROM audio_features", db)
        db.close()
        
        if df.empty:
            return jsonify({"error": "Database is empty. Please index audio files first."}), 400
        
        feature_cols = df.columns[3:]
        db_matrix = df[feature_cols].values
        
        # Hybrid metric (Z-score + Euclidean + Cosine)
        scaler = StandardScaler()
        db_scaled = scaler.fit_transform(db_matrix)
        query_scaled = scaler.transform(query_vector.reshape(1, -1))[0]
        
        euclid_dists = np.array([distance.euclidean(query_scaled, row) for row in db_scaled])
        cosine_dists = np.array([distance.cosine(query_scaled, row) for row in db_scaled])
        
        euclid_scaler = MinMaxScaler()
        euclid_norm = euclid_scaler.fit_transform(euclid_dists.reshape(-1, 1)).flatten()
        
        hybrid_dists = (alpha * euclid_norm) + ((1 - alpha) * cosine_dists)
        
        df['_dist'] = hybrid_dists
        df['_euclid'] = euclid_norm
        df['_cosine'] = cosine_dists
        top_df = df.sort_values(by='_dist').head(top_k)
        
        results = []
        top_scaled_vecs = []
        top_labels = []
        for idx, row in top_df.iterrows():
            # Similarity = 1 - normalized_distance (capped at 0)
            similarity = max(0.0, 1.0 - row['_dist'])
            results.append({
                "rank": len(results) + 1,
                "label": row['label'],
                "emoji": ANIMAL_EMOJI.get(row['label'], "🔊"),
                "file_path": row['file_path'],
                "file_name": os.path.basename(row['file_path']),
                "hybrid_score": round(float(row['_dist']), 4),
                "similarity": round(float(similarity) * 100, 1),
                "euclid_score": round(float(row['_euclid']), 4),
                "cosine_score": round(float(row['_cosine']), 4),
            })
            top_scaled_vecs.append(db_scaled[idx])
            top_labels.append(row['label'])
        
        # Comparison chart
        comparison_b64 = get_comparison_chart_b64(query_scaled, top_scaled_vecs, top_labels)
        
        return jsonify({
            "results": results,
            "query_features": {
                "mfcc_mean": [round(float(v), 3) for v in query_vector[:13]],
                "centroid_mean": round(float(query_vector[26]), 2),
                "zcr_mean": round(float(query_vector[32]), 5),
            },
            "charts": {
                "waveform_raw": waveform_raw_b64,
                "waveform_clean": waveform_clean_b64,
                "spectrogram": spectrogram_b64,
                "comparison": comparison_b64,
            },
            "db_total": len(df),
        })
    
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
    
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.route('/api/play')
def play_audio():
    file_path = request.args.get('path')
    if not file_path or not os.path.exists(file_path):
        return "Not found", 404
    directory = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    return send_from_directory(directory, filename)

@app.route('/api/db/clear', methods=['POST'])
def api_db_clear():
    try:
        db = get_db_connection()
        cursor = db.cursor()
        cursor.execute("TRUNCATE TABLE audio_features")
        db.commit()
        cursor.close()
        db.close()
        return jsonify({"message": "Database cleared successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print("Animal Sound Retrieval System")
    print("=" * 40)
    ensure_database_exists()
    print(f"Animal Sound Directory: {ANIMAL_SOUND_DIR}")
    print("Starting server at http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)
