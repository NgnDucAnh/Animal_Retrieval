import librosa
import numpy as np
import os
import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from scipy.signal import butter, lfilter
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Tắt cảnh báo UserWarning của Pandas để Terminal sạch sẽ
warnings.filterwarnings('ignore', category=UserWarning)

# --- THÔNG SỐ CẤU HÌNH ---
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "NDa27072004",  # Mật khẩu của bạn
    "database": "animal_sounds"
}

SR = 22050
DURATION = 5
LOWCUT = 50
HIGHCUT = 8000

# ==========================================
# 1. NHÓM HÀM TIỀN XỬ LÝ (PRE-PROCESSING)
# ==========================================
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def clean_audio_signal(y, sr):
    # Lọc nhiễu Bandpass (50Hz - 8000Hz)
    y_filtered = butter_bandpass_filter(y, LOWCUT, HIGHCUT, sr)
    # Cắt khoảng lặng
    y_trimmed, _ = librosa.effects.trim(y_filtered, top_db=25)
    # Khớp độ dài 5 giây
    target_length = sr * DURATION
    if len(y_trimmed) > target_length:
        y_fixed = y_trimmed[:target_length]
    else:
        y_fixed = librosa.util.pad_center(y_trimmed, size=target_length)
    # Chuẩn hóa âm lượng [-1, 1]
    return librosa.util.normalize(y_fixed)

# ==========================================
# 2. NHÓM HÀM TRÍCH XUẤT ĐẶC TRƯNG (34 CHIỀU)
# ==========================================
def get_features(y, sr):
    S = np.abs(librosa.stft(y))
    features = []
    
    # 1. MFCCs (13 Mean + 13 Std)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))
    
    # 2. Spectral Centroid
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    features.extend([np.mean(centroid), np.std(centroid)])
    
    # 3. Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)
    features.extend([np.mean(rolloff), np.std(rolloff)])
    
    # 4. Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    features.extend([np.mean(bandwidth), np.std(bandwidth)])
    
    # 5. Zero Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    features.extend([np.mean(zcr), np.std(zcr)])
    
    return np.array(features)

# ==========================================
# 3. HÀM TRUY VẤN TỔNG HỢP (INTEGRATED)
# ==========================================
def integrated_retrieval(raw_audio_path):
    print(f"--- BẮT ĐẦU QUY TRÌNH TRUY XUẤT CHO: {os.path.basename(raw_audio_path)} ---")
    
    try:
        # BƯỚC 1: Load và tiền xử lý trực tiếp trên RAM
        print("1. Đang tiền xử lý (Lọc nhiễu, cắt khoảng lặng)...")
        y_raw, sr_raw = librosa.load(raw_audio_path, sr=SR)
        y_clean = clean_audio_signal(y_raw, SR)
        
        # BƯỚC 2: Trích xuất đặc trưng
        print("2. Đang trích xuất 34 đặc trưng...")
        query_vector = get_features(y_clean, SR)
        
        # BƯỚC 3: Kết nối MySQL và lấy dữ liệu
        print("3. Đang truy vấn Database...")
        db = mysql.connector.connect(**DB_CONFIG)
        df = pd.read_sql("SELECT * FROM audio_features", db)
        db.close()
        
        if df.empty:
            print("Lỗi: Database đang trống!")
            return

        # BƯỚC 4: TÍNH TOÁN KHOẢNG CÁCH LAI (HYBRID METRIC)
        print("4. Đang tính toán Khoảng cách Lai (50% Euclid + 50% Cosine)...")
        feature_cols = df.columns[3:]
        db_matrix = df[feature_cols].values
        
        # 4.1. Chuẩn hóa (Z-score) để các đặc trưng có cùng trọng lượng
        scaler = StandardScaler()
        db_scaled = scaler.fit_transform(db_matrix)
        query_scaled = scaler.transform(query_vector.reshape(1, -1))[0]
        
        # 4.2. Tính 2 mảng khoảng cách riêng biệt
        euclid_dists = [distance.euclidean(query_scaled, row) for row in db_scaled]
        cosine_dists = [distance.cosine(query_scaled, row) for row in db_scaled]
        
        # 4.3. Chuẩn hóa mảng Euclidean về [0, 1] để công bằng với Cosine
        euclid_scaler = MinMaxScaler()
        euclid_normalized = euclid_scaler.fit_transform(np.array(euclid_dists).reshape(-1, 1)).flatten()
        
        # 4.4. Tính điểm Hybrid (Trọng số Alpha = 0.5)
        ALPHA = 0.5 
        hybrid_dists = (ALPHA * euclid_normalized) + ((1 - ALPHA) * np.array(cosine_dists))
        
        # BƯỚC 5: Sắp xếp và Hiển thị kết quả
        df['Distance'] = hybrid_dists
        top_5 = df.sort_values(by='Distance', ascending=True).head(5)
        
        print("\n" + "="*55)
        print(" TOP 5 KẾT QUẢ TƯƠNG ĐỒNG NHẤT (HYBRID METRIC)")
        print("="*55)
        for i, (idx, row) in enumerate(top_5.iterrows(), 1):
            # Hybrid Score càng nhỏ (gần 0) thì càng giống nhau
            print(f"{i}. [{row['label']}] - Điểm Lai (Hybrid Score): {row['Distance']:.4f}")
            print(f"   Đường dẫn: {row['file_path']}")

        # BƯỚC 6: Vẽ biểu đồ so sánh Top 5
        top5_indices = top_5.index.tolist()
        top5_vecs = [db_scaled[idx] for idx in top5_indices]
        top5_labels = top_5['label'].tolist()
        
        visualize_comparison(query_scaled, top5_vecs, top5_labels)

    except Exception as e:
        print(f"Lỗi hệ thống: {e}")

# ==========================================
# 4. HÀM VẼ ĐỒ THỊ SO SÁNH (TOP 5)
# ==========================================
def visualize_comparison(q_vec, top5_vecs, top5_labels):
    labels = [f'M_Mean_{i+1}' for i in range(13)] + [f'M_Std_{i+1}' for i in range(13)] + \
             ['Cent_m', 'Cent_s', 'Roll_m', 'Roll_s', 'Band_m', 'Band_s', 'ZCR_m', 'ZCR_s']
    
    plt.figure(figsize=(16, 6))
    
    # 1. Vẽ đường Query (Xanh, nét liền, đậm)
    plt.plot(labels, q_vec, color='blue', marker='o', linestyle='-', 
             linewidth=3, markersize=8, label='Query (Đã tiền xử lý)')
    
    # 2. Vẽ 5 đường kết quả
    colors = ['red', 'green', 'purple', 'orange', 'cyan']
    markers = ['x', 's', '^', 'd', 'v']
    
    for i in range(5):
        plt.plot(labels, top5_vecs[i], color=colors[i], marker=markers[i], 
                 linestyle='--', linewidth=1.5, alpha=0.7, 
                 label=f'Top {i+1} ({top5_labels[i]})')
    
    plt.xticks(rotation=90)
    plt.title("So sánh Phổ Đặc trưng: Query vs Top 5 (Hybrid Euclidean-Cosine)", fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.12, 1.0))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test trực tiếp với file âm thanh thô của bạn (Ví dụ: tiếng mèo)
    raw_file = r"D:\archive\Animal-Soundprepros\Aslan\Lion_1.wav"
    if os.path.exists(raw_file):
        integrated_retrieval(raw_file)
    else:
        print(f"Không tìm thấy file: {raw_file}")