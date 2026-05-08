import librosa
import numpy as np
import os

def extract_audio_features(file_path):
    """
    Hàm trích xuất 34 đặc trưng âm thanh bao gồm:
    - MFCCs (13 Mean + 13 Std) = 26
    - Spectral Centroid (1 Mean + 1 Std) = 2
    - Spectral Rolloff (1 Mean + 1 Std) = 2
    - Spectral Bandwidth (1 Mean + 1 Std) = 2
    - Zero Crossing Rate (1 Mean + 1 Std) = 2
    Tổng cộng: 34 chiều.
    """
def get_features(y, sr):
    features = []
    # TÍNH BIẾN ĐỔI FOURIER (STFT) 1 LẦN
    S = np.abs(librosa.stft(y))

    # 1. MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))

    # 2. Spectral Centroid
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    features.append(np.mean(centroid))
    features.append(np.std(centroid))

    # 3. Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)
    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))

    # 4. Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    features.append(np.mean(bandwidth))
    features.append(np.std(bandwidth))

    # 5. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    return np.array(features)

def extract_audio_features(file_path):
    try:
        # Load audio, giữ nguyên tần số lấy mẫu gốc
        y, sr = librosa.load(file_path, sr=None) 
        return get_features(y, sr)
    
    except Exception as e:
        print(f"Lỗi khi xử lý file {file_path}: {e}")
        return None
