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
    try:
        # Load audio, giữ nguyên tần số lấy mẫu gốc
        y, sr = librosa.load(file_path, sr=None) 
        features = []

        # ==========================================
        # BƯỚC TỐI ƯU: TÍNH BIẾN ĐỔI FOURIER (STFT) 1 LẦN
        # ==========================================
        # Lấy giá trị tuyệt đối của STFT để có Magnitude Spectrogram (Biên độ phổ)
        S = np.abs(librosa.stft(y))

        # 1. MFCCs (Truyền thẳng y để librosa tự xử lý qua phổ Mel)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfccs, axis=1)) # 13 giá trị trung bình
        features.extend(np.std(mfccs, axis=1))  # 13 giá trị độ lệch chuẩn (THÊM MỚI)

        # 2. Spectral Centroid (Truyền phổ S vào)
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
        features.append(np.mean(centroid))
        features.append(np.std(centroid))       # THÊM MỚI

        # 3. Spectral Rolloff (Truyền phổ S vào)
        rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)
        features.append(np.mean(rolloff))
        features.append(np.std(rolloff))        # THÊM MỚI

        # 4. Spectral Bandwidth (Truyền phổ S vào)
        bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
        features.append(np.mean(bandwidth))
        features.append(np.std(bandwidth))      # THÊM MỚI

        # 5. Zero Crossing Rate - Đo độ "khàn/nhiễu" (Tính trên sóng âm y)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        features.append(np.mean(zcr))           # THÊM MỚI
        features.append(np.std(zcr))            # THÊM MỚI

        # Trả về numpy array kích thước (34,)
        return np.array(features)
    
    except Exception as e:
        print(f"Lỗi khi xử lý file {file_path}: {e}")
        return None

# --- VÍ DỤ CÁCH CHẠY THỬ ---
if __name__ == "__main__":
    # Test với file Dog_3.wav của bạn
    sample_file = r"D:\archive\Animal-Soundprepros\Dog_Processed\Dog_3.wav" 
    
    if os.path.exists(sample_file):
        vector = extract_audio_features(sample_file)
        if vector is not None:
            print(f"Trích xuất thành công! Kích thước vector: {vector.shape}") # Kết quả mong đợi: (34,)
            print(f"Giá trị vector đặc trưng:\n{vector}")
    else:
        print("Vui lòng cập nhật đường dẫn sample_file để test.")