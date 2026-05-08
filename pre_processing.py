import librosa
import numpy as np
import os
import soundfile as sf
from scipy.signal import butter, lfilter
from config import SR, DURATION, LOWCUT, HIGHCUT, ANIMAL_SOUND_DIR

# LỌC NHIỄU (BANDPASS FILTER)
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def clean_audio_signal(y, sr):
    # Tiền xử lý 1 đoạn âm thanh (Dùng chung cho cả tạo Dataset và Web Query)
    y_filtered = butter_bandpass_filter(y, LOWCUT, HIGHCUT, sr)
    y_trimmed, _ = librosa.effects.trim(y_filtered, top_db=25)
    
    target_length = sr * DURATION
    if len(y_trimmed) > target_length:
        squared = y_trimmed**2
        cum_sum = np.insert(np.cumsum(squared), 0, 0)
        energy = cum_sum[target_length:] - cum_sum[:-target_length]
        max_idx = np.argmax(energy)
        y_fixed = y_trimmed[max_idx : max_idx + target_length]
    else:
        y_fixed = librosa.util.pad_center(y_trimmed, size=target_length) 
        
    return librosa.util.normalize(y_fixed)

def process_all_datasets(base_dir):
    print(f"BẮT ĐẦU QUÉT TOÀN BỘ DATASET TẠI: {base_dir}\n" + "="*50)

    # Duyệt qua tất cả các mục (thư mục/file) nằm trong thư mục gốc
    for folder_name in os.listdir(base_dir):
        input_dir = os.path.join(base_dir, folder_name)

        # Chỉ xử lý nếu nó là một THƯ MỤC và KHÔNG có chữ "_Processed" ở đuôi
        if os.path.isdir(input_dir) and not folder_name.endswith("_Processed"):
            
            # Tự động tạo tên thư mục đầu ra (VD: "Bear" -> "Bear_Processed")
            output_folder_name = f"{folder_name}_Processed"
            output_dir = os.path.join(base_dir, output_folder_name)

            # Tự động tạo thư mục đầu ra nếu chưa tồn tại
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            elif len(os.listdir(output_dir)) > 0:
                print(f"Bỏ qua loài {folder_name} vì đã được xử lý từ trước.")
                continue

            print(f"\nĐang xử lý loài vật: {folder_name} ...")
            
            file_count = 0
            # Duyệt qua tất cả các file .wav trong thư mục con vật này
            for filename in os.listdir(input_dir):
                if filename.endswith(".wav"):
                    file_path = os.path.join(input_dir, filename)
                    
                    try:
                        # Tiền xử lý bằng hàm chuẩn
                        y, sr = librosa.load(file_path, sr=SR, mono=True)
                        y_final = clean_audio_signal(y, SR)
                        
                        # LƯU FILE ĐÃ XỬ LÝ
                        output_path = os.path.join(output_dir, filename)
                        sf.write(output_path, y_final, SR)
                        
                        file_count += 1

                    except Exception as e:
                        print(f"Lỗi khi xử lý file {filename}: {e}")

            print(f"   => Hoàn thành: Đã lưu {file_count} file sạch vào mục {output_folder_name}")

    print("ĐÃ HOÀN TẤT TIỀN XỬ LÝ TOÀN BỘ DỮ LIỆU")

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    process_all_datasets(ANIMAL_SOUND_DIR)

