import librosa
import numpy as np
import os
import soundfile as sf
from scipy.signal import butter, lfilter

# --- CẤU HÌNH THÔNG SỐ ---
SR = 22050          # Sample Rate chuẩn cho nhận dạng âm thanh
DURATION = 5        # Độ dài cố định 5 giây cho mọi file
LOWCUT = 50         # Tần số cắt thấp (loại bỏ tiếng ù nền)
HIGHCUT = 8000      # Tần số cắt cao (loại bỏ nhiễu điện tử cao tần)

# Đặt thư mục cha chứa toàn bộ dữ liệu của bạn
# Hãy đảm bảo đường dẫn này trỏ đúng đến thư mục Animal-Soundprepros trên máy bạn
BASE_DIR = r"D:\archive\Animal-Soundprepros" 

# --- HÀM BỔ TRỢ BƯỚC 2: LỌC NHIỄU (BANDPASS FILTER) ---
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def process_all_datasets(base_dir):
    print(f"BẮT ĐẦU QUÉT TOÀN BỘ DATASET TẠI: {base_dir}\n" + "="*50)

    # Duyệt qua tất cả các mục (thư mục/file) nằm trong thư mục gốc
    for folder_name in os.listdir(base_dir):
        input_dir = os.path.join(base_dir, folder_name)

        # LỌC THÔNG MINH:
        # Chỉ xử lý nếu nó là một THƯ MỤC và KHÔNG có chữ "_Processed" ở đuôi
        if os.path.isdir(input_dir) and not folder_name.endswith("_Processed"):
            
            # Tự động tạo tên thư mục đầu ra (VD: "Bear" -> "Bear_Processed")
            output_folder_name = f"{folder_name}_Processed"
            output_dir = os.path.join(base_dir, output_folder_name)

            # Tự động tạo thư mục đầu ra nếu chưa tồn tại
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            print(f"\n📁 Đang xử lý loài vật: {folder_name} ...")
            
            file_count = 0
            # Duyệt qua tất cả các file .wav trong thư mục con vật này
            for filename in os.listdir(input_dir):
                if filename.endswith(".wav"):
                    file_path = os.path.join(input_dir, filename)
                    
                    try:
                        # BƯỚC 1: Format Standardization
                        y, sr = librosa.load(file_path, sr=SR, mono=True)
                        
                        # BƯỚC 2: Noise Reduction
                        y_filtered = butter_bandpass_filter(y, LOWCUT, HIGHCUT, SR)
                        
                        # BƯỚC 3: Silence Removal
                        y_trimmed, _ = librosa.effects.trim(y_filtered, top_db=25)
                        
                        # BỔ SUNG: Khớp độ dài cố định 5s
                        target_length = SR * DURATION
                        if len(y_trimmed) > target_length:
                            y_fixed = y_trimmed[:target_length] 
                        else:
                            y_fixed = librosa.util.pad_center(y_trimmed, size=target_length) 
                        
                        # BƯỚC 4: Normalization
                        y_final = librosa.util.normalize(y_fixed)
                        
                        # LƯU FILE ĐÃ XỬ LÝ
                        output_path = os.path.join(output_dir, filename)
                        sf.write(output_path, y_final, SR)
                        
                        file_count += 1
                        # Bạn có thể bật dòng dưới đây lên nếu muốn in chi tiết từng file
                        # print(f"   -> Đã xử lý: {filename}")
                        
                    except Exception as e:
                        print(f"   ❌ Lỗi khi xử lý file {filename}: {e}")

            print(f"   => Hoàn thành: Đã lưu {file_count} file sạch vào mục {output_folder_name}")

    print("\n" + "="*50)
    print("TỔNG KẾT: ĐÃ HOÀN TẤT TIỀN XỬ LÝ TOÀN BỘ DỮ LIỆU!")

if __name__ == "__main__":
    process_all_datasets(BASE_DIR)







# import librosa
# import numpy as np
# import os
# import soundfile as sf
# from scipy.signal import butter, lfilter

# # --- CẤU HÌNH THÔNG SỐ ---
# SR = 22050          # Sample Rate chuẩn cho nhận dạng âm thanh
# DURATION = 5        # Độ dài cố định 5 giây cho mọi file
# LOWCUT = 50         # Tần số cắt thấp (loại bỏ tiếng ù nền)
# HIGHCUT = 8000      # Tần số cắt cao (loại bỏ nhiễu điện tử cao tần)
# INPUT_DIR = r"Animal-Soundprepros\Sheep" # Thư mục chứa file gốc của bạn
# OUTPUT_DIR = r"Animal-Soundprepros\Sheep_Processed" # Thư mục lưu file đã sạch

# # --- HÀM BỔ TRỢ BƯỚC 2: LỌC NHIỄU (BANDPASS FILTER) ---
# def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = butter(order, [low, high], btype='band')
#     y = lfilter(b, a, data)
#     return y

# def process_all_files():
#     if not os.path.exists(OUTPUT_DIR):
#         os.makedirs(OUTPUT_DIR)

#     print(f"Bắt đầu tiền xử lý các file trong thư mục: {INPUT_DIR}...")

#     for filename in os.listdir(INPUT_DIR):
#         if filename.endswith(".wav"):
#             file_path = os.path.join(INPUT_DIR, filename)
            
#             # BƯỚC 1: Format Standardization (Chuẩn hóa định dạng)
#             # Load file, tự động chuyển về Mono và Resample về 22050Hz
#             y, sr = librosa.load(file_path, sr=SR, mono=True)
            
#             # BƯỚC 2: Noise Reduction (Loại bỏ nhiễu)
#             # Sử dụng bộ lọc Bandpass để giữ lại dải tần đặc trưng của động vật
#             y_filtered = butter_bandpass_filter(y, LOWCUT, HIGHCUT, SR)
            
#             # BƯỚC 3: Silence Removal (Cắt bỏ khoảng lặng)
#             # Loại bỏ đoạn im lặng ở đầu và cuối có cường độ dưới 25dB
#             y_trimmed, _ = librosa.effects.trim(y_filtered, top_db=25)
            
#             # BỔ SUNG: Khớp độ dài cố định (để phục vụ việc so sánh sau này)
#             target_length = SR * DURATION
#             if len(y_trimmed) > target_length:
#                 y_fixed = y_trimmed[:target_length] # Cắt nếu quá dài
#             else:
#                 y_fixed = librosa.util.pad_center(y_trimmed, size=target_length) # Đệm 0 nếu quá ngắn
            
#             # BƯỚC 4: Normalization (Chuẩn hóa biên độ/âm lượng)
#             # Đưa biên độ về khoảng [-1, 1] để đồng nhất năng lượng
#             y_final = librosa.util.normalize(y_fixed)
            
#             # LƯU FILE ĐÃ XỬ LÝ
#             output_path = os.path.join(OUTPUT_DIR, filename)
#             sf.write(output_path, y_final, SR)
#             print(f" -> Hoàn thành: {filename}")

#     print("\n--- TẤT CẢ FILE ĐÃ ĐƯỢC LÀM SẠCH VÀ LƯU TẠI:", OUTPUT_DIR, "---")

# if __name__ == "__main__":
#     process_all_files()

# import librosa
# import numpy as np
# import os
# import soundfile as sf
# from scipy.signal import butter, lfilter

# # --- CẤU HÌNH THÔNG SỐ ---
# SR = 22050          
# DURATION = 5        
# LOWCUT = 50         
# HIGHCUT = 8000      
# INPUT_DIR = r"Animal-Soundprepros\Aslan" 
# OUTPUT_DIR = r"Animal-Soundprepros\Aslan_Processed" 

# def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = butter(order, [low, high], btype='band')
#     y = lfilter(b, a, data)
#     return y

# def process_all_files():
#     if not os.path.exists(OUTPUT_DIR):
#         os.makedirs(OUTPUT_DIR)

#     print(f"Bắt đầu tiền xử lý các file trong thư mục: {INPUT_DIR}...")

#     for filename in os.listdir(INPUT_DIR):
#         if filename.endswith(".wav"):
#             file_path = os.path.join(INPUT_DIR, filename)
            
#             # BƯỚC 1: Số hóa và Biểu diễn dữ liệu (Digitization)
#             # Dòng này chuyển file .wav thành mảng số thực float32 [cite: 154, 163]
#             y, sr = librosa.load(file_path, sr=SR, mono=True)
            
#             # --- HIỂN THỊ CHUỖI SỐ ĐỂ BÁO CÁO ---
#             print(f"\n[Dữ liệu số hóa của file: {filename}]")
#             print(f"Mảng amplitudes (y): {y}") 
#             print(f"Kiểu dữ liệu: {y.dtype}") # Thường là float32 
#             print(f"Số lượng mẫu (N): {len(y)}") # N = SR * Thời gian thực 
#             print("-" * 50)
            
#             # BƯỚC 2: Noise Reduction
#             y_filtered = butter_bandpass_filter(y, LOWCUT, HIGHCUT, SR)
            
#             # BƯỚC 3: Silence Removal
#             y_trimmed, _ = librosa.effects.trim(y_filtered, top_db=25)
            
#             # BỔ SUNG: Khớp độ dài 5 giây
#             target_length = SR * DURATION
#             if len(y_trimmed) > target_length:
#                 y_fixed = y_trimmed[:target_length]
#             else:
#                 y_fixed = librosa.util.pad_center(y_trimmed, size=target_length)
            
#             # BƯỚC 4: Normalization
#             y_final = librosa.util.normalize(y_fixed)
            
#             # LƯU FILE
#             output_path = os.path.join(OUTPUT_DIR, filename)
#             sf.write(output_path, y_final, SR)
#             print(f" -> Đã tiền xử lý xong: {filename}")

# if __name__ == "__main__":
#     process_all_files()