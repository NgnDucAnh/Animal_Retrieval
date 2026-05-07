import os
import mysql.connector
from extract_audio_features import extract_audio_features
from dotenv import load_dotenv

load_dotenv()

def process_all_and_save_to_mysql(base_dir):
    # ==========================================
    # 1. THIẾT LẬP KẾT NỐI TỚI MYSQL
    # ==========================================
    try:
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password=os.getenv("DB_PASSWORD", ""),
            database="animal_sounds"
        )
        cursor = db.cursor()
    except mysql.connector.Error as err:
        print(f"Lỗi kết nối MySQL: {err}")
        return

    # Tự động tạo bảng với 34 cột đặc trưng (Thêm các cột Std và ZCR)
    create_table_query = """
    CREATE TABLE IF NOT EXISTS audio_features (
        id INT AUTO_INCREMENT PRIMARY KEY,
        file_path VARCHAR(255) UNIQUE,
        label VARCHAR(50),
        mfcc_mean_1 FLOAT, mfcc_mean_2 FLOAT, mfcc_mean_3 FLOAT, mfcc_mean_4 FLOAT,
        mfcc_mean_5 FLOAT, mfcc_mean_6 FLOAT, mfcc_mean_7 FLOAT, mfcc_mean_8 FLOAT,
        mfcc_mean_9 FLOAT, mfcc_mean_10 FLOAT, mfcc_mean_11 FLOAT, mfcc_mean_12 FLOAT, mfcc_mean_13 FLOAT,
        mfcc_std_1 FLOAT, mfcc_std_2 FLOAT, mfcc_std_3 FLOAT, mfcc_std_4 FLOAT,
        mfcc_std_5 FLOAT, mfcc_std_6 FLOAT, mfcc_std_7 FLOAT, mfcc_std_8 FLOAT,
        mfcc_std_9 FLOAT, mfcc_std_10 FLOAT, mfcc_std_11 FLOAT, mfcc_std_12 FLOAT, mfcc_std_13 FLOAT,
        centroid_mean FLOAT, centroid_std FLOAT,
        rolloff_mean FLOAT, rolloff_std FLOAT,
        bandwidth_mean FLOAT, bandwidth_std FLOAT,
        zcr_mean FLOAT, zcr_std FLOAT
    )
    """
    cursor.execute(create_table_query)

    # Chuỗi lệnh Insert cho đường dẫn, nhãn và 34 cột đặc trưng (tổng cộng 36 giá trị)
    insert_query = """
    INSERT IGNORE INTO audio_features 
    (file_path, label, 
    mfcc_mean_1, mfcc_mean_2, mfcc_mean_3, mfcc_mean_4, mfcc_mean_5, mfcc_mean_6, mfcc_mean_7, mfcc_mean_8, mfcc_mean_9, mfcc_mean_10, mfcc_mean_11, mfcc_mean_12, mfcc_mean_13,
    mfcc_std_1, mfcc_std_2, mfcc_std_3, mfcc_std_4, mfcc_std_5, mfcc_std_6, mfcc_std_7, mfcc_std_8, mfcc_std_9, mfcc_std_10, mfcc_std_11, mfcc_std_12, mfcc_std_13,
    centroid_mean, centroid_std, rolloff_mean, rolloff_std, bandwidth_mean, bandwidth_std, zcr_mean, zcr_std)
    VALUES (%s, %s, """ + ", ".join(["%s"] * 34) + ")"

    print("BẮT ĐẦU QUÉT TOÀN BỘ DỮ LIỆU...\n" + "-"*40)
    total_files_saved = 0

    # ==========================================
    # 2. TRÍCH XUẤT VÀ LƯU THEO TỪNG THƯ MỤC
    # ==========================================
    for label in os.listdir(base_dir):
        # Lấy TẤT CẢ các thư mục có đuôi _Processed
        if label.endswith("_Processed"):
            class_dir = os.path.join(base_dir, label)
            animal_name = label.replace("_Processed", "")
            
            # flush=True giúp in ra màn hình ngay lập tức, không bị đợi buffer
            print(f"Đang xử lý thư mục: {label}...", end=" ", flush=True) 
            
            dataset_features = [] # Tạo danh sách chứa riêng cho từng loài
            
            for file_name in os.listdir(class_dir):
                if file_name.endswith(".wav"):
                    file_path = os.path.join(class_dir, file_name)
                    features = extract_audio_features(file_path)
                    
                    if features is not None:
                        # Kiểm tra an toàn: Đảm bảo trích xuất đủ 34 số mới đưa vào Database
                        if len(features) == 34:
                            features_list = [float(val) for val in features]
                            row_data = (file_path, animal_name) + tuple(features_list)
                            dataset_features.append(row_data)
                        else:
                            print(f"\n[Cảnh báo] File {file_name} trả về {len(features)} đặc trưng. Bỏ qua.")

            # Nếu có dữ liệu thì lưu luôn vào MySQL
            if len(dataset_features) > 0:
                try:
                    cursor.executemany(insert_query, dataset_features)
                    db.commit() # Chốt lưu vào ổ cứng
                    print(f"-> Đã lưu {cursor.rowcount} file.")
                    total_files_saved += cursor.rowcount
                except mysql.connector.Error as err:
                    print(f"\nLỗi khi lưu thư mục {label}: {err}")
            else:
                print("-> Thư mục trống hoặc lỗi định dạng.")

    # ==========================================
    # 3. ĐÓNG KẾT NỐI
    # ==========================================
    cursor.close()
    db.close()
    print("-" * 40)
    print(f"TỔNG KẾT: Đã hoàn thành! Tổng cộng {total_files_saved} file đã được đẩy lên MySQL.")

if __name__ == "__main__":
    # Chạy hàm này với thư mục gốc của bạn
    process_all_and_save_to_mysql(r"D:\archive\Animal-Soundprepros")