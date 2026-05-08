import os
import librosa
import soundfile as sf

from config import SR
from database import get_db_connection
from pre_processing import clean_audio_signal
from extract_audio_features import get_features

# Biến toàn cục để theo dõi tiến độ indexing
indexing_status = {"running": False, "progress": 0, "total": 0, "message": "", "done": False, "error": None}

def run_indexing(base_dir):
    global indexing_status
    indexing_status = {"running": True, "progress": 0, "total": 0, "message": "Connecting to database...", "done": False, "error": None}
    
    try:
        db = get_db_connection()
        cursor = db.cursor()
        
        # Create table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS audio_features (
            id INT AUTO_INCREMENT PRIMARY KEY,
            file_path VARCHAR(512) UNIQUE,
            label VARCHAR(100),
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
        """)
        db.commit()
        
        insert_query = """
        INSERT IGNORE INTO audio_features 
        (file_path, label,
        mfcc_mean_1, mfcc_mean_2, mfcc_mean_3, mfcc_mean_4, mfcc_mean_5, mfcc_mean_6, mfcc_mean_7, mfcc_mean_8, mfcc_mean_9, mfcc_mean_10, mfcc_mean_11, mfcc_mean_12, mfcc_mean_13,
        mfcc_std_1, mfcc_std_2, mfcc_std_3, mfcc_std_4, mfcc_std_5, mfcc_std_6, mfcc_std_7, mfcc_std_8, mfcc_std_9, mfcc_std_10, mfcc_std_11, mfcc_std_12, mfcc_std_13,
        centroid_mean, centroid_std, rolloff_mean, rolloff_std, bandwidth_mean, bandwidth_std, zcr_mean, zcr_std)
        VALUES (%s, %s, """ + ", ".join(["%s"] * 34) + ")"

        # Lấy danh sách file đã có trong DB để bỏ qua
        cursor.execute("SELECT file_path FROM audio_features")
        existing_paths = {str(row[0]).replace('\\', '/') for row in cursor.fetchall()}

        raw_dirs = []
        for folder_name in os.listdir(base_dir):
            input_dir = os.path.join(base_dir, folder_name)
            if os.path.isdir(input_dir) and not folder_name.endswith("_Processed"):
                wav_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
                if wav_files:
                    raw_dirs.append((folder_name, input_dir, wav_files))
        
        total_files = sum(len(wavs) for _, _, wavs in raw_dirs)
        indexing_status["total"] = total_files
        processed = 0
        saved = 0

        for animal_name, input_dir, wav_files in raw_dirs:
            indexing_status["message"] = f"Đang xử lý {animal_name}..."
            
            output_folder_name = f"{animal_name}_Processed"
            output_dir = os.path.join(base_dir, output_folder_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            batch = []
            for file_name in wav_files:
                raw_file_path = os.path.join(input_dir, file_name)
                processed_file_path = os.path.join(output_dir, file_name)
                
                # So sánh đường dẫn đã tồn tại chưa
                if processed_file_path.replace('\\', '/') in existing_paths:
                    processed += 1
                    indexing_status["progress"] = processed
                    continue
                
                try:
                    if not os.path.exists(processed_file_path):
                        y, sr = librosa.load(raw_file_path, sr=SR, mono=True)
                        y_clean = clean_audio_signal(y, SR)
                        sf.write(processed_file_path, y_clean, SR)
                    else:
                        y_clean, sr = librosa.load(processed_file_path, sr=SR)
                    
                    features = get_features(y_clean, SR)
                    
                    if len(features) == 34:
                        row = (processed_file_path, animal_name) + tuple(float(v) for v in features)
                        batch.append(row)
                except Exception as e:
                    print(f"Lỗi khi xử lý {file_name}: {e}")
                
                processed += 1
                indexing_status["progress"] = processed

            if batch:
                cursor.executemany(insert_query, batch)
                db.commit()
                saved += cursor.rowcount

        cursor.close()
        db.close()
        indexing_status["message"] = f"Done! {saved} files indexed."
        indexing_status["done"] = True
        indexing_status["running"] = False

    except Exception as e:
        indexing_status["error"] = str(e)
        indexing_status["running"] = False
        indexing_status["message"] = f"Error: {e}"
