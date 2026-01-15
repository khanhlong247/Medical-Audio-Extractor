import os
import io
import pandas as pd
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download
from dotenv import load_dotenv

load_dotenv()
my_token = os.getenv("HF_TOKEN")
REPO_ID = "Hani89/Synthetic-Medical-Speech-Dataset"
LOCAL_DIR = "data"
NUM_FILES_TO_EXTRACT = 10  # Số lượng file muốn trích xuất

os.makedirs(LOCAL_DIR, exist_ok=True)

try:
    print(f"Đang kiểm tra cấu trúc repo: {REPO_ID}...")
    api = HfApi(token=my_token)
    all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    
    # Tìm file parquet (chứa dữ liệu)
    parquet_files = [f for f in all_files if f.endswith(".parquet")]
    
    if not parquet_files:
        print("Không tìm thấy file .parquet hay .wav nào. Repo này có cấu trúc lạ.")
        # In ra 5 file đầu tiên để debug
        print("File mẫu trong repo:", all_files[:5])
        exit()
        
    target_parquet = parquet_files[0] # Lấy file đầu tiên
    print(f"Tìm thấy gói dữ liệu: {target_parquet}")
    
    # 2. Tải file Parquet về
    print("Đang tải gói dữ liệu (có thể tốn vài giây)...")
    local_parquet_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=target_parquet,
        repo_type="dataset",
        local_dir="data/cache",
        token=my_token
    )
    
    # 3. Đọc và trích xuất Audio
    print("Đang giải nén âm thanh từ Parquet...")
    df = pd.read_parquet(local_parquet_path)
    
    # Giới hạn số lượng
    # Cột chứa audio thường tên là 'audio'
    if 'audio' not in df.columns:
        print("Không tìm thấy cột 'audio' trong file parquet.")
        print("Các cột có sẵn:", df.columns)
        exit()

    count = 0
    for idx, row in df.iterrows():
        if count >= NUM_FILES_TO_EXTRACT:
            break
            
        audio_data = row['audio']
        
        if 'bytes' in audio_data:
            audio_bytes = audio_data['bytes']
            filename = f"{LOCAL_DIR}/extra_sample_{count}.wav"
            
            with open(filename, "wb") as f:
                f.write(audio_bytes)
                
            print(f"Đã trích xuất: {filename}")
            count += 1
            
    print(f"\nHoàn tất! Đã tạo {count} file .wav trong thư mục '{LOCAL_DIR}'.")

except Exception as e:
    print(f"\nLỗi: {e}")