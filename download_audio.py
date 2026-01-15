import os
import shutil
from dotenv import load_dotenv  # Import thư viện đọc .env
from huggingface_hub import hf_hub_download

# 1. Load biến môi trường từ file .env
load_dotenv()
my_token = os.getenv("HF_TOKEN")

# Kiểm tra token
if not my_token:
    print("⚠️  Cảnh báo: Không tìm thấy HF_TOKEN trong file .env")
    print("Code sẽ thử tải chế độ công khai (có thể thất bại với gated repo).")

# Định nghĩa thư mục đích
local_dir = "data"
filename = "test_audio.wav"  # File gốc trên HuggingFace
target_path = os.path.join(local_dir, "sample_medical.wav") # Tên file lưu trên máy

print(f"Downloading {filename} from google/medasr...")

try:
    # 2. Tải file (có truyền tham số token)
    file_path = hf_hub_download(
        repo_id="google/medasr",
        filename=filename,
        repo_type="model",
        token=my_token  # <--- QUAN TRỌNG: Token được truyền vào đây
    )
    
    # Tạo thư mục data nếu chưa có
    os.makedirs(local_dir, exist_ok=True)
    
    # Copy file từ cache về thư mục data
    shutil.copy(file_path, target_path)
    
    print(f"✅ Success! File saved to: {target_path}")

except Exception as e:
    print(f"❌ Error: {e}")
    print("Tip: Hãy kiểm tra lại token trong file .env và đảm bảo bạn đã Accept điều khoản trên web HuggingFace.")