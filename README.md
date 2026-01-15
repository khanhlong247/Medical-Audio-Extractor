# MedASR Clinical Flow Integration

Automated pipeline for converting medical speech into structured clinical data using **Google MedASR** and **Qwen-7B LLM**.

## 🚀 Features
- **Speech-to-Text**: Integration with Google's `medasr` (running on local GPU).
- **Entity Extraction**: Uses `Qwen2.5-7B-Instruct` to parse clinical entities (Symptoms, Demographics, etc.).
- **Clinical Logic**: Auto-detects scenarios (e.g., Pulmonary Embolism) and maps answers to clinical questions.

## 🛠 Installation

**Prerequisites:**
- Python 3.10
- GPU with 16GB+ VRAM (Tested on Tesla P40 24GB)
- HuggingFace Token (with access to `google/medasr`)

**Setup:**

```bash
# 1. Clone repo
git clone [https://github.com/your-username/MedASR-Clinical-Flow.git](https://github.com/your-username/MedASR-Clinical-Flow.git)
cd MedASR-Clinical-Flow

# 2. Create environment & Install dependencies
conda create -n medasr python=3.10 -y
conda activate medasr
pip install uv
uv pip install -r requirements.txt

# 3. Fix transformers version for MedASR (Important!)
# This installs the specific commit required by Google's model
uv pip install --force-reinstall [https://github.com/huggingface/transformers/archive/65dc261512cbdb1ee72b88ae5b222f2605aad8e5.zip](https://github.com/huggingface/transformers/archive/65dc261512cbdb1ee72b88ae5b222f2605aad8e5.zip)
```

## ⚙️ Configuration

### 1. Create a .env file in the root directory:

```
HF_TOKEN=hf_your_huggingface_token_here
```

### 2. Download sample audio (optional if you have your own):

```
python download_audio.py
```

## ▶️ Usage

Run the main pipeline:

```
# Use -s flag to avoid local package conflicts
python -s main.py
```

## 📊 Sample Output

```
{
  "clinical_flow": {
    "scenario": "chest_pain_pe",
    "answers": [
      {
        "question_id": 101,
        "question_text": "Is there shortness of breath?",
        "answer": "Yes",
        "confidence": 1.0,
        "auto_fill": true
      }
    ]
  }
}
```