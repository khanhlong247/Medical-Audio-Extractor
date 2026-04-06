# 🩺 Medical Audio Extractor

Automated pipeline for converting medical speech into structured clinical data using **Google MedASR** and **Qwen-7B LLM**.

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=flat&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E.svg?style=flat&logo=huggingface)

## 📝 Description

This project provides an automated pipeline designed to transcribe medical audio recordings and extract structured clinical information. It leverages state-of-the-art models to process audio, clean the transcript, extract key clinical entities, and map them to predefined clinical protocols (e.g., Pulmonary Embolism, Fever & Respiratory).

## 📋 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)

## ✨ Features

- 🎙️ **Speech-to-Text**: Utilizes `google/medasr` for high-accuracy medical audio transcription.
- 🧠 **Clinical Entity Extraction**: Employs `Qwen2.5-7B-Instruct` (LLM) to parse clinical data such as demographics, symptoms, and chief complaints.
- 🏥 **Clinical Logic Mapping**: Automatically detects medical scenarios and maps extracted entities to specific clinical question protocols.
- ⚙️ **Configurable Pipeline**: Modular architecture with separate engines for ASR, NLP, and Logic.

## 🛠 Tech Stack

| Component | Technology |
| :--- | :--- |
| **Core Language** | Python 3.10 |
| **Speech Model** | `google/medasr` |
| **NLP Model** | `Qwen/Qwen2.5-7B-Instruct` |
| **Libraries** | PyTorch, HuggingFace Transformers, Librosa, Pandas |

## 🚀 Installation

### Prerequisites
- Python 3.10
- GPU with 16GB+ VRAM recommended.
- HuggingFace access token (required for private models).

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/khanhlong247/Medical-Audio-Extractor
cd Medical-Audio-Extractor

# 2. Setup environment
conda create -n medasr python=3.10 -y
conda activate medasr

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install specific transformers version required for MedASR
pip install https://github.com/huggingface/transformers/archive/65dc261512cbdb1ee72b88ae5b222f2605aad8e5.zip
```

## ▶️ Usage

### Configuration
Create a `.env` file in the root directory:
```env
HF_TOKEN=your_huggingface_token_here
```

### Run the Pipeline
Ensure your audio file is placed in the `data/` folder, then execute:
```bash
python main.py
```

## 📁 Project Structure

```text
├── data/                # Audio samples and cache
├── src/                 # Core engine implementation
│   ├── asr_engine.py    # MedASR transcription logic
│   ├── nlp_engine.py    # Qwen-based entity extraction
│   └── logic_engine.py  # Clinical protocol mapping
├── main.py              # Main pipeline entry point
├── requirements.txt     # Project dependencies
└── .env                 # Environment variables
```

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements, bug fixes, or new protocol mappings.

## 🔗 Footer
**Medical-Audio-Extractor** | [https://github.com/khanhlong247/Medical-Audio-Extractor](https://github.com/khanhlong247/Medical-Audio-Extractor)
