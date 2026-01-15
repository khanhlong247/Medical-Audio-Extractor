import os
import torch
import librosa
import warnings
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()
my_token = os.getenv("HF_TOKEN")

warnings.filterwarnings("ignore")

class MedASRService:
    def __init__(self):
        self.model_id = "google/medasr"
        
        if torch.cuda.is_available():
            self.device = 0
            device_name = torch.cuda.get_device_name(0)
            print(f"Found GPU: {device_name}")
            self.torch_dtype = torch.float32 
        else:
            self.device = -1
            self.torch_dtype = torch.float32
            print("GPU not found. Using CPU (slow).")

        print(f"Loading MedASR model on device map: {self.device}...")
        
        try:
            self.pipe = pipeline(
                "automatic-speech-recognition", 
                model=self.model_id, 
                device=self.device,
                torch_dtype=self.torch_dtype,
                token=my_token
            )
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def transcribe_audio(self, audio_path: str) -> dict:
        """
        Transcribes audio file using MedASR.
        """
        if not os.path.exists(audio_path):
            return {"error": "File not found", "path": audio_path}
        
        try:
            # 1. Load Audio
            speech, sample_rate = librosa.load(audio_path, sr=16000)
            
            if len(speech) == 0:
                return {"error": "Empty audio file"}

            # 2. Inference
            result = self.pipe(
                speech, 
                chunk_length_s=20, 
                stride_length_s=2
            )

            # 3. Handle Output
            if isinstance(result, dict):
                transcript = result.get('text', '')
                segments = result.get('chunks', [])
            elif isinstance(result, list):
                transcript = " ".join([chunk.get('text', '') for chunk in result])
                segments = result
            else:
                transcript = str(result)
                segments = []

            # Clean text
            transcript = transcript.strip()

            # Mock confidence
            confidence = 0.95 if len(transcript) > 0 else 0.0
            
            return {
                "transcript": transcript,
                "confidence": confidence,
                "segments": segments
            }

        except Exception as e:
            print(f"DEBUG Error: {e}")
            return {"error": str(e)}

_service = None

def get_transcribe_function():
    global _service
    if _service is None:
        _service = MedASRService()
    return _service.transcribe_audio