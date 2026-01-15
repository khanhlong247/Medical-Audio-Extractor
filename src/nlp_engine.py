import re
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

class ClinicalEntityExtractor:
    def __init__(self):
        self.model_id = "Qwen/Qwen2.5-7B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading LLM ({self.model_id}) on {self.device}...")
        
        # Load model & tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        print("LLM loaded successfully.")

    def clean_transcript(self, raw_text: str) -> str:
        """
        Chuyển đổi token đặc biệt của MedASR thành dấu câu chuẩn.
        """
        # Map các token đặc biệt sang ký tự thường
        replacements = {
            r"\{period\}": ".",
            r"\{comma\}": ",",
            r"\{colon\}": ":",
            r"\{question_mark\}": "?",
            r"\{new_paragraph\}": "\n",
            r"\{new paragraph\}": "\n",
            r"\[.*?\]": "",
        }
        
        cleaned_text = raw_text
        for pattern, replacement in replacements.items():
            cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)
        
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

    def extract_entities(self, transcript: str) -> dict:
        """
        Dùng LLM để trích xuất thông tin theo cấu trúc JSON.
        """
        cleaned_text = self.clean_transcript(transcript)
        print(f"Cleaned Text: {cleaned_text[:100]}...")

        # System Prompt
        system_prompt = """You are a medical scribe AI. Your job is to extract explicit clinical entities from the transcript.
Rules:
1. Extract ONLY explicitly stated information. DO NOT infer.
2. If information is missing, use "not_stated".
3. Return valid JSON only.
4. Output format:
{
  "chief_complaint": {"value": "string", "confidence": float (0.0-1.0), "source_phrase": "string"},
  "symptoms": [{"value": "string", "confidence": float, "source_phrase": "string"}],
  "duration": {"value": "string", "confidence": float, "source_phrase": "string"},
  "severity": {"value": "string", "confidence": float, "source_phrase": "string"},
  "demographics": {
    "age": {"value": "string", "confidence": float},
    "gender": {"value": "string", "confidence": float}
  }
}"""

        user_prompt = f"Transcript: {cleaned_text}\n\nExtract entities:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        outputs = self.pipe(
            prompt,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.1,
            top_p=0.9,
            return_full_text=False
        )

        generated_text = outputs[0]["generated_text"]
        
        # Parse JSON từ output của LLM
        try:
            json_match = re.search(r"\{.*\}", generated_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data
            else:
                return {"error": "No JSON found in LLM output", "raw": generated_text}
        except Exception as e:
            return {"error": f"JSON parsing failed: {str(e)}", "raw": generated_text}

_extractor = None
def get_extractor():
    global _extractor
    if _extractor is None:
        _extractor = ClinicalEntityExtractor()
    return _extractor