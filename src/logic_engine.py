import json

# Định nghĩa "Registry" (Bộ câu hỏi theo kịch bản)
QUESTION_REGISTRY = {
    "fever_respiratory": {
        "scenario_name": "Fever & Respiratory",
        "question_ids": [1, 5, 7, 12],
        "keywords": ["cough", "sore throat", "fever", "breath", "respiratory"]
    },
    "abdominal_pain": {
        "scenario_name": "Abdominal Pain",
        "question_ids": [2, 8, 15, 20],
        "keywords": ["stomach", "abdominal", "belly", "pain", "vomit"]
    },
    "headache": {
        "scenario_name": "Headache",
        "question_ids": [3, 9, 14],
        "keywords": ["headache", "migraine", "head"]
    },
    "chest_pain_pe": { # Thêm kịch bản khớp với file audio mẫu
        "scenario_name": "Pulmonary Embolism (PE) Protocol",
        "question_ids": [101, 102, 103, 104],
        "keywords": ["pe", "chest", "pulmonary", "embolism", "shortness of breath"]
    }
}

# Định nghĩa Schema câu hỏi (Nội dung câu hỏi)
QUESTION_SCHEMA = {
    1: {"text": "What is the patient's age?", "type": "demographics"},
    5: {"text": "Is fever present?", "type": "boolean"},
    7: {"text": "Is cough present?", "type": "boolean"},
    101: {"text": "Is there shortness of breath?", "type": "boolean"},
    102: {"text": "Patient Age", "type": "integer"},
    103: {"text": "Patient Gender", "type": "string"},
    104: {"text": "Exam Type", "type": "string"}
}

class ClinicalLogicEngine:
    def detect_scenario(self, entities: dict) -> str:
        """
        Dựa vào triệu chứng và chief_complaint để đoán bệnh (Scenario).
        """
        # Gộp tất cả text quan trọng lại để search từ khóa
        text_corpus = (
            entities.get("chief_complaint", {}).get("value", "") + " " +
            " ".join([s.get("value", "") for s in entities.get("symptoms", [])])
        ).lower()

        best_scenario = "unknown"
        max_matches = 0

        for scenario, config in QUESTION_REGISTRY.items():
            matches = sum(1 for kw in config["keywords"] if kw in text_corpus)
            if matches > max_matches:
                max_matches = matches
                best_scenario = scenario
        
        return best_scenario

    def map_entities_to_questions(self, entities: dict, scenario: str) -> list:
        """
        Điền câu trả lời vào Question IDs dựa trên kịch bản.
        """
        if scenario == "unknown":
            return []

        mapped_answers = []
        target_q_ids = QUESTION_REGISTRY[scenario]["question_ids"]
        
        for q_id in target_q_ids:
            answer = None
            conf = 0.0
            
            if q_id == 101: # Shortness of breath
                symptoms = [s.get("value", "").lower() for s in entities.get("symptoms", [])]
                if any("breath" in s for s in symptoms):
                    answer = "Yes"
                    conf = 1.0
            
            elif q_id == 102: # Age
                answer = entities.get("demographics", {}).get("age", {}).get("value")
                conf = entities.get("demographics", {}).get("age", {}).get("confidence", 0.0)

            elif q_id == 103: # Gender
                answer = entities.get("demographics", {}).get("gender", {}).get("value")
                conf = entities.get("demographics", {}).get("gender", {}).get("confidence", 0.0)

            if answer and answer != "not_stated":
                mapped_answers.append({
                    "question_id": q_id,
                    "question_text": QUESTION_SCHEMA.get(q_id, {}).get("text"),
                    "answer": answer,
                    "confidence": conf,
                    "auto_fill": conf >= 0.7
                })

        return mapped_answers

# Singleton
_logic = None
def get_logic_engine():
    global _logic
    if _logic is None:
        _logic = ClinicalLogicEngine()
    return _logic