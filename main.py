import json
import warnings
from src.asr_engine import get_transcribe_function
from src.nlp_engine import get_extractor
from src.logic_engine import get_logic_engine

warnings.filterwarnings("ignore")

def main():
    print("=== MEDASR CLINICAL FLOW ===")
    
    # SETUP
    print("\nInitializing ASR Engine...")
    transcribe = get_transcribe_function()
    
    print("\nInitializing NLP Extractor...")
    extractor = get_extractor()

    # INPUT
    test_audio = "data/extra_sample_9.wav" 

    # TASK 1: TRANSCRIBE
    print(f"\nTranscribing audio: {test_audio}...")
    asr_result = transcribe(test_audio)
    
    if "error" in asr_result:
        print(f"ASR Error: {asr_result['error']}")
        return

    raw_transcript = asr_result["transcript"]
    
    # --- IN KẾT QUẢ TASK 1 ---
    print("\n" + "-"*20 + " TASK 1 OUTPUT (ASR) " + "-"*20)
    print(json.dumps(asr_result, indent=2, ensure_ascii=False))
    # ------------------------------------

    # TASK 2: EXTRACT ENTITIES
    print(f"\nExtracting Clinical Entities...")
    entities = extractor.extract_entities(raw_transcript)
    
    # --- IN KẾT QUẢ TASK 2 ---
    print("\n" + "-"*20 + " TASK 2 OUTPUT (NLP) " + "-"*20)
    print(json.dumps(entities, indent=2, ensure_ascii=False))
    # ------------------------------------

    # TASK 3: LOGIC MAPPING
    print("\nMapping to Clinical Protocol...")
    logic = get_logic_engine()
    
    # Detect Scenario
    scenario = logic.detect_scenario(entities)
    print(f"Detected Scenario: {scenario.upper()}")

    # Map Questions
    mapped_data = logic.map_entities_to_questions(entities, scenario)
    
    # --- IN KẾT QUẢ TASK 3 ---
    print("\n" + "-"*20 + " TASK 3 OUTPUT (MAPPING) " + "-"*20)
    print(json.dumps(mapped_data, indent=2, ensure_ascii=False))
    # ------------------------------------

    # FINAL OUTPUT
    final_output = {
        "raw_transcript": raw_transcript,
        "extracted_entities": entities,
        "clinical_flow": {
            "scenario": scenario,
            "answers": mapped_data
        }
    }

    print("\n" + "="*30)
    print("FINAL COMBINED RESULT")
    print("="*30)
    print(json.dumps(final_output, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()