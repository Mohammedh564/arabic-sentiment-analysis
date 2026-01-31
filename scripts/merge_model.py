from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig
import os

def merge_and_save_model():
    # Build relative paths from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    adapter_path = os.path.join(parent_dir, "models", "transformer", "araBERT")
    output_path = os.path.join(parent_dir, "models", "transformer", "araBERT_merged")
    
    print(f"Adapter path: {adapter_path}")
    print(f"Output path: {output_path}")
    
    print("\nLoading adapter config...")
    config = PeftConfig.from_pretrained(adapter_path)
    base_model_name = config.base_model_name_or_path
    
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=3
    )
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path)
    
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)
    
    print("✓ Done! Merged model saved successfully!")
    print(f"\nMerged model location: {output_path}")

if __name__ == "__main__":
    merge_and_save_model()