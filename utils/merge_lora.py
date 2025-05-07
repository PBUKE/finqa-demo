import argparse
from transformers import AutoModelForCausalLM
from peft import PeftModel

def merge_and_save(base_path, adapter_path, output_path):
    base_model = AutoModelForCausalLM.from_pretrained(base_path)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(output_path)
    print(f"✅ Merged model saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter with a base model.")
    parser.add_argument("--base", required=True, help="Path to the base model (local or HF hub)")
    parser.add_argument("--adapter", required=True, help="Path to the LoRA adapter directory")
    parser.add_argument("--output", required=True, help="Path to save the merged model")
    args = parser.parse_args()

    merge_and_save(args.base, args.adapter, args.output)