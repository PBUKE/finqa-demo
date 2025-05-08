from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = "./models/gptneo_model"  # Replace with your GPT-Neo model path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=dtype).to(device)

def gptneo_infer(question, context):
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            temperature=0.0
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("Answer:")[-1].strip()