from transformers import BitsAndBytesConfig
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16"
)

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

BASE_DIR   = "./mistral-7b-instruct-v0_3"
ADAPTER_DIR = "./qlora-mistral-v0.3-med"

tok = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=True)
model_base = AutoModelForCausalLM.from_pretrained(
    BASE_DIR, device_map="auto", quantization_config=bnb_cfg
)

model = PeftModel.from_pretrained(model_base, ADAPTER_DIR)
model.eval()

prompt = "Summarize this pathology report:\n..."
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=256)
print(tok.decode(out[0], skip_special_tokens=True))
