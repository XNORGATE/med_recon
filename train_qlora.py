from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from datasets import load_dataset
import torch
from transformers import TrainerCallback
import os
# ★ 一開始先設環境變數，再匯入任何會連帶 import torch 的套件
os.environ["PYTORCH_SDP_DISABLE_FLASH"] = "1"
os.environ["PYTORCH_SDP_DISABLE_MEM_EFFICIENT"] = "1"
os.environ["PYTORCH_SDP_DISABLE"] = "1"          # 保險起見全禁

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
print("Flash SDP enabled:", torch.backends.cuda.flash_sdp_enabled())
print("Mem-efficient SDP enabled:", torch.backends.cuda.mem_efficient_sdp_enabled())

class LogCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        print(f"Step {state.global_step} start...")
    def on_step_end(self, args, state, control, **kwargs):
        if state.log_history:                       # 先確認 list 不為空
            loss = state.log_history[-1].get("loss")
            print(f"Step {state.global_step} end, loss = {loss}")
        else:
            print(f"Step {state.global_step} end (loss pending)")



MODEL_DIR  = "./mistral-7b-instruct-v0_3"
OUTPUT_DIR = "./qlora-mistral-v0.3-med"

# ---------- 資料 ----------
def merge_messages(ex):
    ex["text"] = "\n".join(m["content"] for m in ex["messages"])
    return ex

ds = load_dataset("json", data_files="finetune_train_data.jsonl", split="train")\
       .map(merge_messages, remove_columns=["messages"])

# ---------- 量化載入 ----------
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit        = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_use_double_quant = True,
    bnb_4bit_compute_dtype    = torch.bfloat16
)

tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
tok.pad_token = tok.eos_token
# import torch.utils.checkpoint as ckpt

# _orig_ckpt = ckpt.checkpoint
# def _ckpt_no_reentrant(func, *args, **kwargs):
#     kwargs.setdefault("use_reentrant", False)
#     return _orig_ckpt(func, *args, **kwargs)
# ckpt.checkpoint = _ckpt_no_reentrant


model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    quantization_config=bnb_cfg,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_cfg = LoraConfig(
    r=32, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"]
)

model = get_peft_model(model, lora_cfg)

# ---------- SFTConfig ----------
sft_cfg = SFTConfig(
    output_dir                 = OUTPUT_DIR,
    num_train_epochs           = 3,
    per_device_train_batch_size= 1,
    gradient_accumulation_steps= 8,
    learning_rate              = 2e-4,
    logging_steps              = 1,
    save_steps                 = 200,
    bf16                       = True,        # 只影響計算 dtype，不吃顯存
    optim                      = "adamw_torch",
    dataset_text_field         = "text",
    max_seq_length             = 512 ,
    packing                    = True,
)

# ---------- Trainer ----------
trainer = SFTTrainer(
    model,
    train_dataset = ds,
    args = sft_cfg,
    callbacks=[LogCallback()],
)

trainer.train()
trainer.save_model()
