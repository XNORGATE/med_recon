from gguf import convert_hf
# 目標：fp16 → Q4_K_M（或你選擇的量化方案）
convert_hf(
    model_dir=MERGED_DIR,
    gguf_output_path="./mistral-7b-med-merged-q4k.gguf",
    quantization="Q4_K_M"
)

