import os
import glob
import subprocess
from llama_cpp import Llama

# 目錄設定
input_dir = "input"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 模型路徑
model_path = "./models/JSL-MedMNX-7B-SFT-Q4_K_M.gguf"
llm = Llama(
    model_path=model_path,
    n_ctx=8192,
    n_gpu_layers=-1,     # ✅ 將前 100 層放到 GPU，加速推理
    use_mlock=True,       # ✅ 鎖定記憶體避免被換出（可選）
    use_mmap=True,        # ✅ 使用 mmap 加快載入（預設為 True）
    verbose=True          # ✅ 顯示詳細 log，幫助確認 GPU 是否啟用
)
print(llm.ctx)
print("🚀 GPU loaded successfully!")

# 處理每一份 PDF
for pdf_path in glob.glob(os.path.join(input_dir, "*.pdf")):
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    sidecar_path = os.path.join(output_dir, base_name + ".txt")

    # 執行 OCRmyPDF 並產生純文字 sidecar
    temp_output_pdf = os.path.join(output_dir, base_name + "_tmp.pdf")
    subprocess.run([
        "ocrmypdf",
        "--force-ocr",
        "--sidecar", sidecar_path,
        pdf_path,
        temp_output_pdf
    ], check=True)

    # 讀取 OCR 結果（限制長度）
    with open(sidecar_path, "r", encoding="utf-8") as f:
        ocr_text = f.read()[:8000]

    # prompt 設定：要求 markdown 結構、加上標題符號
    system_prompt = (
        "You are a professional medical assistant. Given an OCR-extracted pathology report, "
        "analyze and summarize the findings below in markdown format using heading syntax (###):\n\n"
        "### Histologic type:\n- (short summary)\n\n"
        "### Histologic grade:\n- (short summary)\n\n"
        "### Primary tumor (pT):\n- (short summary)\n\n"
        "### FINAL DIAGNOSIS:\n- (summary line)\n- (brief 2–4 line explanation based on histologic evidence)\n\n"
        "Rules:\n- Only output the markdown (no explanations or extra tags).\n"
        "Please begin the output with ### Histologic type:"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": ocr_text}
    ]

    # 執行模型推論
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=1024,
        stop=["<|end_of_turn|>", "</s>"]
    )
    output_text = response["choices"][0]["message"]["content"].strip()

    # 儲存 Markdown 結果
    md_path = os.path.join(output_dir, base_name + ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"✅ Saved clean output to {md_path}")
