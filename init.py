import os
import glob
import subprocess
import sys
import shutil
from pathlib import Path
from llama_cpp import Llama

# 目錄設定
input_dir = "input/test"  # 請設為你的來源資料夾
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 確認 Ghostscript 可執行檔
GS_CMD = shutil.which("gs") or shutil.which("gswin64c") or shutil.which("gswin32c")
if GS_CMD is None:
    print("Ghostscript 可執行檔 gs/gswin64c 未找到，請安裝並加入 PATH。")
    sys.exit(1)

# 模型設定
model_path = "./models/JSL-MedMNX-7B-SFT-Q4_K_M.gguf"
llm = Llama(
    model_path=model_path,
    n_ctx=8192,
    n_gpu_layers=-1,     # 加速推理
    use_mlock=True,      # 鎖定記憶體避免換出
    use_mmap=True,       # mmap 加快載入
    verbose=True         # 詳細 log
)
print(llm.ctx)
print("🚀 GPU loaded successfully!")

# 處理每一份 PDF
for pdf_path in glob.glob(os.path.join(input_dir, "*.pdf")):
    base_name    = os.path.splitext(os.path.basename(pdf_path))[0]
    sidecar_path = os.path.join(output_dir, base_name + ".txt")
    temp_output  = os.path.join(output_dir, base_name + "_tmp.pdf")
    fixed_pdf    = os.path.join(output_dir, base_name + "_fixed.pdf")

    # 嘗試直接 OCR
    try:
        subprocess.run([
            "ocrmypdf",
            "--force-ocr",
            "--sidecar", sidecar_path,
            "--continue-on-soft-render-error",
            pdf_path, temp_output
        ], check=True, capture_output=True, text=True)

    except subprocess.CalledProcessError as e:
        err = e.stderr or ""
        if "NegativeDimensionError" in err:
            print(f"⚠️ {base_name} 遇到 NegativeDimensionError，使用 Ghostscript 重寫 PDF …")
            # 用 Ghostscript 重寫 PDF
            subprocess.run([
                GS_CMD, "-q", "-o", fixed_pdf,
                "-sDEVICE=pdfwrite",
                "-dSAFER", "-dNOPAUSE", "-dBATCH",
                pdf_path
            ], check=True)
            # 再次執行 OCR
            subprocess.run([
                "ocrmypdf",
                "--force-ocr",
                "--sidecar", sidecar_path,
                "--continue-on-soft-render-error",
                fixed_pdf, temp_output
            ], check=True)
            os.remove(fixed_pdf)
        else:
            print(f"❌ {base_name} OCR 失敗：{err}")
            continue

    # 讀取 OCR 結果
    with open(sidecar_path, "r", encoding="utf-8") as f:
        ocr_text = f.read()[:8000]

    # 準備 prompt
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

    # 模型推論
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=1024,
        stop=["<|end_of_turn|>", "</s>"]
    )
    output_text = response["choices"][0]["message"]["content"].strip()

    # 儲存結果
    md_path = os.path.join(output_dir, base_name + ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"✅ Saved clean output to {md_path}")
