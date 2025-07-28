import os
import base64
import tempfile
import subprocess
from flask import Flask, request, jsonify, send_from_directory
import os
from flask_cors import CORS
from dotenv import load_dotenv
from llama_cpp import Llama
from google.oauth2 import id_token
from google.auth.transport import requests
import jwt
import datetime
from functools import wraps
import threading
import uuid
from werkzeug.datastructures import FileStorage
import textwrap


# # 暫存所有 job 的狀態與結果
# jobs = {}  # 格式：jobs[job_id] = {'status':'running'|'done'|'error', 'result': markdown 或 error 訊息}


# Initialize Flask app and CORS
app = Flask(__name__, static_folder='frontend')
CORS(app, supports_credentials=True, origins=[
    "http://127.0.0.1:5500",
    "https://medrecon.xnor-development.com"
])

# Load environment variables
load_dotenv()
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = "application/json;charset=utf-8"

JWT_SECRET = os.getenv('JWT_SECRET')
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
DEBUG_BYPASS_AUTH = os.getenv("DEBUG_BYPASS_AUTH", "false").lower() == "true"


def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if request.method == 'OPTIONS':
            return '', 200
        if DEBUG_BYPASS_AUTH:
            return f(*args, **kwargs)
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': 'No authorization token provided'}), 401
        try:
            token = auth_header.split(' ')[1]
            payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            request.user = payload
            return f(*args, **kwargs)
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
    return decorated


# In-memory job store
jobs = {}  # job_id -> {'status':'running'|'done'|'error', 'result': markdown or error message}
llm_old = None
llm_new = None


def load_model(path: str, target_gpu_layers: int):
    """
    動態嘗試不同 GPU 層數載入模型，若 OOM 則自動 fallback。
    """
    # 依序嘗試 target_gpu_layers, 一半, 最後全部 CPU
    # 縮到 2048，並開啟 8-bit KV cache
    base_kwargs = dict(
        model_path=path,
        n_ctx=2048,
        f16_kv=True,               # 讓 KV cache 8-bit
        use_mlock=True,
        use_mmap=True,
        n_threads=os.cpu_count(),
        verbose=True
    )
    for gpu_layers in (target_gpu_layers, target_gpu_layers // 2, 0):
        try:
            return Llama(**base_kwargs, n_gpu_layers=gpu_layers)

        except RuntimeError:
            print(f"⚠️ 無法在 GPU 上載入 {gpu_layers} 層，改用更低層數或 CPU... ")
    # 最後全部跑 CPU
    return Llama(**base_kwargs, n_gpu_layers=0)


# 在檔案開頭先 import textwrap，並定義完整模板
import textwrap

SYSTEM_PROMPT = textwrap.dedent("""\
You are a professional medical assistant. Given an OCR-extracted pathology report, extract **exactly** the following four sections **in this order**, using markdown headings (###). **Do not** output any other headings, sections, or narrative text:

### Histologic type:
- <short, one-line summary or "N/A">

### Histologic grade:
- <short, one-line summary or "N/A">

### Primary tumor (pT):
- <short, one-line summary or "N/A">

### FINAL DIAGNOSIS:
- <**2 to 4 summary sentences** of explanation based on histologic evidence. Provide exactly 2-4 summary sentences, each on its own line.>

Begin output **immediately** with the first heading and nothing else.
""")

def run_llm(llm_name, ocr_text, sections=None):
    global llm_old, llm_new

    # 動態載入或重用模型
    if llm_name == 'old':
        if llm_old is None:
            llm_old = load_model(
                "./models/JSL-MedMNX-7B-SFT-Q4_K_M.gguf",
                target_gpu_layers=16
            )
        llm = llm_old
    else:  # new
        if llm_new is None:
            llm_new = load_model(
                "./mistral-7b-med-merged/mistral-7b-med-q4k.gguf",
                target_gpu_layers=16
            )
        llm = llm_new

    # 根據 sections 參數動態建立 system prompt
    if sections:
        parts = []
        for sec in sections:
            if sec == "Histologic type":
                parts.append("### Histologic type:\n- <short, one-line summary or \"N/A\">")
            elif sec == "Histologic grade":
                parts.append("### Histologic grade:\n- <short, one-line summary or \"N/A\">")
            elif sec == "Primary tumor":
                parts.append("### Primary tumor (pT):\n- <short, one-line summary or \"N/A\">")
            elif sec == "FINAL DIAGNOSIS":
                parts.append(
                    "### FINAL DIAGNOSIS:\n"
                    "- <one summary sentence or \"N/A\">\n"
                    "- <2–4 sentences of explanation based on histologic evidence, each on its own line. Provide exactly 2–4 sentences.>"
                )
        system_prompt = "You are a professional medical assistant. Extract exactly the following sections in order, using markdown headings (###). Do not output any other text or headings:\n\n" \
                        + "\n\n".join(parts) + "\n\nBegin your output immediately with the first heading and nothing else."
    else:
        # 全模板
        system_prompt = SYSTEM_PROMPT

    messages = [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": ocr_text}
    ]

    # 呼叫 LLM，不用額外 stop token，讓它跑完所有 bullet
    res = llm.create_chat_completion(
        messages=messages,
        max_tokens=1024,
        temperature=0.2,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.1
    )

    output = res["choices"][0]["message"]["content"].strip()
    # 確保從第一個 ### 開始
    idx = output.find("### ")
    return output[idx:] if idx != -1 else output


# Conversion helper: runs OCR and LLM to produce markdown


def do_conversion(file_bytes: bytes, model: str) -> str:
    from PIL import Image

    # Save bytes to temp PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    sidecar_txt = tmp_path + ".txt"
    ocr_output_pdf = tmp_path.replace(".pdf", "_ocr.pdf")

    # Run OCR
    subprocess.run([
        "ocrmypdf",
        "--force-ocr",
        "--image-dpi", "300",      # ← 指定一個合理的掃描解析度
        "--sidecar", sidecar_txt,
        tmp_path,
        ocr_output_pdf
    ], check=True)
    # Read OCR text
    with open(sidecar_txt, "r", encoding="utf-8") as f:
        ocr_text = f.read()[:5000]
        if model == 'old':
            markdown = run_llm('old', ocr_text)
        elif model == 'new':
            markdown = run_llm('new', ocr_text)
        elif model == 'mixed':
            # 舊模型跑前三節
            md1 = run_llm('old', ocr_text, sections=[
                "Histologic type", "Histologic grade", "Primary tumor"
            ])
            # 新模型只跑 FINAL DIAGNOSIS，並擷取這一段
            md2_raw = run_llm('new', ocr_text, sections=["FINAL DIAGNOSIS"])
            start2 = md2_raw.find("### FINAL DIAGNOSIS:")
            if start2 != -1:
                # 找下一個 ### (如果有)，否則取到尾
                next_h = md2_raw.find("\n### ", start2 + 1)
                md2 = md2_raw[start2: next_h] if next_h != -1 else md2_raw[start2:]
            else:
                md2 = md2_raw
            # 合併時去除多餘空行
            markdown = md1.rstrip() + "\n\n" + md2.lstrip()
        else:
            raise ValueError("Invalid model option")

   

    # 強制清除：只留從 ### Histologic type: 開始的部分
    start = markdown.find("### Histologic type:")
    if start != -1:
        markdown = markdown[start:]

    markdown = markdown[:10000]  # 最多保護到1萬字

    # Clean up temp files
    for path in (tmp_path, sidecar_txt, ocr_output_pdf):
        try:
            os.remove(path)
        except:
            pass

    return markdown

# Route: submit conversion job


@app.route('/submit_conversion', methods=['POST', 'OPTIONS'])
@require_auth
def submit_conversion():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    data = file.read()  # read file bytes once
    model_choice = request.form.get('model', 'new')
    job_id = str(uuid.uuid4())
    jobs[job_id] = {'status': 'running', 'result': None}

    # Background worker
    def worker(data_bytes, jid):
        try:
            # md = do_conversion(data_bytes)
            md = do_conversion(data_bytes, model_choice)

            jobs[jid] = {'status': 'done', 'result': md}
        except Exception as e:
            jobs[jid] = {'status': 'error', 'result': str(e)}

    threading.Thread(target=worker, args=(data, job_id), daemon=True).start()

    return jsonify({'job_id': job_id}), 202




@app.route('/job_status/<job_id>', methods=['GET'])
@require_auth
def job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Unknown job_id'}), 404
    if job['status'] == 'running':
        return jsonify({'status': 'running'}), 202
    if job['status'] == 'done':
        return jsonify({'status': 'done', 'markdown': job['result']}), 200
    # error case
    return jsonify({'status': 'error', 'error': job['result']}), 500



@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'converter.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# Existing merge_and_convert route remains unchanged


@app.route('/merge_and_convert', methods=['POST', 'OPTIONS'])
@require_auth
def merge_and_convert():
    from PIL import Image
    from PyPDF2 import PdfMerger
    import io

    uploaded_files = request.files.getlist("files")
    if not uploaded_files:
        return jsonify({'error': 'No files provided'}), 400

    temp_files = []
    pdf_paths = []

    try:
        for file in uploaded_files:
            filename = file.filename.lower()
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(file.stream).convert('RGB')
                temp_pdf = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf")
                img.save(temp_pdf.name, "PDF")
                temp_files.append(temp_pdf.name)
                pdf_paths.append(temp_pdf.name)
            elif filename.endswith('.pdf'):
                temp_pdf = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf")
                file.save(temp_pdf.name)
                temp_files.append(temp_pdf.name)
                pdf_paths.append(temp_pdf.name)
            else:
                return jsonify({'error': f'Unsupported file format: {filename}'}), 400

        merged_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        merger = PdfMerger()
        for path in pdf_paths:
            merger.append(path)
        merger.write(merged_pdf.name)
        merger.close()

        sidecar_txt = merged_pdf.name + ".txt"
        ocr_output_pdf = merged_pdf.name.replace(".pdf", "_ocr.pdf")

        subprocess.run([
            "ocrmypdf", "--force-ocr", "--sidecar", sidecar_txt, merged_pdf.name, ocr_output_pdf
        ], check=True)

        with open(sidecar_txt, "r", encoding="utf-8") as f:
            ocr_text = f.read()[:16000]
            model_choice = request.form.get('model', 'new')  # 預設new

            if model_choice == 'old':
                markdown = run_llm('old', ocr_text)
            elif model_choice == 'new':
                markdown = run_llm('new', ocr_text)
            elif model_choice == 'mixed':
                # 舊模型取前三節
                md1 = run_llm('old', ocr_text, sections=[
                    "Histologic type", "Histologic grade", "Primary tumor"
                ])
                # 新模型只跑 FINAL DIAGNOSIS，並截取該段
                md2_raw = run_llm('new', ocr_text, sections=["FINAL DIAGNOSIS"])
                start2 = md2_raw.find("### FINAL DIAGNOSIS:")
                if start2 != -1:
                    next_h = md2_raw.find("\n### ", start2 + 1)
                    md2 = md2_raw[start2: next_h] if next_h != -1 else md2_raw[start2:]
                else:
                    md2 = md2_raw
                markdown = md1.rstrip() + "\n\n" + md2.lstrip()
            else:
                raise ValueError("Invalid model option")

            start = markdown.find("### Histologic type:")
            if start != -1:
                markdown = markdown[start:]
            markdown = markdown[:10000]  # 最多保護到1萬字



        return jsonify({'markdown': markdown})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Merge conversion failed: {str(e)}'}), 500
    finally:
        for path in temp_files + [merged_pdf.name, sidecar_txt, ocr_output_pdf]:
            try:
                os.remove(path)
            except:
                pass

import tempfile
import time

def cleanup_temp_files(older_than_secs=3600):

    tmpdir = tempfile.gettempdir()
    now = time.time()
    for name in os.listdir(tmpdir):
        path = os.path.join(tmpdir, name)
        try:
            if not os.path.isfile(path):
                continue
            if not (name.endswith('.pdf') or name.endswith('.txt') or name.endswith('_ocr.pdf')):
                continue
            if os.path.getmtime(path) < now - older_than_secs:
                os.remove(path)
        except Exception:
            pass

if __name__ == '__main__':
    cleanup_temp_files(older_than_secs=600)  
    app.run(debug=True, host='0.0.0.0', port=85)
