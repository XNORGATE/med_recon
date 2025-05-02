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

# # Load local model
# llm_old = Llama(
#     model_path="./models/JSL-MedMNX-7B-SFT-Q4_K_M.gguf",
#     n_ctx=8192, n_gpu_layers=-1, use_mlock=True, use_mmap=True, verbose=True
# )

# llm_new = Llama(
#     model_path="./mistral-7b-med-merged/mistral-7b-med-q4k.gguf",
#     n_ctx=8192, n_gpu_layers=-1, use_mlock=True, use_mmap=True, verbose=True
# )

# print("✅ Local model loaded!")

# Authentication decorator


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


def run_llm(llm_name, ocr_text, sections=None):
    global llm_old, llm_new
    if llm_name == 'old':
        if llm_old is None:
            llm_old = Llama(
                model_path="./models/JSL-MedMNX-7B-SFT-Q4_K_M.gguf",
                n_ctx=8192, n_gpu_layers=-1, use_mlock=True, use_mmap=True, verbose=True
            )
        llm = llm_old
    elif llm_name == 'new':
        if llm_new is None:
            llm_new = Llama(
                model_path="./mistral-7b-med-merged/mistral-7b-med-q4k.gguf",
                n_ctx=8192, n_gpu_layers=-1, use_mlock=True, use_mmap=True, verbose=True
            )
        llm = llm_new
    prompts = {
        "Histologic type": "### Histologic type:\n- <one-line summary or \"N/A\">",
        "Histologic grade": "### Histologic grade:\n- <one-line summary or \"N/A\">",
        "Primary tumor": "### Primary tumor (pT):\n- <one-line summary or \"N/A\">",
        "FINAL DIAGNOSIS": "### FINAL DIAGNOSIS:\n- <2–4 line summary sentence or \"N/A\">"
    }

    if sections:
        selected = [prompts[k] for k in sections if k in prompts]
    else:
        selected = prompts.values()

    system_prompt = """
        You are a professional medical assistant. Given an OCR-extracted pathology report, analyze and summarize the findings below in markdown format using heading syntax (###):\n\n### Histologic type:\n- (short summary)\n\n### Histologic grade:\n- (short summary)\n\n### Primary tumor (pT):\n- (short summary)\n\n ### FINAL DIAGNOSIS:\n- (summary line)\n- (brief 2–4 line explanation based on histologic evidence)\n\n Rules:\n- Only output the markdown (no explanations or extra tags).\nPlease begin the output with ### Histologic type:"
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": ocr_text}
    ]

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=1024,
        temperature=0.3,      # 🔵 降低隨機性，避免亂跳
        top_p=0.9,            # 🔵 降低發散程度
        frequency_penalty=0.5,  # 🔵 防止重複
        presence_penalty=0.2,  # 🔵 適度鼓勵少量新資訊
        stop=["Specimen","<|end_of_turn|>"]  # 🔵 強制遇到標題就停
    )

    output = response['choices'][0]['message']['content'].strip()
    if "### Histologic type:" in output:
        output = output[output.index("### Histologic type:"):]
    return output


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
        "ocrmypdf", "--force-ocr", "--sidecar", sidecar_txt, tmp_path, ocr_output_pdf
    ], check=True)

    # Read OCR text
    with open(sidecar_txt, "r", encoding="utf-8") as f:
        ocr_text = f.read()[:8000]
        if model == 'old':
            markdown = run_llm('old', ocr_text)
        elif model == 'new':
            markdown = run_llm('new', ocr_text)
        elif model == 'mixed':
            md1 = run_llm('old', ocr_text, sections=[
                "Histologic type", "Histologic grade", "Primary tumor"
            ])
            md2 = run_llm('new', ocr_text, sections=["FINAL DIAGNOSIS"])
            markdown = md1 + "\n\n" + md2
        else:
            raise ValueError("Invalid model option")

    # # System prompt with strict template instructions
    #     system_prompt = """
    #     You are a professional medical assistant. I will provide you with the raw OCR text of a pathology report.
    #     Your TASK is to extract exactly four pieces of information and output ONLY the following Markdown template, with no extra text or explanation:

    #     ### Histologic type:
    #     - <one-line summary or "N/A">

    #     ### Histologic grade:
    #     - <one-line summary or "N/A">

    #     ### Primary tumor (pT):
    #     - <one-line summary or "N/A">

    #     ### FINAL DIAGNOSIS:
    #     - <summary sentence or "N/A">
    #     - <2–4 line explanation or "N/A">

    #     RULES:
    #     1. Follow the above header order exactly, with no additional blank lines or sections.
    #     2. If a field cannot be determined, respond with "N/A".
    #     3. Do not include any additional headings, comments, system messages, or footnotes.
    #     4. Your output MUST begin immediately with "### Histologic type:" as the first line. Do not write anything before it.
    #     """

    # messages = [
    #     {"role": "system", "content": system_prompt},
    #     {"role": "user", "content": ocr_text}
    # ]

    # # Call LLM
    # response = llm.create_chat_completion(
    #     messages=messages,
    #     max_tokens=1024,
    #     stop=["<|end_of_turn|>"]
    # )
    # markdown = response['choices'][0]['message']['content'].strip()

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

# Route: check job status and result


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

# Serve frontend static files


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
                md1 = run_llm('old', ocr_text, sections=[
                    "Histologic type", "Histologic grade", "Primary tumor"
                ])
                md2 = run_llm('new', ocr_text, sections=["FINAL DIAGNOSIS"])
                markdown = md1 + "\n\n" + md2
            else:
                raise ValueError("Invalid model option")

            start = markdown.find("### Histologic type:")
            if start != -1:
                markdown = markdown[start:]
            markdown = markdown[:10000]  # 最多保護到1萬字

        #     system_prompt = """
        #     You are a professional medical assistant. I will provide you with the raw OCR text of a pathology report.
        #     Your TASK is to extract exactly four pieces of information and output ONLY the following Markdown template, with no extra text or explanation:

        #     ### Histologic type:
        #     - <one-line summary or "N/A">

        #     ### Histologic grade:
        #     - <one-line summary or "N/A">

        #     ### Primary tumor (pT):
        #     - <one-line summary or "N/A">

        #     ### FINAL DIAGNOSIS:
        #     - <summary sentence or "N/A">
        #     - <2–4 line explanation or "N/A">

        #     RULES:
        #     1. Follow the above header order exactly, with no additional blank lines or sections.
        #     2. If a field cannot be determined, respond with "N/A".
        #     3. Do not include any additional headings, comments, system messages, or footnotes.
        #     4. Your output MUST begin immediately with "### Histologic type:" as the first line. Do not write anything before it.
        #     """

        # messages = [
        #     {"role": "system", "content": system_prompt},
        #     {"role": "user", "content": ocr_text}
        # ]

        # response = llm.create_chat_completion(
        #     messages=messages,
        #     max_tokens=1024,
        #     stop=["<|end_of_turn|>"]
        # )
        # markdown = response["choices"][0]["message"]["content"].strip()

        # ===== post-process: 去掉任何 ### Histologic type: 之前的內容 =====

        return jsonify({'markdown': markdown})

        # filenames = [file.filename for file in uploaded_files]
        # files_info = []
        # for file in uploaded_files:
        #     filetype = file.content_type or "application/octet-stream"
        #     file.stream.seek(0)
        #     encoded = base64.b64encode(file.read()).decode('utf-8')
        #     url = f"data:{filetype};base64,{encoded}"
        #     files_info.append(
        #         {"name": file.filename, "type": filetype, "url": url})
        #     file.stream.seek(0)

        # return jsonify({'markdown': markdown, 'filenames': filenames, 'files': files_info})

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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=85)
