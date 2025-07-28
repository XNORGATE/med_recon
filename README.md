# Pathology Report Refinement Tool

A real-time pathology report optimization system utilizing advanced Optical Character Recognition (OCR) and fine-tuned Large Language Models (LLMs) to streamline medical documentation.

## 🚀 Overview

This tool automates the digitization and standardization of medical pathology reports. By combining precise OCR technology with customized language models, it ensures efficient and accurate report processing, significantly reducing manual workload and minimizing errors.

## 🎯 Key Features

* **Advanced OCR Processing**: Leveraging myOCRpdf with Tesseract, including deskewing, denoising, and block segmentation, supporting mixed-language (English and Chinese) documents.
* **Fine-Tuned Language Models**: Employs Mistral-7B-Instruct-v0.3 with QLoRA optimization (4-bit quantization, LoRA adapters, paged AdamW optimizer, and gradient checkpointing).
* **Real-Time Conversion**: Instantly converts PDF uploads to structured Markdown reports.
* **User-Friendly Web UI**: Allows easy editing, validation, and submission of processed reports.
* **Secure Data Management**: Utilizes Flask APIs, MySQL databases, and encrypted HTTPS communications for robust data handling and security.

## 🛠️ Technology Stack

* **OCR**: Tesseract OCR integrated via myOCRpdf
* **LLM**: Mistral-7B-Instruct-v0.3 (Fine-tuned via QLoRA)
* **Framework**: Flask
* **Database**: MySQL
* **Deployment & Security**: Cloudflare Tunnel, Google OAuth

## 📂 Project Structure

```
project-root/
├── backend/
│   ├── flask_api/
│   └── model_inference/
├── frontend/
│   ├── src/
│   └── public/
├── OCR_modules/
├── models/
│   └── fine_tuned_LLM/
└── database/
```

## 🚧 Installation & Setup

Clone the repository:

```bash
git clone https://github.com/yourusername/pathology-report-refinement.git
cd pathology-report-refinement
```

Set up environment:

```bash
pip install -r requirements.txt
```

Run backend services:

```bash
python backend/flask_api/app.py
```

Run frontend:

```bash
npm install
npm start
```

## 📖 Usage

* Navigate to the web interface.
* Log in using your Google account.
* Upload pathology report PDFs.
* Validate and download processed Markdown files.

## 📈 Performance Metrics

* **ROUGE-L**: Improved from 0.282 to 0.292
* **Inference Time**: Detailed output \~98.2 sec/report

## 🔒 Security

* Data stored securely in MySQL databases.
* HTTPS encrypted data transfers.
* OAuth authentication.

## 📌 Future Directions

* Automation of data augmentation.
* Reinforcement learning with human feedback (RLHF).
* Microservice and SaaS platform deployment.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
