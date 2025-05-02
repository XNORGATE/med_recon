import os
import json

# 你的 prompt，可放你剛剛提供的那段 system_prompt
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

# 設定檔案路徑
input_folder = "output"  # 修改為你txt/md的資料夾路徑
output_jsonl = "finetune_train_data.jsonl"

# 收集配對檔案
pairs = []
for file in os.listdir(input_folder):
    if file.endswith(".txt"):
        base_name = file[:-4]
        txt_path = os.path.join(input_folder, base_name + ".txt")
        md_path = os.path.join(input_folder, base_name + ".md")
        if os.path.exists(md_path):
            with open(txt_path, "r", encoding="utf-8") as f_in:
                input_text = f_in.read().strip()
            with open(md_path, "r", encoding="utf-8") as f_out:
                output_md = f_out.read().strip()
            pairs.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text},
                    {"role": "assistant", "content": output_md}
                ]
            })

# 輸出 jsonl
with open(output_jsonl, "w", encoding="utf-8") as fout:
    for item in pairs:
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ Done! Generated {len(pairs)} samples into {output_jsonl}")



{"messages": [{"role": "system", 
               "content": "You are a professional medical assistant. Given an OCR-extracted pathology report, analyze and summarize the findings below in markdown format using heading syntax (###):\n\n### Histologic type:\n- (short summary)\n\n### Histologic grade:\n- (short summary)\n\n### Primary tumor (pT):\n- (short summary)\n\n### FINAL DIAGNOSIS:\n- (summary line)\n- (brief 2–4 line explanation based on histologic evidence)\n\nRules:\n- Only output the markdown (no explanations or extra tags).\nPlease begin the output with ### Histologic type:"}, 
               {"role": "user", "content": "Page I\n\nV3.00 110614\n\nTCGA Missing Pathology Report Form\n\navailable.\n\nInstructions: The TCGA Missing Pathology Report Form should be completed for cases for which a pathology report is not\n\nUUID: E49239A6-58BD-4C7A- acep- ela\n\nTCGA-62-0023-01A-P\n\nHi I tl | | HTT Trim | i H i\n\nCRT EC TA\n\nA mil nT i iM\n\nCompleted Date (MM/ppsvyvy):__\\\\____\\\\_y_ ZON*\nGeneral Pathology Report Information |\n# Data Element Entry Alternatives Working Instructions\nProvide the tumor type of the case.\nAamM\n\n1 | Tumor type:\n\nIndicate to which BCR location the case was originally sent.\n\nelsewhere)\n\nBCR specimen originally sent O NCH\n2\nto: IGC\nProvide the date (MM/DD/YYYY) of shipment arrival at the\nDat . ved at Biospecimen Core Resource (BCR).\nate specimen received a\n3 | Date sp \\|_,lw / 2000\nBCR:\nICD-O0-3 Histology Code: Provide the histology code for the sample from the Case\nQuality Control Form (CQCF) and the overall case (patient\ndi is, if different).\n- For Specimen: (CQCF) ho COLE Sumitted lagnosis,ifdifierent)\n4 |- For Case: (patient\ndiagnosis, if available 1449/5\nelsewhere)\nICD-0-3 Site Code: Provide the site code for the sample from the Case Quality\nControl Form (CQCF) and the overall case (patient diagnosis, if\n; different).\n- For Specimen: (CQCF) vw COLE sivomitted ifferent)\n5 |- For Case: (patient\ndiagnosis, if available (7.4"},
                 {"role": "assistant", "content": "### Histologic type:\n- Adenocarcinoma\n\n### Histologic grade:\n- Grade 3\n\n### Primary tumor (pT):\n- T3\n\n### FINAL DIAGNOSIS:\n- Adenocarcinoma, Grade 3, T3\n- The histologic evidence shows a high-grade adenocarcinoma with a solid and glandular pattern, invading the muscularis propria and submucosa."}]}
