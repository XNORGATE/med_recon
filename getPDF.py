# import os
# import shutil

# # ─── 設定 ────────────────────────────────────────────────────────────────
# SRC_ROOT = r'Z:\gihub desktop\pdf_data'                       # 原始根目錄
# DST_DIR  = r'Z:\gihub desktop\med_recon\input'                 # 統一目標資料夾
# FOLDERS  = ['tcgaGBM', 'tcgaLUAD', 'tcgaLUSC', 'tcgaCOAD']     # 要處理的子資料夾
# USE_OG_SUBFOLDER = True                                        # 若 PDF 在 og_pdf 子資料夾就設 True
# # ────────────────────────────────────────────────────────────────────────

# os.makedirs(DST_DIR, exist_ok=True)

# for folder in FOLDERS:
#     # 來源路徑
#     src_dir = (os.path.join(SRC_ROOT, folder, 'og_pdf')
#                if USE_OG_SUBFOLDER
#                else os.path.join(SRC_ROOT, folder))
#     # 排序後取前 250 個
#     pdfs = sorted(f for f in os.listdir(src_dir) if f.lower().endswith('.pdf'))[:250]
#     for fname in pdfs:
#         shutil.copy(
#             os.path.join(src_dir, fname),
#             os.path.join(DST_DIR, fname)
#         )
#     print(f'✓ {folder}：已複製 {len(pdfs)} 個 PDF 到 {DST_DIR}')

import os
import shutil

# ─── 設定 ────────────────────────────────────────────────────────────────
# 請設為你的來源資料夾（包含約1000個 PDF）
SRC_DIR = r'Z:\gihub desktop\med_recon\input'
# 請設為你要複製到的新資料夾
DST_DIR = r'Z:\gihub desktop\med_recon\input\test'
# ────────────────────────────────────────────────────────────────────────

os.makedirs(DST_DIR, exist_ok=True)

# 取所有 PDF，排序後再切片
pdfs = sorted(f for f in os.listdir(SRC_DIR) if f.lower().endswith('.pdf'))
total = len(pdfs)
print(f'來源共找到 {total} 個 PDF')

# 選取第 924 到 1000（含）的檔案
start, end = 924, 1000
selected = pdfs[start-1 : end]    # 注意：list 是從 0 開始索引

# 複製到目標資料夾
for fname in selected:
    shutil.copy(os.path.join(SRC_DIR, fname),
                os.path.join(DST_DIR, fname))

print(f'已複製 {len(selected)} 個 PDF（第 {start} 到 {end}）到：{DST_DIR}')
