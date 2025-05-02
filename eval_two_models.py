# -*- coding: utf-8 -*-
import json, re, time, csv, argparse, os, pandas as pd
from pathlib import Path
from rouge import Rouge
from evaluate import load as load_metric
from llama_cpp import Llama

# ---------- ??? ----------
ap = argparse.ArgumentParser()
ap.add_argument('--start',      type=int, default=1,    help='')
ap.add_argument('--limit',      type=int, default=500,  help='')
ap.add_argument('--model_old',  default=r'Z:\gihub desktop\med_recon\models\JSL-MedMNX-7B-SFT-Q4_K_M.gguf')
ap.add_argument('--model_new',  default=r'Z:\gihub desktop\med_recon\mistral-7b-med-merged\mistral-7b-med-q4k.gguf')
ap.add_argument('--data',       default=r'Z:\gihub desktop\med_recon\finetune_train_data.jsonl')
ap.add_argument('--ctx',        type=int, default=8192)
ap.add_argument('--gpu_layers', type=int, default=35)
ap.add_argument('--batch',      type=int, default=1024)
args = ap.parse_args()

# ---------- Llama init ----------
os.environ['LLAMA_CUBLAS'] = '1'        # ?j?�� CUDA kernel (?? GPU ?? wheel)
def load_llm(path):
    return Llama(model_path=path, n_ctx=args.ctx, n_gpu_layers=args.gpu_layers,
                 n_batch=args.batch, use_mmap=True, use_mlock=True, verbose=False)
models = {'old': load_llm(args.model_old), 'new': load_llm(args.model_new)}

# ---------- ???? ----------
rouge = Rouge()
bleu  = load_metric('sacrebleu')
bert  = load_metric('bertscore')
def metrics(pred, ref):
    ref  = re.sub(r'\s+',' ', ref.strip())
    pred = re.sub(r'\s+',' ', pred.strip()) or ' '
    r = rouge.get_scores(pred, ref, avg=True)['rouge-l']['f']
    b = bleu.compute(predictions=[pred], references=[[ref]])['score']/100
    bf = bert.compute(predictions=[pred], references=[ref], lang='en')['f1'][0]
    return r, b, bf

sys_prompt = ("You are a professional medical assistant. Given an OCR-extracted "
              "pathology report, summarise it in markdown (### Histologic type ). "
              "Begin with ### Histologic type:")

def infer(llm, txt):
    t0 = time.time()
    out = llm.create_chat_completion(
        messages=[{'role':'system','content':sys_prompt},
                  {'role':'user','content':txt[:8000]}],
        max_tokens=1024, stop=['<|end_of_turn|>','</s>'], temperature=0)
    sec  = time.time() - t0
    toks = out['usage']['completion_tokens']
    return out['choices'][0]['message']['content'].strip(), sec, toks

# ---------- ???N ----------
rows, lim = [], args.limit if args.limit>0 else 10**9
with open(args.data, encoding='utf-8') as f:
    for _ in range(args.start-1): next(f, None)
    for i, line in enumerate(f, args.start):
        if i > lim: break
        j   = json.loads(line)
        txt = j.get('text') or next((m['content'] for m in j.get('messages',[]) if m.get('role')=='user'), '')
        for tag, llm in models.items():
            pred, sec, toks = infer(llm, txt)
            r, b, bf = metrics(pred, txt)
            rows.append({'id':i,'model':tag,'rougeL':r,'bleu':b,'bertscore':bf,
                         'sec':sec,'tok_out':toks,'tok_per_sec':toks/sec})
            print(f"[{tag}] {i:04d}  R={r:.3f}  B={b:.3f}  F1={bf:.3f}  {sec:.1f}s")

# ---------- ???X ----------
pd.DataFrame(rows).to_csv('eval_results.csv', index=False)
(pd.DataFrame(rows)
 .groupby('model').agg(N=('id','count'),
                       rougeL_mean=('rougeL','mean'),
                       bleu_mean=('bleu','mean'),
                       bert_f1_mean=('bertscore','mean'),
                       tok_per_sec_avg=('tok_per_sec','mean'))
 .reset_index()
).to_csv('eval_summary.csv', index=False)
