import os
PKG_DIR = os.path.dirname(os.path.dirname(__file__))
HF_BASE = os.environ.get('HF_HOME', os.path.join(PKG_DIR, 'hf_cache'))
OUT_DIR = os.path.join(PKG_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)
os.environ['HF_HOME'] = HF_BASE
os.environ['HF_HUB_CACHE'] = os.environ.get('HF_HUB_CACHE', os.path.join(HF_BASE, 'hub'))
os.environ['TRANSFORMERS_CACHE'] = os.environ.get('TRANSFORMERS_CACHE', os.path.join(HF_BASE, 'transformers'))
os.environ['HF_DATASETS_CACHE'] = os.environ.get('HF_DATASETS_CACHE', os.path.join(HF_BASE, 'datasets'))
os.environ['DATASETS_CACHE'] = os.environ.get('DATASETS_CACHE', os.path.join(HF_BASE, 'datasets'))
os.environ['TORCH_HOME'] = os.environ.get('TORCH_HOME', os.path.join(HF_BASE, 'torch'))
#
for _p in [os.environ['HF_HOME'], os.environ['HF_HUB_CACHE'], os.environ['TRANSFORMERS_CACHE'], os.environ['HF_DATASETS_CACHE'], os.environ['TORCH_HOME']]:
    os.makedirs(_p, exist_ok=True)
import sys
import json
import re
import random
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#

def get_device():
    import os, torch
    d = os.environ.get('DEVICE')
    if d == 'cpu':
        return 'cpu'
    if d == 'cuda':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if torch.cuda.is_available() else 'cpu'
def set_seed(s):
    random.seed(s)
    np.random.seed(s)

def concat_persona(e):
    return (e.get('user 1 personas','') + '\n' + e.get('user 2 personas','')).strip()

def get_label(e):
    return e.get('Best Generated Conversation','').strip()

def split_dataset(ds, seed=42):
    a = ds.train_test_split(test_size=0.2, seed=seed)
    b = a['test'].train_test_split(test_size=0.5, seed=seed)
    return a['train'], b['train'], b['test']

def build_small_splits(train, val, test, n_train=1000, n_val=200, n_test=200):
    nt = min(n_train, len(train))
    nv = min(n_val, len(val))
    ns = min(n_test, len(test))
    return train.shuffle(seed=42).select(range(nt)), val.shuffle(seed=42).select(range(nv)), test.shuffle(seed=42).select(range(ns))

def filter_nonempty_labels(ds):
    return ds.filter(lambda e: isinstance(e.get('Best Generated Conversation',''), str) and len(e.get('Best Generated Conversation','').strip()) > 0)

def determine_max_lengths(train, val, tok, p_src=85, p_tgt=90, sample=200):
    import numpy as _np
    src_lens, tgt_lens = [], []
    def _collect(d):
        n = min(len(d), sample)
        for i in range(n):
            e = d[i]
            inp = (e.get('user 1 personas','') + '\n' + e.get('user 2 personas','')).strip()
            lab = e.get('Best Generated Conversation','').strip()
            src_lens.append(len(tok(inp).input_ids))
            tgt_lens.append(len(tok(lab).input_ids))
    _collect(train)
    _collect(val)
    max_src = int(_np.percentile(src_lens if src_lens else [256], p_src))
    max_tgt = int(_np.percentile(tgt_lens if tgt_lens else [256], p_tgt))
    return max_src, max_tgt

def build_rag_knowledge(train, val, k=3):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    tr_inputs = [concat_persona(x) for x in train]
    tr_labels = [get_label(x) for x in train]
    va_inputs = [concat_persona(x) for x in val]
    vec = TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=2)
    X = vec.fit_transform(tr_inputs)
    nn_tr = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    nn_tr.fit(X)
    # train-side neighbors excluding self
    idx_tr = nn_tr.kneighbors(X, return_distance=False)
    ext_train = []
    for i, row in enumerate(idx_tr):
        cand = [j for j in row if j != i][:k]
        ext_train.append(" || ".join([tr_labels[j] for j in cand]))
    # val-side neighbors
    Qv = vec.transform(va_inputs)
    idx_val = nn_tr.kneighbors(Qv, n_neighbors=k, return_distance=False)
    ext_val = [" || ".join([tr_labels[j] for j in row]) for row in idx_val]
    return ext_train, ext_val

def baseline_retrieval(train, val):
    train_inputs = [concat_persona(x) for x in train]
    train_labels = [get_label(x) for x in train]
    val_inputs = [concat_persona(x) for x in val]
    vec = TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=2)
    X = vec.fit_transform(train_inputs)
    Q = vec.transform(val_inputs)
    sims = cosine_similarity(Q, X)
    idx = sims.argmax(axis=1)
    preds = [train_labels[i] for i in idx]
    refs = [[get_label(x)] for x in val]
    import evaluate
    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')
    m_bleu = bleu.compute(predictions=preds, references=refs)
    m_rouge = rouge.compute(predictions=preds, references=[r[0] for r in refs])
    return {'bleu': m_bleu.get('bleu',0.0), 'rougeL': m_rouge.get('rougeL',0.0)}

def template_baseline(val):
    preds = [concat_persona(x) for x in val]
    refs = [[get_label(x)] for x in val]
    try:
        import evaluate
        bleu = evaluate.load('bleu')
        rouge = evaluate.load('rouge')
        m_bleu = bleu.compute(predictions=preds, references=refs)
        m_rouge = rouge.compute(predictions=preds, references=[r[0] for r in refs])
        return {'bleu': m_bleu.get('bleu',0.0), 'rougeL': m_rouge.get('rougeL',0.0)}
    except Exception:
        refs_flat = [r[0] for r in refs]
        return {'bleu': _bleu_unigram(preds, refs_flat), 'rougeL': _rougeL(preds, refs_flat)}

def predict_retrieval(train, val):
    train_inputs = [concat_persona(x) for x in train]
    train_labels = [get_label(x) for x in train]
    val_inputs = [concat_persona(x) for x in val]
    vec = TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=2)
    X = vec.fit_transform(train_inputs)
    Q = vec.transform(val_inputs)
    sims = cosine_similarity(Q, X)
    idx = sims.argmax(axis=1)
    return [train_labels[i] for i in idx]

def predict_template(val):
    return [concat_persona(x) for x in val]

def predict_t5(val, model_name='google/flan-t5-small', max_new_tokens=64):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch, os
    tok = None
    model = None
    if os.path.isdir(model_name):
        tok = AutoTokenizer.from_pretrained(model_name)
        from peft import PeftModel
        base_model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')
        model = PeftModel.from_pretrained(base_model, model_name)
        print('loaded_adapter', model_name)
    else:
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print('loaded_base', model_name)
    device = get_device()
    beams = int(os.environ.get('NUM_BEAMS', '1'))
    lp = float(os.environ.get('LENGTH_PENALTY', '1.0'))
    model.to(device)
    model.eval()
    print('device', device)
    preds = []
    with torch.inference_mode():
        for e in val:
            inp = concat_persona(e)
            enc = tok(inp, return_tensors='pt', truncation=True, max_length=256).to(device)
            out = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=beams, length_penalty=lp)
            preds.append(tok.decode(out[0], skip_special_tokens=True))
    return preds

#

_tok_re = re.compile(r"[A-Za-z0-9\u4e00-\u9fa5]+")
def _tokens(s):
    if not isinstance(s, str):
        return []
    return _tok_re.findall(s.lower())

def persona_coverage(pred, e):
    p = set(_tokens(concat_persona(e)))
    if not p:
        return 0.0
    q = set(_tokens(pred))
    return len(p & q) / float(len(p))

def jaccard(a, b):
    A, B = set(_tokens(a)), set(_tokens(b))
    if not A and not B:
        return 0.0
    return len(A & B) / float(len(A | B))

def _bleu_unigram(preds, refs_flat):
    sc = []
    for p, r in zip(preds, refs_flat):
        P = _tokens(p)
        R = set(_tokens(r))
        if len(P) == 0:
            sc.append(0.0)
            continue
        overlap = sum(1 for t in P if t in R)
        sc.append(overlap / float(len(P)))
    return float(np.mean(sc)) if sc else 0.0

def _lcs_len(a, b):
    A = _tokens(a)
    B = _tokens(b)
    n, m = len(A), len(B)
    if n == 0 or m == 0:
        return 0
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        ai = A[i-1]
        for j in range(1, m+1):
            if ai == B[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = dp[i-1][j] if dp[i-1][j] >= dp[i][j-1] else dp[i][j-1]
    return dp[n][m]

def _rougeL(preds, refs_flat):
    sc = []
    for p, r in zip(preds, refs_flat):
        denom = max(1, len(_tokens(r)))
        sc.append(_lcs_len(p, r) / float(denom))
    return float(np.mean(sc)) if sc else 0.0

#

def run_stage3(base, tr_s, va_s, model_name='google/flan-t5-small', n_cases=20):
    refs = [[get_label(x)] for x in va_s]
    refs_flat = [r[0] for r in refs]
    preds_ret = predict_retrieval(tr_s, va_s)
    preds_tmp = predict_template(va_s)
    _max_gen = int(os.environ.get('MAX_NEW_TOKENS', '48'))
    preds_t5 = predict_t5(va_s, model_name=model_name, max_new_tokens=_max_gen)
    try:
        import evaluate
        bleu = evaluate.load('bleu')
        rouge = evaluate.load('rouge')
        m_ret_bleu = bleu.compute(predictions=preds_ret, references=refs)
        m_ret_rouge = rouge.compute(predictions=preds_ret, references=refs_flat)
        m_tmp_bleu = bleu.compute(predictions=preds_tmp, references=refs)
        m_tmp_rouge = rouge.compute(predictions=preds_tmp, references=refs_flat)
        m_t5_bleu = bleu.compute(predictions=preds_t5, references=refs)
        m_t5_rouge = rouge.compute(predictions=preds_t5, references=refs_flat)
    except Exception:
        m_ret_bleu = {'bleu': _bleu_unigram(preds_ret, refs_flat)}
        m_ret_rouge = {'rougeL': _rougeL(preds_ret, refs_flat)}
        m_tmp_bleu = {'bleu': _bleu_unigram(preds_tmp, refs_flat)}
        m_tmp_rouge = {'rougeL': _rougeL(preds_tmp, refs_flat)}
        m_t5_bleu = {'bleu': _bleu_unigram(preds_t5, refs_flat)}
        m_t5_rouge = {'rougeL': _rougeL(preds_t5, refs_flat)}
    cov_ret = np.mean([persona_coverage(p, e) for p, e in zip(preds_ret, va_s)])
    cov_tmp = np.mean([persona_coverage(p, e) for p, e in zip(preds_tmp, va_s)])
    cov_t5 = np.mean([persona_coverage(p, e) for p, e in zip(preds_t5, va_s)])
    jac_ret = np.mean([jaccard(p, r) for p, r in zip(preds_ret, refs_flat)])
    jac_tmp = np.mean([jaccard(p, r) for p, r in zip(preds_tmp, refs_flat)])
    jac_t5 = np.mean([jaccard(p, r) for p, r in zip(preds_t5, refs_flat)])
    summary = {
        'retrieval': {'bleu': float(m_ret_bleu.get('bleu',0.0)), 'rougeL': float(m_ret_rouge.get('rougeL',0.0)), 'persona_coverage': float(cov_ret), 'jaccard': float(jac_ret)},
        'template':  {'bleu': float(m_tmp_bleu.get('bleu',0.0)), 'rougeL': float(m_tmp_rouge.get('rougeL',0.0)), 'persona_coverage': float(cov_tmp), 'jaccard': float(jac_tmp)},
        't5_small':  {'bleu': float(m_t5_bleu.get('bleu',0.0)), 'rougeL': float(m_t5_rouge.get('rougeL',0.0)), 'persona_coverage': float(cov_t5), 'jaccard': float(jac_t5)}
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    _rid = os.environ.get('RUN_ID')
    _sum_path = os.path.join(OUT_DIR, 'stage3_summary.json') if not _rid else os.path.join(OUT_DIR, f'stage3_summary_{_rid}.json')
    with open(_sum_path,'w',encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('write', _sum_path)
    # cases
    k = min(n_cases, len(va_s))
    _cases_path = os.path.join(OUT_DIR, 'stage3_cases.jsonl') if not _rid else os.path.join(OUT_DIR, f'stage3_cases_{_rid}.jsonl')
    with open(_cases_path,'w',encoding='utf-8') as f:
        for i in range(k):
            e = va_s[i]
            line = {
                'persona1': e.get('user 1 personas',''),
                'persona2': e.get('user 2 personas',''),
                'reference': refs_flat[i],
                'pred_retrieval': preds_ret[i],
                'pred_template': preds_tmp[i],
                'pred_t5': preds_t5[i],
                'metrics': {
                    'retrieval': {'coverage': persona_coverage(preds_ret[i], e), 'jaccard': jaccard(preds_ret[i], refs_flat[i])},
                    'template': {'coverage': persona_coverage(preds_tmp[i], e), 'jaccard': jaccard(preds_tmp[i], refs_flat[i])},
                    't5_small': {'coverage': persona_coverage(preds_t5[i], e), 'jaccard': jaccard(preds_t5[i], refs_flat[i])}
                }
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print('write', _cases_path)
    return summary

def quick_eval(base, n_val=30, model_name='google/flan-t5-small'):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    val = base.train_test_split(test_size=0.2, seed=42)['test'].shuffle(seed=42).select(range(min(n_val, len(base))))
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = get_device()
    model.to(device)
    print('device', device)
    preds, refs = [], []
    for e in val:
        inp = (e.get('user 1 personas','') + '\n' + e.get('user 2 personas','')).strip()
        enc = tok(inp, return_tensors='pt', truncation=True, max_length=256).to(device)
        out = model.generate(**enc, max_new_tokens=64)
        preds.append(tok.decode(out[0], skip_special_tokens=True))
        refs.append([e.get('Best Generated Conversation','').strip()])
    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')
    m_bleu = bleu.compute(predictions=preds, references=refs)
    m_rouge = rouge.compute(predictions=preds, references=[r[0] for r in refs])
    return {'bleu': float(m_bleu.get('bleu',0.0)), 'rougeL': float(m_rouge.get('rougeL',0.0))}

def train_t5(train, val, outdir='outputs/t5-small-lora', model_name='google/flan-t5-small'):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
    import torch
    set_seed(42)
    os.makedirs(outdir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    max_src_len, max_tgt_len = determine_max_lengths(train, val, tok)
    pad_id = tok.pad_token_id
    # build RAG knowledge and attach as a column
    k = int(os.environ.get('RAG_TOPK', '3'))
    ext_train, ext_val = build_rag_knowledge(train, val, k=k)
    train = train.add_column('external_knowledge', ext_train)
    val = val.add_column('external_knowledge', ext_val)
    def preprocess(batch):
        u1s = batch['user 1 personas']
        u2s = batch['user 2 personas']
        utext = [((u1 or '') + '\n' + (u2 or '')).strip() for u1, u2 in zip(u1s, u2s)]
        ek = batch.get('external_knowledge', ["" for _ in utext])
        inputs = []
        for ut, ek_i in zip(utext, ek):
            parts = ut.split('\n')
            p1 = parts[0] if parts else ''
            p2 = parts[1] if len(parts) > 1 else ''
            inputs.append(f"persona-dialogue: <USER> {p1} <SYSTEM> {p2} <EXTERNAL KNOWLEDGE> {ek_i}")
        labels = [s.strip() for s in batch['Best Generated Conversation']]
        m = tok(inputs, max_length=max_src_len, truncation=True)
        y = tok(text_target=labels, max_length=max_tgt_len, truncation=True)
        lbl = y['input_ids']
        m['labels'] = [[t if t != pad_id else -100 for t in seq] for seq in lbl]
        return m
    train_proc = train.map(preprocess, batched=True, remove_columns=train.column_names, load_from_cache_file=False)
    val_proc = val.map(preprocess, batched=True, remove_columns=val.column_names, load_from_cache_file=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    use_lora = False
    if os.environ.get('USE_LORA', '1') == '1':
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            r = int(os.environ.get('LORA_R', '8'))
            alpha = int(os.environ.get('LORA_ALPHA', '16'))
            dropout = float(os.environ.get('LORA_DROPOUT', '0.1'))
            targets = os.environ.get('LORA_TARGETS', 'q,v,k,o,wi,wo').split(',')
            modules_to_save = os.environ.get('MODULES_TO_SAVE', 'lm_head').split(',')
            cfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=targets, modules_to_save=modules_to_save)
            model = get_peft_model(model, cfg)
            use_lora = True
            if os.environ.get('PARTIAL_UNFREEZE', '0') == '1':
                import re
                k_last = int(os.environ.get('UNFREEZE_LAST', '4'))
                blk_idxs = set()
                for name, _ in model.named_parameters():
                    m = re.search(r"encoder\.block\.(\d+)\.", name)
                    if m:
                        blk_idxs.add(int(m.group(1)))
                if blk_idxs:
                    total = max(blk_idxs) + 1
                    cutoff = max(0, total - k_last)
                    for name, p in model.named_parameters():
                        m = re.search(r"encoder\.block\.(\d+)\.", name)
                        if m:
                            idx = int(m.group(1))
                            if idx >= cutoff:
                                p.requires_grad = True
        except Exception:
            pass
    if not use_lora:
        for n,p in model.named_parameters():
            if 'encoder' in n:
                p.requires_grad = False
    tn = sum(1 for p in model.parameters() if p.requires_grad)
    print('trainable_params', tn)
    device = get_device()
    model.to(device)
    print('device', device)
    dc = DataCollatorForSeq2Seq(tok, model=model)
    _max_steps = int(os.environ.get('MAX_STEPS', '500'))
    _lr = float(os.environ.get('LEARNING_RATE', '1e-4'))
    _warmup = int(os.environ.get('WARMUP_STEPS', '0'))
    _weight_decay = float(os.environ.get('WEIGHT_DECAY', '0.0'))
    _clip = float(os.environ.get('GRADIENT_CLIP', '1.0'))
    _save_steps = int(os.environ.get('SAVE_STEPS', '100'))
    _sched = os.environ.get('LR_SCHEDULER', 'constant')
    _grad_acc = int(os.environ.get('GRAD_ACC', '1'))
    args = Seq2SeqTrainingArguments(
        output_dir=outdir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=_grad_acc,
        auto_find_batch_size=True,
        learning_rate=_lr,
        num_train_epochs=1,
        max_steps=_max_steps,
        logging_steps=10,
        save_steps=_save_steps,
        save_total_limit=2,
        save_safetensors=True,
        fp16=False,
        gradient_checkpointing=False,
        report_to=[],
        predict_with_generate=True,
        warmup_steps=_warmup,
        weight_decay=_weight_decay,
        lr_scheduler_type=_sched,
        max_grad_norm=_clip,
        label_smoothing_factor=float(os.environ.get('LABEL_SMOOTHING','0.0'))
    )
    import evaluate
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.where(preds!=-100, preds, tok.pad_token_id)
        decoded_preds = tok.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels!=-100, labels, tok.pad_token_id)
        decoded_labels = tok.batch_decode(labels, skip_special_tokens=True)
        m_rouge = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        m_bleu = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
        return {'rougeL': m_rouge.get('rougeL',0.0), 'bleu': m_bleu.get('bleu',0.0)}
    tr = Seq2SeqTrainer(model=model, args=args, train_dataset=train_proc, eval_dataset=val_proc, tokenizer=tok, data_collator=dc, compute_metrics=compute_metrics)
    print('start_train')
    tr.train()
    print('saving_model')
    tr.save_model(outdir)
    tok.save_pretrained(outdir)
    print('start_eval')
    m = tr.evaluate()
    res = {'bleu': float(m.get('eval_bleu',0.0)), 'rougeL': float(m.get('eval_rougeL',0.0))}
    print('eval_metrics', res)
    return res

def main():
    set_seed(42)
    print('load_dataset')
    print('env_RUN_ID', os.environ.get('RUN_ID',''))
    print('env_OUTDIR', os.environ.get('OUTDIR',''))
    print('env_DEVICE', os.environ.get('DEVICE',''))
    ds_all = load_dataset('google/Synthetic-Persona-Chat')
    base = ds_all[list(ds_all.keys())[0]]
    tr, va, te = split_dataset(base, seed=42)
    tr = filter_nonempty_labels(tr)
    va = filter_nonempty_labels(va)
    _n_train = int(os.environ.get('N_TRAIN', '300'))
    _n_val = int(os.environ.get('N_VAL', '60'))
    _n_test = int(os.environ.get('N_TEST', '60'))
    tr_s, va_s, te_s = build_small_splits(tr, va, te, n_train=_n_train, n_val=_n_val, n_test=_n_test)
    os.makedirs(OUT_DIR, exist_ok=True)
    print('compute_baselines')
    b_ret = baseline_retrieval(tr_s, va_s)
    b_tmp = template_baseline(va_s)
    with open(os.path.join(OUT_DIR,'baseline.json'),'w',encoding='utf-8') as f:
        json.dump({'retrieval': b_ret, 'template': b_tmp}, f, ensure_ascii=False, indent=2)
    outdir = os.environ.get('OUTDIR', os.path.join(OUT_DIR,'t5-small-lora'))
    print('outdir', outdir)
    skip_train = os.environ.get('SKIP_TRAIN','0') == '1'
    t5m = None
    if not skip_train:
        print('train_t5')
        try:
            t5m = train_t5(tr_s, va_s, outdir=outdir, model_name=os.environ.get('T5_MODEL_NAME','google/flan-t5-small'))
        except Exception as e:
            import traceback
            print('train_t5_error', str(e))
            print('train_t5_error_tb', traceback.format_exc())
            t5m = {'bleu': 0.0, 'rougeL': 0.0}
            _rid = os.environ.get('RUN_ID')
            _err_path = os.path.join(OUT_DIR,'train_error.txt') if not _rid else os.path.join(OUT_DIR, f'train_error_{_rid}.txt')
            with open(_err_path,'w',encoding='utf-8') as f:
                f.write(str(e) + "\n" + traceback.format_exc())
        _rid = os.environ.get('RUN_ID')
        _t5m_path = os.path.join(OUT_DIR,'t5_metrics.json') if not _rid else os.path.join(OUT_DIR, f't5_metrics_{_rid}.json')
        with open(_t5m_path,'w',encoding='utf-8') as f:
            json.dump(t5m, f, ensure_ascii=False, indent=2)
        print('write', _t5m_path)
    print('run_stage3')
    s3 = run_stage3(base, tr_s, va_s, model_name=outdir, n_cases=20)
    _rid_done = os.environ.get('RUN_ID')
    _done_path = os.path.join(OUT_DIR,'stage3_done.txt') if not _rid_done else os.path.join(OUT_DIR, f'stage3_done_{_rid_done}.txt')
    with open(_done_path,'w',encoding='utf-8') as f:
        f.write('ok')
    print('baseline', b_ret, b_tmp)
    if t5m is not None:
        print('t5', t5m)

if __name__ == '__main__':
    main()
