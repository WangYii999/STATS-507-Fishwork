import os
import json
from pathlib import Path
import shutil
import glob
import matplotlib.pyplot as plt

ASSETS_DIR = Path('final_package copy/report_images')

def ensure_dir():
    ASSETS_DIR.mkdir(exist_ok=True)

def read_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def save_fig(name):
    p = ASSETS_DIR / name
    plt.tight_layout()
    plt.savefig(p, dpi=180)
    plt.close()
    return str(p)

def _find_latest(patterns):
    cands = []
    for p in patterns:
        cands.extend(glob.glob(p, recursive=True))
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

def plot_t5_compare():
    p2 = _find_latest(['final_package copy/outputs/stage3_summary_beams2*.json'])
    p4 = _find_latest(['final_package copy/outputs/stage3_summary_beams4pp*.json'])
    d2 = read_json(p2) or {}
    d4 = read_json(p4) or {}
    m2 = d2.get('t5_small', {})
    m4 = d4.get('t5_small', {})
    metrics = ['rougeL','persona_coverage','jaccard','bleu']
    vals2 = [float(m2.get(k,0)) for k in metrics]
    vals4 = [float(m4.get(k,0)) for k in metrics]
    x = range(len(metrics))
    width = 0.38
    plt.figure(figsize=(6,3.2))
    plt.bar([i-width/2 for i in x], vals2, width, label='beam=2')
    plt.bar([i+width/2 for i in x], vals4, width, label='beam=4')
    plt.xticks(list(x), ['ROUGE-L','Coverage','Jaccard','BLEU'])
    plt.ylabel('Score')
    plt.title('T5-small: beam2 vs beam4')
    plt.legend()
    return save_fig('t5_beam_compare.png')

def plot_t5_beam_triple():
    p1 = _find_latest(['final_package copy/outputs/stage3_summary_beams1*.json'])
    p2 = _find_latest(['final_package copy/outputs/stage3_summary_beams2*.json'])
    p4 = _find_latest(['final_package copy/outputs/stage3_summary_beams4pp*.json'])
    if not (p2 and p4):
        return None
    d1 = read_json(p1) or {}
    d2 = read_json(p2) or {}
    d4 = read_json(p4) or {}
    m1 = d1.get('t5_small', {})
    m2 = d2.get('t5_small', {})
    m4 = d4.get('t5_small', {})
    metrics = ['rougeL','persona_coverage','jaccard','bleu']
    x = range(len(metrics))
    width = 0.24
    plt.figure(figsize=(7.2,3.4))
    vals1 = [float(m1.get(k,0)) for k in metrics]
    vals2 = [float(m2.get(k,0)) for k in metrics]
    vals4 = [float(m4.get(k,0)) for k in metrics]
    plt.bar([i-width for i in x], vals1, width, label='beam=1') if p1 else None
    plt.bar(list(x), vals2, width, label='beam=2')
    plt.bar([i+width for i in x], vals4, width, label='beam=4')
    plt.xticks(list(x), ['ROUGE-L','Coverage','Jaccard','BLEU'])
    plt.ylabel('Score')
    plt.title('T5-small: beam1 vs beam2 vs beam4')
    plt.legend()
    return save_fig('t5_beam_1_2_4.png')

def plot_len_compare():
    pl = _find_latest(['final_package copy/outputs/stage3_summary_longmax*.json'])
    pm = _find_latest(['final_package copy/outputs/stage3_summary_max96*.json'])
    if not (pl and pm):
        return None
    dl = read_json(pl) or {}
    dm = read_json(pm) or {}
    ml = dl.get('t5_small', {})
    mm = dm.get('t5_small', {})
    metrics = ['rougeL','persona_coverage','jaccard','bleu']
    x = range(len(metrics))
    width = 0.38
    plt.figure(figsize=(6.4,3.2))
    vl = [float(ml.get(k,0)) for k in metrics]
    vm = [float(mm.get(k,0)) for k in metrics]
    plt.bar([i-width/2 for i in x], vl, width, label='MAX_NEW_TOKENS=48')
    plt.bar([i+width/2 for i in x], vm, width, label='MAX_NEW_TOKENS=96')
    plt.xticks(list(x), ['ROUGE-L','Coverage','Jaccard','BLEU'])
    plt.ylabel('Score')
    plt.title('T5-small: length comparison')
    plt.legend()
    return save_fig('t5_len_compare.png')

def plot_baselines():
    b = read_json('final_package copy/outputs/baseline.json') or {}
    d_long = read_json(_find_latest(['final_package copy/outputs/stage3_summary_longmax*.json'])) or {}
    ret = b.get('retrieval', {})
    tmp = b.get('template', {})
    t5 = d_long.get('t5_small', {})
    labels = ['Retrieval','Template','T5-small (beam=2)']
    metrics = ['rougeL','persona_coverage','jaccard']
    plt.figure(figsize=(6.4,3.8))
    for idx,k in enumerate(metrics):
        vals = [float(ret.get(k,0)), float(tmp.get(k,0)), float(t5.get(k,0))]
        plt.subplot(2,2,idx+1)
        plt.bar(labels, vals, color=['#4e79a7','#f28e2b','#59a14f'])
        plt.xticks(rotation=15)
        plt.ylabel(k)
        plt.title(k)
    plt.subplot(2,2,4)
    vals_bleu = [float(ret.get('bleu',0)), float(tmp.get('bleu',0)), float(t5.get('bleu',0))]
    plt.bar(labels, vals_bleu, color=['#4e79a7','#f28e2b','#59a14f'])
    plt.xticks(rotation=15)
    plt.ylabel('bleu')
    plt.title('bleu')
    return save_fig('baselines_compare.png')

def plot_training_curve():
    latest = _find_latest(['final_package copy/outputs/**/trainer_state.json'])
    d = read_json(latest) or {}
    logs = d.get('log_history', [])
    steps = [e.get('step') for e in logs if 'step' in e and 'loss' in e]
    loss = [e.get('loss') for e in logs if 'step' in e and 'loss' in e]
    gsteps = [e.get('step') for e in logs if 'step' in e and 'grad_norm' in e]
    gnorm = [e.get('grad_norm') for e in logs if 'step' in e and 'grad_norm' in e]
    plt.figure(figsize=(6.4,3.6))
    if steps and loss:
        plt.plot(steps, loss, label='loss', color='#4e79a7')
    else:
        plt.text(0.5, 0.5, 'no training logs', ha='center', va='center', transform=plt.gca().transAxes)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('Training loss over steps')
    save_fig('train_loss.png')
    plt.figure(figsize=(6.4,3.6))
    if gsteps and gnorm:
        plt.plot(gsteps, gnorm, label='grad_norm', color='#e15759')
    else:
        plt.text(0.5, 0.5, 'no training logs', ha='center', va='center', transform=plt.gca().transAxes)
    plt.xlabel('step')
    plt.ylabel('grad_norm')
    plt.title('Gradient norm over steps')
    return save_fig('train_gradnorm.png')

def copy_existing_images():
    copied = []
    for fn in ['model_arch.png','model_arch_cn.png','eval_summary.png']:
        src = Path('final_package copy/outputs')/fn
        if src.exists():
            dst = ASSETS_DIR/src.name
            shutil.copy2(src, dst)
            copied.append(str(dst))
    # Only operate within this package copy; skip external notebooks
    return copied

def main():
    ASSETS_DIR.mkdir(exist_ok=True)
    out1 = plot_t5_compare()
    out1b = plot_t5_beam_triple()
    out2 = plot_baselines()
    out3 = plot_training_curve()
    out4 = plot_len_compare()
    copied = copy_existing_images()
    manifest = {
        't5_beam_compare': out1,
        't5_beam_1_2_4': out1b,
        'baselines_compare': out2,
        'train_gradnorm': out3,
        't5_len_compare': out4,
        'copied': copied
    }
    with open(ASSETS_DIR/'manifest.json','w',encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
