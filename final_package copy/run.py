import os
import argparse
import importlib.machinery

def set_env_from_args(args):
    os.environ['RUN_ID'] = args.run_id
    os.environ['OUTDIR'] = args.outdir
    os.environ['DEVICE'] = args.device
    os.environ['RAG_TOPK'] = str(args.rag_topk)
    os.environ['MAX_NEW_TOKENS'] = str(args.max_new_tokens)
    os.environ['NUM_BEAMS'] = str(args.beams)
    os.environ['LENGTH_PENALTY'] = str(args.length_penalty)
    if args.mode == 'eval':
        os.environ['SKIP_TRAIN'] = '1'
    else:
        os.environ['SKIP_TRAIN'] = '0'
        os.environ['N_TRAIN'] = str(args.n_train)
        os.environ['N_VAL'] = str(args.n_val)
        os.environ['N_TEST'] = str(args.n_test)
        os.environ['MAX_STEPS'] = str(args.max_steps)
        os.environ['WARMUP_STEPS'] = str(args.warmup_steps)
        os.environ['GRAD_ACC'] = str(args.grad_acc)
        os.environ['UNFREEZE_LAST'] = str(args.unfreeze_last)
        os.environ['PARTIAL_UNFREEZE'] = '1'
        os.environ['USE_LORA'] = '1'
        os.environ['LORA_R'] = str(args.lora_r)
        os.environ['LORA_ALPHA'] = str(args.lora_alpha)
        os.environ['SAVE_STEPS'] = str(args.save_steps)

def main():
    ap = argparse.ArgumentParser(description='Final package runner')
    ap.add_argument('--mode', choices=['train','eval'], required=True)
    ap.add_argument('--run_id', required=True)
    ap.add_argument('--outdir', default='outputs/t5-small-lora-longmax')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--rag_topk', type=int, default=4)
    ap.add_argument('--max_new_tokens', type=int, default=160)
    ap.add_argument('--beams', type=int, default=2)
    ap.add_argument('--length_penalty', type=float, default=1.0)
    # training
    ap.add_argument('--n_train', type=int, default=1600)
    ap.add_argument('--n_val', type=int, default=320)
    ap.add_argument('--n_test', type=int, default=320)
    ap.add_argument('--max_steps', type=int, default=1600)
    ap.add_argument('--warmup_steps', type=int, default=250)
    ap.add_argument('--grad_acc', type=int, default=2)
    ap.add_argument('--unfreeze_last', type=int, default=4)
    ap.add_argument('--lora_r', type=int, default=16)
    ap.add_argument('--lora_alpha', type=int, default=32)
    ap.add_argument('--save_steps', type=int, default=400)
    args = ap.parse_args()

    set_env_from_args(args)
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(pkg_dir,'pipeline','run_stage2.py')
    loader = importlib.machinery.SourceFileLoader('run_stage2', p)
    mod = loader.load_module()
    mod.main()

    rid = os.environ.get('RUN_ID')
    out_dir = os.path.join(pkg_dir, 'outputs')
    sum_name = os.path.join(out_dir,'stage3_summary.json') if not rid else os.path.join(out_dir,f'stage3_summary_{rid}.json')
    cases_name = os.path.join(out_dir,'stage3_cases.jsonl') if not rid else os.path.join(out_dir,f'stage3_cases_{rid}.jsonl')
    print('summary_path', sum_name)
    print('cases_path', cases_name)

if __name__ == '__main__':
    main()
