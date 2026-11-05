import _paths  # noqa: F401  (repository root on path, run from root)
import json, pickle, os, traceback
from foucal import train
from pathlib import Path
OUT = Path('results/results.jsonl')
METHODS = ['FOUCAL', 'FOUCAL-noMondrian', 'IT2-raw', 'IT2-split', 'IT2-MSE', 'T1-TSK', 'Ens-MLP', 'HGB-CQR', 'LSTM-CQR']
done = set()
if OUT.exists():
    for l in open(OUT): r = json.loads(l); done.add((r['data'], r['method'], r['seed']))
for s in range(int(os.environ.get("NSEED", 5))):
    for fd in ['FD001', 'FD002', 'FD003', 'FD004']:
        for m in METHODS:
            if (fd, m, s) in done: continue
            try: r = train.run(fd, m, s)
            except Exception: traceback.print_exc(); continue
            if m == 'FOUCAL' and s == 0: pickle.dump(r['_p'], open(f'checkpoints/foucal_{fd}_s0.pkl', 'wb'))
            r = {k: v for k, v in r.items() if not k.startswith('_')}; r.update(data=fd, method=m, seed=s)
            open(OUT, 'a').write(json.dumps(r, default=float) + '\n'); print(fd, m, s, round(r['RMSE'], 2), round(r['cov'], 3), round(r['winkler'], 1), flush=True)
