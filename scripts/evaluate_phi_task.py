#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, torch
from tqdm.auto import tqdm
from eventttt.io import read_task_samples
from eventttt.metrics import classification_metrics_nclass
from eventttt.task_phi import load_phi, score_task_sample

def main():
 p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--model-id',required=True); p.add_argument('--limit',type=int,default=None); p.add_argument('--seed',type=int,default=1729); a=p.parse_args()
 np.random.seed(a.seed); torch.manual_seed(a.seed)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
 samples=read_task_samples(a.manifest); samples=samples[:a.limit] if a.limit else samples
 model,processor=load_phi(a.model_id); device=next(model.parameters()).device
 rows=[score_task_sample(model,processor,s,device) for s in tqdm(samples,desc='Phi scoring',dynamic_ncols=True)]
 labels=samples[0].candidate_labels; metrics=classification_metrics_nclass([r['label_id'] for r in rows],np.asarray([r['probabilities'] for r in rows]),labels)
 out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
 with (out/'predictions.jsonl').open('w') as h:
  for r in rows: h.write(json.dumps(r)+'\n')
 print(json.dumps({'output_dir':str(out),'count':len(rows),'metrics':metrics},indent=2))
if __name__=='__main__': main()
