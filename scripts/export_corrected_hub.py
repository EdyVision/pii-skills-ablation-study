"""Build corrected Hub records locally. No network calls or publication."""
from pathlib import Path
import argparse,sys,json,collections,hashlib
import pyarrow as pa
import pyarrow.parquet as pq
p=argparse.ArgumentParser();p.add_argument('--repo-root',type=Path,required=True);p.add_argument('--hub-snapshot',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
sys.path.insert(0,str(args.repo_root/'lib'))
from ablation_harness.scoring.saved_analysis import load_saved_run,norm,type_metrics,f1
args.out.mkdir(parents=True,exist_ok=True);summary={};features=None
for run in ['main','pilot','baselines','detector','fp16','scaling']:
 source=next((args.hub_snapshot/'data').glob(run+'_v1-*.parquet'));old=pq.read_table(source).to_pylist();local=load_saved_run(args.repo_root,run);key=lambda r:(r['sample_id'],r['model'],r['condition']);bykey={key(r):r for r in local}
 assert len(bykey)==len(old) and set(bykey)=={key(r)for r in old}
 def redact(items):return [{**x,**({'text':'[REDACTED]'} if 'text'in x else {})}for x in items]
 def signature(items):return collections.Counter((norm(x['type']),x.get('start'),x.get('end'))for x in items)
 rows=[];changed=0;redacted=0
 for row in old:
  ref=bykey[key(row)];pred=json.loads(row['predictions']);gold=ref['ground_truth'];assert signature(pred)==signature(ref['predictions']),f'Unexplained prediction difference: {run}'
  metrics=type_metrics(pred,gold);mixed=f1(pred,gold,norm,True);before=json.loads(row['scores']or'null');changed+=(before!=metrics)
  redacted+=sum(bool(x.get('text')) and 'redacted'not in x['text'].lower() for x in pred)
  rows.append({**row,'predictions':json.dumps(redact(pred)),'scores':json.dumps(metrics),'ground_truth':json.dumps(redact(gold)),'mixed_span_type_f1':mixed,'scoring_version':'report_v0.3_aligned_type_and_mixed','gold_provenance':'primary_gold_join_by_sample_id' if run=='detector' else 'archived_local_scoring_gold'})
 table=pa.Table.from_pylist(rows);dest=args.out/'data'/source.name;dest.parent.mkdir(exist_ok=True);pq.write_table(table,dest)
 summary[run]={'rows':len(rows),'changed_score_records':changed,'redacted_entity_text_fields':redacted,'mean_type_f1':sum(json.loads(x['scores'])['f1']for x in rows)/len(rows),'mean_mixed_span_type_f1':sum(x['mixed_span_type_f1']for x in rows)/len(rows),'parquet_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
 for row in pq.read_table(dest).to_pylist():
  for field in ['predictions','ground_truth']:
   assert all(not x.get('text') or x['text']=='[REDACTED]'for x in json.loads(row[field]))
 if run=='main':
  expected={'zero_shot':(.6308502271,.3246421964),'with_docs':(.6015646037,.3006421902),'with_tools':(.5272177146,.4552550592),'with_skills':(.5105172575,.4478683417)}
  for c,(t,m) in expected.items():
   subset=[x for x in rows if x['condition']==c];assert abs(sum(json.loads(x['scores'])['f1']for x in subset)/len(subset)-t)<1e-8;assert abs(sum(x['mixed_span_type_f1']for x in subset)/len(subset)-m)<1e-8
assert abs(summary['detector']['mean_type_f1']-.5021779950611417)<1e-10
assert abs(summary['detector']['mean_mixed_span_type_f1']-.4532139160701965)<1e-10
(args.out/'CORRECTIONS.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
