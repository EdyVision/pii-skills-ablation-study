"""Recompute saved diagnostics offline; no inference, raw-run writes, or browser launch."""
from pathlib import Path
import os,sys,json,ast,io,contextlib,traceback,hashlib
root=Path(__file__).resolve().parents[1]
os.chdir(root/'notebooks');sys.path.insert(0,str(root/'lib'))
import pandas as pd
import plotly.graph_objects as go
path=root/'notebooks/04_analyses_r&r.ipynb'; nb=json.loads(path.read_text())
outputs=[]
def display(value):
 data={'text/plain':str(value)}
 if isinstance(value,pd.DataFrame):data['text/html']=value.to_html(index=False)
 outputs.append({'output_type':'display_data','data':data,'metadata':{}})
def show(fig,*args,**kwargs):
 outputs.append({'output_type':'display_data','data':{'application/vnd.plotly.v1+json':json.loads(fig.to_json()),'text/plain':'Corrected interactive figure; offline PDF export uses Matplotlib.'},'metadata':{}})
go.Figure.show=show
ns={'__name__':'__main__','display':display}
count=0
for i,cell in enumerate(nb['cells']):
 if cell['cell_type']!='code':continue
 count+=1; outputs=[]; buf=io.StringIO(); tree=ast.parse(''.join(cell['source']))
 try:
  with contextlib.redirect_stdout(buf):
   if tree.body and isinstance(tree.body[-1],ast.Expr):
    last=tree.body.pop();exec(compile(tree,f'cell-{i}','exec'),ns)
    val=eval(compile(ast.Expression(last.value),f'cell-{i}','eval'),ns)
    if val is not None:display(val)
   else:exec(compile(tree,f'cell-{i}','exec'),ns)
 except Exception:
  print('FAILED CELL',i,buf.getvalue());traceback.print_exc();raise
 if buf.getvalue():outputs.insert(0,{'output_type':'stream','name':'stdout','text':buf.getvalue()})
 cell['outputs']=outputs;cell['execution_count']=count
 print('Executed cell',i,flush=True)
path.write_text(json.dumps(nb,indent=2,ensure_ascii=False)+'\n')
out=root/'analysis-results';out.mkdir(exist_ok=True)
for key in ['df_dual','df_over','df_funnel','df_sig','df_strict','df_mapsens','df_permodel']:
 ns[key].to_csv(out/(key+'.csv'),index=False)
source=root
manifest={run:hashlib.sha256((source/'results/past_runs'/run/'experiment_results.json').read_bytes()).hexdigest() for run in ['main','baselines','detector','fp16','scaling']}
(out/'raw-run-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(ns['df_dual'].to_string(index=False));print(ns['df_funnel'].to_string(index=False))
# Additional correction tables, calculated from saved runs only.
from ablation_harness.scoring.diagnostics import sample_cluster_ci
main_ix={(r['model'],r['condition'],r['sample_id']):r for r in ns['main']}
precision=[]; all_observations={False:[],True:[]}
for model in sorted({r['model'] for r in ns['fp16']}):
 for span in [False,True]:
  observations=[];old=[];new=[]
  for row in ns['fp16']:
   if row['model']!=model:continue
   ref=main_ix[(model,row['condition'],row['sample_id'])]
   assert row['ground_truth']==ref['ground_truth'], 'Gold differs within a precision pair'
   a=ns['f1'](row['predictions'] or [],row['ground_truth'] or [],ns['norm'],span)
   b=ns['f1'](ref['predictions'] or [],ref['ground_truth'] or [],ns['norm'],span)
   observations.append((row['sample_id'],row['condition'],a-b))
   all_observations[span].append((row['sample_id'],(model,row['condition']),a-b))
   old.append(b);new.append(a)
  summary=sample_cluster_ci(observations)
  precision.append({'model':model,'metric':'span' if span else 'type','4bit':sum(old)/len(old),'16bit':sum(new)/len(new),**summary})
pd.DataFrame(precision).to_csv(out/'precision_clustered.csv',index=False)
pd.DataFrame([{'metric':'span' if sp else 'type',**sample_cluster_ci(obs)} for sp,obs in all_observations.items()]).to_csv(out/'precision_pooled.csv',index=False)
# Same sample sets across conditions: required for paired comparisons and equal-model means.
from collections import Counter
counts=Counter((r['model'],r['condition']) for r in ns['main'])
assert len(counts)==16 and set(counts.values())=={2000}
for model in sorted({r['model'] for r in ns['main']}):
 sets=[{r['sample_id'] for r in ns['main'] if r['model']==model and r['condition']==c} for c in ns['CONDITIONS']]
 assert all(x==sets[0] for x in sets)
expected={'type':[.631,.602,.527,.511],'span':[.325,.301,.455,.448]}
for metric,values in expected.items():
 row=ns['df_dual'].query('metric == @metric and align == "after"').iloc[0]
 assert list(row[['Zero-shot','+Docs','+Tool','+Skills']])==values
(out/'verification.txt').write_text('PASS: 32,000 primary rows; 2,000 matched examples in each of 16 cells.\nPASS: original rounded headline F1 values reproduced exactly.\nPASS: precision comparisons pair identical gold annotations.\nNo new inference, detector execution, or raw-run writes.\n')
# Save figure data separately from the notebook for browser-free vector PDF export.
figures=[]
for cell in nb['cells']:
 for output in cell.get('outputs',[]):
  figure=output.get('data',{}).get('application/vnd.plotly.v1+json')
  if figure:figures.append(figure)
(out/'figure-data.json').write_text(json.dumps(figures))
scale=[]
for label,recs,model in [('Qwen 7B',ns['main'],'qwen2_7b'),('Qwen 14B',ns['scaling'],'qwen2_14b')]:
 scale.append({'model':label,**{ns['CLABEL'][c]:ns['mean_f1']([r for r in recs if r['model']==model],c,ns['norm'],False) for c in ns['CONDITIONS']}})
pd.DataFrame(scale).to_csv(out/'scaling_within_family.csv',index=False)
configs=[]
for label,run,condition in [('Zero-shot','main','zero_shot'),('Few-shot','baselines','few_shot'),('CoT','baselines','cot'),('+Docs','main','with_docs'),('+Tool','main','with_tools'),('+Skills','main','with_skills'),('Detector','det','detector')]:
 configs.append({'label':label,**{metric:ns['mean_f1'](ns[run],condition,ns['norm'],span) for metric,span in [('type',False),('span',True)]}})
pd.DataFrame(configs).to_csv(out/'configuration_means.csv',index=False)
