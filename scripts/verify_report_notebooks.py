from pathlib import Path
import json,os,sys,ast,contextlib,io,traceback,socket
root=Path(__file__).resolve().parents[1];os.chdir(root/'notebooks');sys.path.insert(0,str(root/'lib'))
socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('Network disabled during offline verification'))
import pandas as pd
import plotly.graph_objects as go
results=[]
for filename in ['03_analyses_main.ipynb','03_analyses_pilot.ipynb','04_analyses_r&r.ipynb']:
 path=root/'notebooks'/filename;nb=json.loads(path.read_text());ns={'__name__':'__main__'};figures=[]
 def display(obj):
  if isinstance(obj,pd.DataFrame):out.append({'output_type':'display_data','data':{'text/plain':obj.to_string(index=False)},'metadata':{}})
 def show(fig,*a,**k):
  figures.append(json.loads(fig.to_json()))
  out.append({'output_type':'display_data','data':{'application/vnd.plotly.v1+json':figures[-1]},'metadata':{}})
 go.Figure.show=show
 go.Figure.write_image=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('Browser rendering disabled'))
 ns['display']=display;failed=[];count=0
 for i,c in enumerate(nb['cells']):
  if c['cell_type']!='code':continue
  count+=1;out=[];buf=io.StringIO()
  try:
   tree=ast.parse(''.join(c['source']))
   with contextlib.redirect_stdout(buf):
    if tree.body and isinstance(tree.body[-1],ast.Expr):
     last=tree.body.pop();exec(compile(tree,f'{filename}:{i}','exec'),ns)
     v=eval(compile(ast.Expression(last.value),f'{filename}:{i}','eval'),ns)
     if v is not None:display(v)
    else:exec(compile(tree,f'{filename}:{i}','exec'),ns)
  except Exception as e:
   failed.append({'cell':i,'error':repr(e)});print(filename,i,'FAILED',repr(e),flush=True);traceback.print_exc();break
  if buf.getvalue():out.insert(0,{'output_type':'stream','name':'stdout','text':buf.getvalue()})
  c['outputs']=out;c['execution_count']=count
  print(filename,i,'OK',flush=True)
 if not failed:
  path.write_text(json.dumps(nb,indent=2)+'\n');(root/'analysis-results').mkdir(exist_ok=True)
  (root/'analysis-results'/(filename+'.figures.json')).write_text(json.dumps(figures))
  if filename.startswith('04'):
   for k,v in ns.items():
    if k.startswith('df_') and isinstance(v,pd.DataFrame):v.to_csv(root/'analysis-results'/(k+'.csv'),index=False)
 results.append({'notebook':filename,'executed_cells':count,'failures':failed,'figures':len(figures)})
(root/'analysis-results/notebook-execution.json').write_text(json.dumps(results,indent=2));print(json.dumps(results),flush=True)
