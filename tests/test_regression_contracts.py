"""Offline synthetic tests for scientific-pipeline contracts; no model or network calls."""
import json,threading,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest
from ablation_harness.parsing.json_response import parse_json_response
from ablation_harness.config import HarnessConfig
from ablation_harness.runner import _LockedModel,_run_conditions_batch
from ablation_harness.scoring.metrics import ground_truth_for_scoring
from ablation_harness.scoring.confidence import compute_confidence_interval
from ablation_harness.tools.registry import ToolRegistry
import skill_agent.harness_adapter as bridge
from skill_agent.agent import SkillsAgent
from skill_agent.adapters import ModelAdapter
from skill_agent.config import RunConfig
import pandas as pd

def test_parser_brackets_inside_json_string():
 assert parse_json_response('[{"type":"PERSON","text":"A [B]","start":0,"end":5}]')[0]['text']=='A [B]'

def test_parser_later_valid_json_array():
 assert parse_json_response('Preamble [not JSON]. Final: [{"type":"PERSON"}]')==[{'type':'PERSON'}]

def test_parser_rejects_nonstring_type():
 assert parse_json_response('[{"type":null}]')==[]

def test_list_ground_truth_preserves_gold():
 g=[{'type':'PERSON','start':1,'end':3}]
 assert ground_truth_for_scoring({'ground_truth':g})==g

def test_lock_covers_message_generation():
 class Model:
  active=0;peak=0
  def generate_from_messages(self,*a,**kw):
   self.active+=1;self.peak=max(self.peak,self.active);time.sleep(.03);self.active-=1;return '[]'
 m=Model();locked=_LockedModel(m,threading.Lock())
 with ThreadPoolExecutor(2) as e:list(e.map(lambda _:locked.generate_from_messages([],max_tokens=2),range(2)))
 assert m.peak==1

def test_short_batch_is_not_silently_dropped(tmp_path):
 class Model:
  def generate_batch(self,*a):return ['[]']
  def generate(self,*a):return '[]'
 cfg=HarnessConfig(batch_size=2,resolve_offsets=False)
 ss=[{'id':str(i),'text':'x','source':'test','ground_truth':'[]'} for i in range(2)]
 try: result=_run_conditions_batch(Model(),'mock',ss,['zero_shot'],cfg,{'zero_shot':'{text}'},ToolRegistry())
 except ValueError:return
 assert len(result)==2

def test_document_loader_uses_configured_prompt_directory(tmp_path):
 p=tmp_path/'prompts';p.mkdir();(p/'with_docs.txt').write_text('DOCUMENT_SENTINEL')
 cfg=HarnessConfig(prompts_dir=p,skills_dir=tmp_path/'skills')
 assert any('DOCUMENT_SENTINEL' in s for s in bridge._build_docs(cfg))

def test_skill_system_context_reaches_string_model(tmp_path):
 bridge._agent_cache.clear();sd=tmp_path/'pii-detection';sd.mkdir();(sd/'SKILL.md').write_text('SKILL_SENTINEL')
 class Model:
  seen=[]
  def generate(self,prompt,max_tokens):self.seen.append(prompt);return '[]'
 m=Model();cfg=HarnessConfig(skills_dir=tmp_path,prompts_dir=tmp_path)
 bridge.run_skill_agent(m,{'text':'sample'},'TASK',cfg,ToolRegistry(),with_skills=True,with_docs=True,with_tools=True,model_name='mistral_7b')
 assert 'SKILL_SENTINEL' in m.seen[0]

def test_tool_call_queue_respects_budget():
 calls=[]
 def ping():calls.append(1);return 'ok'
 responses=iter(['[TOOL_CALL: analyze_pii]\n'*3,'[]'])
 def analyze_pii():return ping()
 agent=SkillsAgent(ModelAdapter(lambda **kw:next(responses)),[analyze_pii])
 agent.run('task',RunConfig(use_skillpack=False,use_rag=False,max_tool_steps=1))
 assert len(calls)<=1

def test_cached_sample_context_is_thread_isolated(tmp_path):
 bridge._agent_cache.clear();barrier=threading.Barrier(2);seen=[];local=threading.local()
 class Model:
  def generate(self,prompt,max_tokens):
   n=getattr(local,'n',0);local.n=n+1
   if n==0:barrier.wait(timeout=3);return '[TOOL_CALL: analyze_pii]'
   return '[]'
 class Detector:
  name='analyze_pii'
  def execute(self,text,**kw):seen.append(text);return {'detections':[]}
 reg=ToolRegistry();reg.register(Detector());m=Model();cfg=HarnessConfig(skills_dir=tmp_path,prompts_dir=tmp_path)
 # Warm the cache without invoking the barrier.
 local.n=1;bridge.run_skill_agent(m,{'text':'warm'},'warm',cfg,reg,with_tools=True,model_name='mistral_7b')
 def run(t):return bridge.run_skill_agent(m,{'text':t},t,cfg,reg,with_tools=True,model_name='mistral_7b')
 with ThreadPoolExecutor(2) as e:list(e.map(run,['sample A','sample B']))
 assert sorted(seen)==['sample A','sample B']

def test_ci_rejects_missing_or_insufficient_values():
 for vals in [[.2],[.2,float('nan'),.8]]:
  try: lo,hi,margin=compute_confidence_interval(pd.Series(vals))
  except ValueError:continue
  assert not (pd.notna(lo) and pd.notna(hi) and margin==0)

def test_detector_gold_join_repairs_missing_gold_without_mutating_raw(tmp_path):
 from ablation_harness.scoring.saved_analysis import load_saved_run
 rows=[{'model':'m','condition':'zero_shot','sample_id':'1','ground_truth':[{'type':'PERSON'}],'predictions':[]}]
 detector=[{'model':'det','condition':'detector','sample_id':'1','ground_truth':[],'predictions':[{'type':'PERSON'}]}]
 for name,value in [('main',rows),('detector',detector)]:
  path=tmp_path/'results/past_runs'/name/'experiment_results.json';path.parent.mkdir(parents=True);path.write_text(json.dumps(value))
 before=(tmp_path/'results/past_runs/detector/experiment_results.json').read_bytes()
 fixed=load_saved_run(tmp_path,'detector')
 assert fixed[0]['ground_truth']==rows[0]['ground_truth']
 assert (tmp_path/'results/past_runs/detector/experiment_results.json').read_bytes()==before

def test_saved_analysis_does_not_merge_unsupported_labels():
 from ablation_harness.scoring.saved_analysis import norm
 assert norm('AGE') != norm('Title')
