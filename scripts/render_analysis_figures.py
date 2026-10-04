"""Browser-free vector PDF exports of corrected analysis tables (requires reportlab)."""
from pathlib import Path
import csv,math,argparse
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
import reportlab
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
FONT_DIR = Path(reportlab.__file__).resolve().parent / "fonts"
pdfmetrics.registerFont(TTFont("ReportSans", str(FONT_DIR / "Vera.ttf")))
pdfmetrics.registerFont(TTFont("ReportSansBold", str(FONT_DIR / "VeraBd.ttf")))
ROOT=Path(__file__).resolve().parents[1]
COLORS=['#5C6B73','#9A8C7D','#3D5A6C','#8B9E6B','#8FB4C4','#5C5346','#AA9ABF']
def rows(name):
 with (DATA/name).open() as f:return list(csv.DictReader(f))
def pdf(name,title,subtitle,width=720,height=420):
 c=canvas.Canvas(str(OUT/name),pagesize=(width,height));c.setTitle(title)
 c.setFont('ReportSansBold',14);c.drawString(28,height-30,title)
 c.setFont('ReportSans',9);c.setFillColor(HexColor('#555555'));c.drawString(28,height-47,subtitle);return c

def line_chart(name,title,subtitle,labels,series,maximum,ylabel):
 c=pdf(name,title,subtitle);x0,y0,w,h=65,83,610,255
 c.setFont('ReportSans',9)
 for n in range(6):
  val=maximum*n/5;y=y0+h*n/5
  c.setStrokeColor(HexColor('#DDDDDD'));c.line(x0,y,x0+w,y);c.setFillColor(HexColor('#555555'))
  c.drawRightString(x0-8,y-3,f'{val:.0f}' if maximum>1 else f'{val:.2f}')
 c.drawString(x0,355,ylabel)
 xs=[x0+w*i/(len(labels)-1) for i in range(len(labels))]
 for x,label in zip(xs,labels):c.drawCentredString(x,y0-18,label)
 for index,(name,values) in enumerate(series):
  col=HexColor(COLORS[index]);c.setFillColor(col);c.setStrokeColor(col);c.setLineWidth(2)
  last=None
  for x,v in zip(xs,values):
   y=y0+h*v/maximum
   if last:c.line(*last,x,y)
   c.circle(x,y,3,fill=1,stroke=0);last=(x,y)
  lx=65+(index%4)*155;ly=40-(index//4)*16;c.line(lx,ly,lx+14,ly);c.drawString(lx+20,ly-3,name)
 c.save()

def render():
 f=rows('df_funnel.csv')
 line_chart('fig_offset_provenance.pdf','Localization diagnostics after matching correction',
            'Independent one-to-one maximum matchings; count ratios do not track the same entities.',
            ['Type match','Any overlap','IoU >= 0.5','Exact boundary'],
            [(r['condition'],[float(r[k]) for k in ['type %','any overlap %','IoU≥0.5 %','exact %']]) for r in f],100,'Gold instances matched (%)')
 v=rows('df_over.csv')[::-1]
 c=pdf('fig_overprediction.pdf','Prediction volume with comparable denominators',
       'Predictions and gold are both pooled over the same 16,000 tool/skill evaluation rows.',820,500)
 x0,y0,w=245,75,535;maxval=max(float(r[k]) for r in v for k in ['predicted','support']); xmax=10**math.ceil(math.log10(maxval))
 def x(value):return x0+w*math.log10(max(value,1))/math.log10(xmax)
 c.setFont('ReportSans',8)
 for exponent in range(int(math.log10(xmax))+1):
  xx=x(10**exponent);c.setStrokeColor(HexColor('#DDDDDD'));c.line(xx,y0,xx,435);c.drawCentredString(xx,y0-16,f'{10**exponent:,}')
 for i,r in enumerate(v):
  y=y0+i*29;c.setFillColor(HexColor('#333333'));c.drawRightString(x0-8,y+7,r['type'])
  for k,dy,col in [('predicted',10,COLORS[2]),('support',0,COLORS[1])]:
   c.setFillColor(HexColor(col));c.rect(x0,y+dy,max(0,x(float(r[k]))-x0),8,stroke=0,fill=1)
 c.setFillColor(HexColor('#333333'));c.drawString(245,28,'Counts (log scale). Teal: predictions. Tan: gold in the same evaluations.')
 c.save()
 precision=rows('precision_clustered.csv'); typ=[r for r in precision if r['metric']=='type']
 c=pdf('fig_fp16_vs_4bit.pdf','Matched precision control',
       '95% t intervals across 300 example means; conditions averaged within each example.',720,350)
 x0,y0,w=180,80,490;lo=min(float(r['low']) for r in typ)-.005;hi=max(float(r['high']) for r in typ)+.005
 def x(v):return x0+w*(v-lo)/(hi-lo)
 c.setFont('ReportSans',10);c.setStrokeColor(HexColor('#AAAAAA'));c.setDash(2,2);c.line(x(0),y0-15,x(0),260);c.setDash()
 names={'gemma2_9b':'Gemma 2 9B','llama3_8b':'Llama 3.1 8B','mistral_7b':'Mistral 7B','qwen2_7b':'Qwen 2.5 7B'}
 for i,r in enumerate(typ):
  y=245-i*48;c.setFillColor(HexColor('#333333'));c.drawRightString(x0-12,y-3,names[r['model']]);c.setStrokeColor(HexColor(COLORS[0]));c.setLineWidth(2)
  a,b,mid=x(float(r['low'])),x(float(r['high'])),x(float(r['mean']));c.line(a,y,b,y);c.line(a,y-4,a,y+4);c.line(b,y-4,b,y+4);c.circle(mid,y,4,fill=1)
  c.drawCentredString(mid,y+12,f"{float(r['mean']):+.3f}")
 for i in range(6):
  v=lo+(hi-lo)*i/5;c.drawCentredString(x(v),55,f'{v:+.3f}')
 c.drawString(245,28,'Mean per-example F1 difference (16-bit minus 4-bit)');c.save()
 # Within-family size comparison: values exported from saved Qwen runs.
 scale=rows('scaling_within_family.csv');labels=['Zero-shot','+Docs','+Tool','+Skills'];c=pdf('fig_scaling_14b.pdf','Existing Qwen size control',
 'Qwen 2.5 7B and 14B on the same benchmark; one family, not a general scaling law.')
 x0,y0,w,h=65,85,610,255;c.setFont('ReportSans',9)
 for i in range(5):
  y=y0+h*i/4;c.setStrokeColor(HexColor('#DDDDDD'));c.line(x0,y,x0+w,y);c.setFillColor(HexColor('#555555'));c.drawRightString(x0-8,y-3,f'{.8*i/4:.1f}')
 for j,label in enumerate(labels):
  mid=x0+(j+.5)*w/4;c.setFillColor(HexColor('#333333'));c.drawCentredString(mid,y0-17,label)
  for k,row in enumerate(scale):
   val=float(row[label]);xx=mid-35+k*36;c.setFillColor(HexColor(COLORS[k+1]));c.rect(xx,y0,31,h*val/.8,stroke=0,fill=1);c.drawCentredString(xx+15.5,y0+h*val/.8+7,f'{val:.3f}')
 c.setFillColor(HexColor('#333333'));c.drawString(65,355,'Mean per-example type F1');c.drawString(210,35,'Tan: Qwen 7B          Teal: Qwen 14B');c.save()
 # Preserve unmodified original plots except those affected by corrected source values.
 configs=rows('configuration_means.csv')
 line_chart('fig_dual_metric_slope.pdf','Type and mixed span/type scores',
            'Arithmetic mean of per-example F1; the mixed score falls back to type matching without gold spans.',
            ['Type-level','Mixed span/type'],[(r['label'],[float(r['type']),float(r['span'])]) for r in configs],.8,'Mean per-example F1')
 c=pdf('fig_all_conditions.pdf','Configuration comparison',
       'Type-level arithmetic mean of per-example F1; detector scored from the saved run.')
 configs=sorted(configs,key=lambda r:-float(r['type']));x0,y0,w,h=55,85,625,255;c.setFont('ReportSans',9)
 for i in range(5):
  y=y0+h*i/4;c.setStrokeColor(HexColor('#DDDDDD'));c.line(x0,y,x0+w,y);c.setFillColor(HexColor('#555555'));c.drawRightString(x0-6,y-3,f'{.8*i/4:.1f}')
 for i,r in enumerate(configs):
  val=float(r['type']);xx=x0+i*w/len(configs)+10;bw=w/len(configs)-20;c.setFillColor(HexColor(COLORS[i]));c.rect(xx,y0,bw,h*val/.8,fill=1,stroke=0);c.drawCentredString(xx+bw/2,y0+h*val/.8+8,f'{val:.3f}');c.setFillColor(HexColor('#333333'));c.drawCentredString(xx+bw/2,y0-18,r['label'])
 c.save()

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=ROOT/'analysis-results');parser.add_argument('--out',type=Path,default=ROOT/'notebooks');args=parser.parse_args()
 DATA,OUT=args.data,args.out;OUT.mkdir(exist_ok=True,parents=True);render();print('Exported six vector figures without launching a browser.')
