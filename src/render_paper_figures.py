from pathlib import Path
import csv

ROOT=Path(__file__).resolve().parents[1]

def pdf_escape(text): return text.replace('\\','\\\\').replace('(','\\(').replace(')','\\)')
def write_pdf(path, width, height, commands):
    stream='\n'.join(commands).encode('latin-1')
    objs=[b'<< /Type /Catalog /Pages 2 0 R >>',b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>'.encode(),b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',f'<< /Length {len(stream)} >>\nstream\n'.encode()+stream+b'\nendstream']
    out=bytearray(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n'); offsets=[0]
    for i,obj in enumerate(objs,1):
        offsets.append(len(out));out.extend(f'{i} 0 obj\n'.encode()+obj+b'\nendobj\n')
    xref=len(out);out.extend(f'xref\n0 {len(objs)+1}\n0000000000 65535 f \n'.encode())
    for off in offsets[1:]:out.extend(f'{off:010d} 00000 n \n'.encode())
    out.extend(f'trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode())
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(out)
def text(cmds,x,y,size,s):cmds.append(f'BT /F1 {size} Tf {x:.1f} {y:.1f} Td ({pdf_escape(s)}) Tj ET')
def fill_rgb(v,maxv):
    z=max(-1,min(1,v/maxv))
    if z>=0:return (1-0.65*z,1-0.2*z,1-0.05*z)
    return (1-0.05*abs(z),1-0.55*abs(z),1-0.55*abs(z))
def rect(cmds,x,y,w,h,color):cmds.append(' '.join(f'{c:.3f}' for c in color)+' rg');cmds.append(f'{x:.1f} {y:.1f} {w:.1f} {h:.1f} re f')
# Figure 1: addition grid
with (ROOT/'data/current/addition_grid_79cell.csv').open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
vals={(int(r['a']),int(r['b'])):float(r['symmetric_logit_diff']) for r in rows}
cmd=['1 1 1 rg 0 0 720 720 re f'];text(cmd,130,680,17,'GPT-2 Small addition grid');text(cmd,130,658,10,'Symmetric logit difference by operand pair; 79 evaluated cells. Repeated target sums are not independent.')
x0,y0,cw,ch=115,135,55,55
text(cmd,300,112,11,'Second operand b');text(cmd,47,375,11,'First operand a')
for a in range(1,10):
 text(cmd,x0-25,y0+(9-a)*ch+20,10,str(a));text(cmd,x0+(a-1)*cw+22,y0+9*ch+8,10,str(a))
for (a,b),v in vals.items():
 x=x0+(b-1)*cw;y=y0+(9-a)*ch;color=fill_rgb(v,0.75);rect(cmd,x,y,cw-2,ch-2,color)
 if a==b:
  cmd.append('0.12 0.23 0.55 RG 2 w');cmd.append(f'{x} {y} {cw-2} {ch-2} re S')
 tc='1 1 1' if abs(v)>0.4 else '0.12 0.18 0.25';cmd.append(tc+' rg');text(cmd,x+9,y+23,9,f'{v:+.3f}')
write_pdf(ROOT/'paper/figures/addition_grid_heatmap.pdf',720,720,cmd)
# Figure 2: clean rerun matched operator target means
with (ROOT/'data/current/equal_operand_operator_advantages.csv').open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
ops=['plus','minus','times','and','then'];targets=[int(r['T']) for r in rows];vals=[[float(r[o]) for o in ops] for r in rows]; vmax=max(abs(v) for r in vals for v in r)
cmd=['1 1 1 rg 0 0 720 590 re f'];text(cmd,155,555,17,'Equal-operand advantage by target and operator');text(cmd,155,533,10,'Clean-rerun target-level contrasts; no plus-versus-other difference after Holm correction (p = 0.6875).')
x0,y0,cw,ch=175,133,93,48
for j,op in enumerate(ops): text(cmd,x0+j*cw+24,y0+7*ch+10,10,op)
for i,T in enumerate(targets):
 y=y0+(6-i)*ch;text(cmd,145,y+18,10,str(T))
 for j,v in enumerate(vals[i]):
  x=x0+j*cw;color=fill_rgb(v,vmax);rect(cmd,x,y,cw-2,ch-2,color);text(cmd,x+20,y+19,9,f'{v:+.3f}')
text(cmd,160,90,9,'The sum token is scored under every string; non-addition rows are not arithmetic-accuracy tests.')
write_pdf(ROOT/'paper/figures/operator_target_advantages.pdf',720,590,cmd)
print('Rendered paper PDFs from current CSV sources.')
