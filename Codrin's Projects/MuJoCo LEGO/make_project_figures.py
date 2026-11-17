"""Create a CAD drawing from the actual meshes and plot recorded training returns."""
from pathlib import Path
import json
import re
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent
MEDIA = ROOT / 'media'


def rotation(text):
    w, x, y, z = np.fromstring(text or '1 0 0 0', sep=' ')
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def main():
    model = ROOT / 'simulation/onshape/beam_robot.xml'
    xml = ET.parse(model).getroot()
    assets = {m.get('name'): m.get('file') for m in xml.findall('./asset/mesh')}
    parts = []
    def walk(body, parent_r, parent_p):
        r = parent_r @ rotation(body.get('quat'))
        p = parent_p + parent_r @ np.fromstring(body.get('pos', '0 0 0'), sep=' ')
        for g in body.findall('geom'):
            if g.get('group') != '2' or g.get('type') != 'mesh':
                continue
            mesh = trimesh.load(model.parent / 'assets' / assets[g.get('mesh')], force='mesh', process=False)
            gr = r @ rotation(g.get('quat'))
            gp = p + r @ np.fromstring(g.get('pos', '0 0 0'), sep=' ')
            vertices = (mesh.vertices @ gr.T + gp) * 1000
            color = '#e5a328' if body.get('name') == 'chassis' else '#8ebccb'
            parts.append((body.get('name'), vertices[mesh.faces], color))
        for b in body.findall('body'):
            walk(b, r, p)
    walk(xml.find('./worldbody/body'), np.eye(3), np.zeros(3))
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor='#f7f9fb')
    titles = ['TOP / x–y', 'SIDE / x–z', 'FRONT / y–z', 'ISOMETRIC / reference pose']
    views = [np.array([[1,0,0],[0,1,0],[0,0,1]]),
             np.array([[1,0,0],[0,0,1],[0,-1,0]]),
             np.array([[0,1,0],[0,0,1],[1,0,0]])]
    a, b = np.deg2rad(35), np.deg2rad(28)
    views.append(np.array([[np.cos(a),-np.sin(a),0],
                           [np.sin(a)*np.sin(b),np.cos(a)*np.sin(b),np.cos(b)],
                           [-np.sin(a)*np.cos(b),-np.cos(a)*np.cos(b),np.sin(b)]]))
    from matplotlib.colors import to_rgb
    for ax, view, title in zip(axes.flat, views, titles):
        all_tri, all_col = [], []
        for _,triangles,color in parts:
            projected = triangles @ view.T
            normals = np.cross(projected[:,1]-projected[:,0], projected[:,2]-projected[:,0])
            normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
            light = .62+.38*np.abs(normals @ np.array([.3,.4,.866]))
            all_tri.append(projected)
            all_col.append(np.clip(light[:,None]*np.array(to_rgb(color)),0,1))
        tri, colors = np.concatenate(all_tri), np.concatenate(all_col)
        order = np.argsort(tri[:,:,2].mean(1))
        ax.add_collection(PolyCollection(tri[order,:,:2],facecolors=colors[order],edgecolors='none',rasterized=True))
        xy = tri[:,:,:2].reshape(-1,2)
        lo, hi = xy.min(0),xy.max(0)
        ax.set_xlim(lo[0]-15,hi[0]+15); ax.set_ylim(lo[1]-17,hi[1]+12)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(title,loc='left',color='#183846',fontsize=12,fontweight='bold')
        if title.startswith('TOP'):
            yy=lo[1]-9
            ax.annotate('',xy=(lo[0],yy),xytext=(hi[0],yy),arrowprops={'arrowstyle':'|-|','color':'#466574'})
            ax.text((lo[0]+hi[0])/2,yy-4,f'{hi[0]-lo[0]:.1f} mm overall',ha='center',va='top',fontsize=9)
    fig.suptitle('LEGO DOUBLE MOTOR / TWO HOOKED BEAMS',x=.06,y=.98,ha='left',fontsize=20,fontweight='bold',color='#123344')
    fig.text(.06,.932,'CAD-derived drawing  •  2 shaft hinges  •  x forward, y left, z up  •  dimensions from exported meshes',color='#476575')
    fig.text(.06,.025,'Original Onshape surfaces retained. Orange = motor case; blue = beams. Views show the model’s zero pose; no wheels.',color='#476575',fontsize=10)
    fig.subplots_adjust(top=.86,bottom=.08,hspace=.17,wspace=.14)
    fig.savefig(MEDIA/'cad-drawing.png',dpi=170,facecolor=fig.get_facecolor()); plt.close(fig)
    mass = sum(float(x.get('mass')) for x in xml.findall('.//inertial')) + sum(float(g.get('mass','0')) for g in xml.findall('.//geom'))
    dims = {name:(tri.reshape(-1,3).max(0)-tri.reshape(-1,3).min(0)).tolist() for name,tri,_ in parts}
    (MEDIA/'cad-measurements.json').write_text(json.dumps({'mass_kg':mass,'visual_extents_mm_xyz':dims},indent=2)+'\n')
    fig, ax = plt.subplots(figsize=(10,4.6),facecolor='white')
    for folder,label,color in [('original','Original policy / job 7271209','#c57a15'),('motor-limited','Motor limits matched / job 7274265','#167586')]:
        text=(ROOT/'experiments'/folder/'training.log').read_text(encoding='utf-8')
        text=re.sub(r'\x1b\[[0-9;]*m','',text)
        rows=re.findall(r'Learning iteration (\d+)/\d+.*?Mean reward:\s*([-\d.]+)',text,re.S)
        xs,ys=zip(*[(int(i),float(y)) for i,y in rows])
        ax.plot(xs,ys,color=color,label=label,lw=1.7)
    ax.set(title='Recorded PPO training returns',xlabel='Training iteration',ylabel='Logged mean reward')
    ax.spines[['top','right']].set_visible(False); ax.grid(alpha=.15); ax.legend(frameon=False)
    fig.text(.125,.015,'512 parallel environments · 24 steps per update · 300 iterations per run. Curves are training logs, not real-robot scores.',fontsize=9,color='#476575')
    fig.tight_layout(rect=(0,.05,1,1)); fig.savefig(MEDIA/'training-rewards.png',dpi=170);plt.close(fig)
    print(json.dumps({'mass_kg':mass,'visual_extents_mm_xyz':dims},indent=2))


if __name__=='__main__':
    main()
