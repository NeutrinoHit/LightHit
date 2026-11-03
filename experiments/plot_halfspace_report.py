#!/usr/bin/env python
"""Render the half-space report figures from reproducible numerical outputs.

No transport solve is performed. Monte Carlo error bars on sums retain the
negative covariance of mutually exclusive collision-order channels.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation',type=Path,required=True)
    parser.add_argument('--materials',type=Path,required=True)
    parser.add_argument('--quadrature-audit',type=Path,required=True)
    parser.add_argument('--banana-data',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args()
    OUT=args.output_directory
    OUT.mkdir(parents=True,exist_ok=True)
    base=json.loads(args.validation.read_text())
    materials=json.loads(args.materials.read_text())
    audit=json.loads(args.quadrature_audit.read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,
                         'axes.titlesize':12,'axes.labelsize':11,'legend.fontsize':9,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.2,'grid.linewidth':.6,
                         'figure.facecolor':'white','savefig.facecolor':'white'})
    BLUE, ORANGE, GREEN = '#0072B2', '#D55E00', '#009E73'

    def save(fig, name):
        fig.savefig(OUT/f'{name}.png', dpi=190)
        fig.savefig(OUT/f'{name}.svg')
        plt.close(fig)

    def total_stderr(mean, stderr, count):
        # Order channels are mutually exclusive per history. Retain their
        # negative covariance when forming an error on the summed signal.
        m,e = np.asarray(mean),np.asarray(stderr)
        variance = np.sum(e*e,axis=-1)-(np.sum(m,axis=-1)**2-np.sum(m*m,axis=-1))/(count-1)
        return np.sqrt(np.maximum(0.,variance))

    fig, axes = plt.subplots(1,2,figsize=(11.2,4.7),gridspec_kw={'width_ratios':[1.1,1]},layout='constrained')
    ax=axes[0]
    for key,label,color in [('ptfe_dn','PTFE',BLUE),('milk_dn','Молоко',ORANGE)]:
        cases=[c for c in materials['cases'] if c['profile']==key]
        x=np.array([c['mu_a_per_cm'] for c in cases])
        y=np.array([c['total_reflectance'] for c in cases])
        ax.plot(x,100*y,'o-',color=color,lw=1.8,label=f'{label}: RTE')
        ymc=[];se=[]
        for c in cases:
            mc=c['monte_carlo'];m=np.array(mc['mean']);e=np.array(mc['stderr'])
            ymc.append(c['specular']+m.sum());se.append(total_stderr(m,e,mc['photons']))
        ax.errorbar(x*1.045,100*np.array(ymc),yerr=100*np.array(se),fmt='s',mfc='white',
                    ms=4,color=color,capsize=3,lw=1,label=f'{label}: МК ±1σ')
    c=next(c for c in materials['cases'] if c['profile']=='skin_dermis_660_hg')
    mc=c['monte_carlo'];m=np.array(mc['mean']);e=np.array(mc['stderr'])
    ax.scatter([c['mu_a_per_cm']],[100*c['total_reflectance']],marker='D',s=42,color=GREEN,label='Дерма, 660 нм')
    ax.errorbar([c['mu_a_per_cm']*1.045],[100*(c['specular']+m.sum())],
                yerr=[100*float(total_stderr(m,e,mc['photons']))],fmt='D',mfc='white',ms=4,color=GREEN,capsize=3)
    ax.set(xscale='log',xlim=(.00075,.23),ylim=(57,100),xlabel=r'$\mu_a$, см$^{-1}$',
           ylabel='Полное отражение, %',title='Зависимость от поглощения')
    ax.legend(loc='lower left',framealpha=.95)
    ax.text(.98,.98,r'PTFE и молоко: скан $\mu_a$'+'\nМК-точки слегка сдвинуты по x',transform=ax.transAxes,
            ha='right',va='top',fontsize=8.5)

    ax=axes[1]
    chosen=[next(c for c in materials['cases'] if c['profile']==key and abs(c['mu_a_per_cm']-a)<1e-12)
            for key,a in [('ptfe_dn',.01),('milk_dn',.01),('skin_dermis_660_hg',.135)]]
    data=np.array([[c['specular'],c['escaped_components'][1],c['escaped_components'][2],c['balance']['absorbed']]
                   for c in chosen])*100
    colors=['#777777','#56B4E9',BLUE,'#E69F00']
    labels=['Зеркальное отражение','1 рассеяние','≥2 рассеяний','Поглощено']
    left=np.zeros(3)
    for j,(label,color) in enumerate(zip(labels,colors)):
        ax.barh(np.arange(3),data[:,j],left=left,color=color,label=label,height=.57)
        for i in range(3):
            if data[i,j]>4:
                ax.text(left[i]+data[i,j]/2,i,f'{data[i,j]:.2f}%',ha='center',va='center',
                        color='white' if j==2 else '#111111',fontsize=9)
        left+=data[:,j]
    ax.set(yticks=np.arange(3),yticklabels=['PTFE\n'+r'$\mu_a=0.01$ см$^{-1}$','Молоко\n'+r'$\mu_a=0.01$ см$^{-1}$',
                                           'Дерма\n'+r'$\mu_a=0.135$ см$^{-1}$'],
           xlim=(0,100),xlabel='Доля внешнего падающего света, %',title='Баланс по каналам')
    ax.invert_yaxis();ax.grid(axis='y',visible=False)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.17),ncol=2,fontsize=8.5)
    save(fig,'materials')

    fig,axes=plt.subplots(1,2,figsize=(11.2,4.5),layout='constrained')
    for case,color,marker in zip(base['cases'],[BLUE,ORANGE],['o','s']):
        rad=case['radial'];rho=np.asarray(rad['radii_m']);model=np.array(rad['flux_density_per_m2']['real'])
        mc=np.array(rad['monte_carlo']['estimate']['real']);se=np.array(rad['monte_carlo']['stderr_real'])
        total=mc.sum(axis=-1);err=total_stderr(mc,se,case['banana']['photons'])
        axes[0].plot(rho,model.sum(axis=1),color=color,lw=2,label=f"RTE, n={case['n']:g}")
        axes[0].errorbar(rho,total,yerr=err,fmt=marker,mfc='white',ms=5,capsize=3,color=color,
                          label=f"МК ±1σ, n={case['n']:g}")
        for j,style in [(1,'o'),(2,'^')]:
            pull=(model[:,j]-mc[:,j])/se[:,j]
            offset=(.015 if case['n']==1 else -.015)+(j-1.5)*.013
            axes[1].plot(rho+offset,pull,style,ms=5,color=color,label=f"n={case['n']:g}, {'1' if j==1 else '≥2'} расс.")
    axes[0].set(yscale='log',xlabel=r'Смещение приёмника $\rho$, м',ylabel=r'$D(\rho)$, м$^{-2}$ на падающий фотон',
                 title='Пространственный выходящий поток')
    axes[0].legend(framealpha=.95)
    axes[1].axhspan(-1,1,color='#aaaaaa',alpha=.15)
    axes[1].axhline(0,color='#444444',lw=.8)
    for sign in [-1,1]:axes[1].axhline(sign*3,color='#777777',ls='--',lw=.8)
    axes[1].set(xlabel=r'$\rho$, м',ylabel=r'$(D_{RTE}-D_{MC})/SE_{MC}$',ylim=(-3.35,3.35),
                 title='Отклонения по порядкам')
    axes[1].legend(ncol=2,fontsize=8.5,loc='upper center')
    fig.suptitle(r'Синтетический тест: g=0.3, $\mu_a=0.2$ м$^{-1}$, $\mu_s=1$ м$^{-1}$; $\sigma_s=\sigma_d=0.4$ м, $\alpha=0.7$ рад',fontsize=11)
    save(fig,'spatial')

    data=np.load(args.banana_data,allow_pickle=False)
    x,z,K=data['x_m'],data['z_m'],data['sensitivity_per_m4']
    fig,ax=plt.subplots(figsize=(9.4,5.2),layout='constrained')
    ax.grid(False)
    im=ax.pcolormesh(x,z,K,shading='auto',cmap='magma',norm=LogNorm(vmin=K.max()/300,vmax=K.max()))
    ax.set(xlabel='x, м',ylabel='Глубина z, м',title='Срез чувствительности к поглощению: y=0, n=1.4, g=0.3')
    ax.invert_yaxis()
    cells=base['cases'][1]['banana']['cells_m']
    for label,(lo,hi) in zip(['A','B','C'],cells):
        ax.add_patch(Rectangle((lo[0],lo[2]),hi[0]-lo[0],hi[2]-lo[2],fill=False,edgecolor='#00BFC4',lw=1.6))
        ax.text((lo[0]+hi[0])/2,(lo[2]+hi[2])/2,label,ha='center',va='center',color='white',weight='bold',fontsize=12,
                 bbox={'facecolor':'black','alpha':.3,'edgecolor':'none','pad':1.5})
    ax.scatter([0,1],[0,0],c='#00AAB5',marker='v',s=60,clip_on=False,zorder=4)
    ax.text(0,-.04,'Пучок по нормали',ha='center',va='bottom',fontsize=9)
    ax.text(1,-.04,'Приёмник',ha='center',va='bottom',fontsize=9)
    fig.colorbar(im,ax=ax,label=r'$K_a=-\delta D/\delta\mu_a$, м$^{-4}$ на падающий фотон')
    save(fig,'banana')

    ref=audit['refined_q_zero'][1]['balance']['escaped']
    entries=audit['polar_quadrature']
    p=np.array([a['polar_nodes'] for a in entries])
    internal=np.array([a['balance']['escaped'] for a in entries])
    external=np.array([sum(a['receiver128']) for a in entries])
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.4),layout='constrained')
    axes[0].semilogx(p,100*(internal/ref-1),'o-',color=BLUE,label='Выход из внутреннего баланса')
    axes[0].semilogx(p,100*(external/ref-1),'s-',color=ORANGE,label='Прямой поток в приёмник')
    axes[0].axhline(0,color='#555555',lw=.8)
    axes[0].set(xlabel='Полярные узлы на интервал',ylabel='Отличие от уточнённого q=0, %',title='Сходимость интегрального сигнала')
    axes[0].legend(fontsize=8.5)
    axes[1].loglog(p,100*np.abs(internal-external)/ref,'o-',color=GREEN,label='Разность двух интегралов')
    axes[1].axhline(.01,color='#777777',ls='--',label='0.01%')
    axes[1].set(xlabel='Полярные узлы на интервал',ylabel='Относительная разность, %',title='Баланс ≠ оценка точности квадратуры')
    axes[1].legend(fontsize=8.5)
    fig.suptitle('Синтетическая среда, n=1.4: кривая L=10; уточнённый ориентир P=128, L=18',fontsize=11)
    save(fig,'convergence')

    print('Rendered materials, spatial, banana and convergence: PNG + SVG.')
    return 0


if __name__=="__main__":
    raise SystemExit(main())
