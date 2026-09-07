"""Render the six quantitative manuscript figures from frozen numerical results.

Plot functions derive from Experiment_Figures.ipynb (Drive snapshot); stopping
labels are calculated from the supplied diagnostics.
No training or image inputs are required.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

NAA='NAA-GDP'; PRIVATE='C-GDP-private-noise'
COLORS={NAA:'#947389',PRIVATE:'#536C8A','I-GDP':'#73777F','Local-only':'#AA895C',
        'MP':'#536C8A','AM':'#58988F','OP':'#947389'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
 'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'axes.titlesize':8,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
 'lines.linewidth':1.25,'lines.markersize':3.3,'pdf.fonttype':42,'ps.fonttype':42,
 'svg.fonttype':'none','savefig.facecolor':'white'})

def stats(df,keys,metrics):
    seed=df.groupby(keys+['outer_seed'])[metrics].mean().reset_index()
    assert seed.groupby(keys).size().eq(5).all()
    result=seed.groupby(keys)[metrics].agg(['mean','std'])
    result.columns=['_'.join(c) for c in result.columns]
    return result.reset_index()

def summary(df):
    s=stats(df,['method','participant_count','noise_scale'],['balanced_accuracy','linkage_rate'])
    for c in s:
        if c.endswith(('_mean','_std')): s[c]*=100
    return s

def polished(ax,axis='both'):
    ax.grid(True,axis=axis,color='#DADDE0',lw=.5,alpha=.65);ax.set_axisbelow(True)

def series(u,method,p):
    base='AA-GDP' if method==NAA else 'C-GDP'
    return u[(u.participant_count==p)&u.method.isin([base,method])].sort_values('noise_scale')

def save(fig,name,out):
    for ext in ('pdf','png'):
        fig.savefig(out/f'{name}.{ext}',dpi=200,bbox_inches='tight',pad_inches=.03)
    plt.close(fig)

def tradeoff(u,p,dataset,out):
    fig,ax=plt.subplots(figsize=(3.45,2.75))
    for method,marker,label in [(NAA,'o','Anchor noise (NAA-GDP)'),(PRIVATE,'s','Private noise (C-GDP)')]:
        d=series(u,method,p)
        ax.errorbar(d.linkage_rate_mean,d.balanced_accuracy_mean,
            xerr=d.linkage_rate_std,yerr=d.balanced_accuracy_std,
            color=COLORS[method],marker=marker,capsize=1.6,elinewidth=.55,capthick=.55,label=label)
    for m,ls in [('I-GDP','--'),('Local-only',':')]:
        y=u[(u.method==m)&(u.participant_count==p)].balanced_accuracy_mean.iloc[0]
        ax.axhline(y,color=COLORS[m],ls=ls,lw=1,label=m)
    ax.axvline(10,color='#B5B8BB',ls='--',lw=.65,zorder=0)
    ax.set(xlabel='Identity linkage (%)',ylabel='Balanced accuracy (%)',
           xlim=(7,81 if dataset=='CelebA' else 73),ylim=(60,92) if dataset=='CelebA' else (48,79))
    if dataset=='LFWA':
        plotted=pd.concat([series(u,m,p) for m in (NAA,PRIVATE)])
        assert (plotted.linkage_rate_mean-plotted.linkage_rate_std).min()>=ax.get_xlim()[0]
        assert (plotted.linkage_rate_mean+plotted.linkage_rate_std).max()<=ax.get_xlim()[1]
        assert (plotted.balanced_accuracy_mean-plotted.balanced_accuracy_std).min()>=ax.get_ylim()[0]
        assert (plotted.balanced_accuracy_mean+plotted.balanced_accuracy_std).max()<=ax.get_ylim()[1]
    polished(ax)
    h,l=ax.get_legend_handles_labels()
    # Utility references are intentionally not assigned a privacy coordinate.
    order=[2,3,0,1] if l[0]=='I-GDP' else list(range(len(l)))
    fig.legend([h[i] for i in order],[l[i] for i in order],loc='lower center',
               ncol=1,frameon=False,bbox_to_anchor=(.57,-.012),fontsize=7)
    fig.subplots_adjust(left=.17,right=.98,top=.97,bottom=.40)
    save(fig,dataset.lower()+'_privacy_utility',out)

def participants(u,out):
    fig,ax=plt.subplots(figsize=(3.45,2.8))
    colors=['#536C8A','#58988F','#AA895C','#947389','#465366']
    handles=[]
    for p,color,marker in zip([2,5,10,20,50],colors,['o','s','^','D','v']):
        d=series(u,NAA,p).sort_values('linkage_rate_mean')
        x=d.linkage_rate_mean.to_numpy();y=d.balanced_accuracy_mean.to_numpy();sd=d.balanced_accuracy_std.to_numpy()
        ax.plot(x,y,color=color,marker=marker,markersize=2.6,label=f'p = {p}')
        ax.fill_between(x,y-sd,y+sd,color=color,alpha=.09,lw=0)
        y0=u[(u.method=='I-GDP')&(u.participant_count==p)].balanced_accuracy_mean.iloc[0]
        ax.axhline(y0,color=color,lw=.8,ls='--',alpha=.85)
        handles.append(Line2D([],[],color=color,marker=marker,label=f'p = {p}'))
    ax.set(xlabel='Identity linkage (%)',ylabel='Balanced accuracy (%)',xlim=(8,80),ylim=(77,91))
    polished(ax)
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.57,.01))
    fig.subplots_adjust(left=.17,right=.98,top=.98,bottom=.29)
    save(fig,'celeba_participant_sensitivity',out)

def lfwa_bars(raw,attacks,out):
    a=stats(attacks[attacks.method==NAA],['noise_scale','attack'],['linkage_rate'])
    levels=np.sort(raw.loc[raw.method==NAA,'noise_scale'].unique())
    assert len(levels)==10 and levels.min()>0
    fig,ax=plt.subplots(figsize=(6.9,2.5));xx=np.arange(len(levels));width=.23
    for j,attack in enumerate(['MP','AM','OP']):
        d=a[a.attack==attack].sort_values('noise_scale')
        np.testing.assert_allclose(d.noise_scale,levels,rtol=0,atol=1e-12)
        ax.bar(xx+(j-1)*width,d.linkage_rate_mean*100,width=.215,
          yerr=d.linkage_rate_std*100,color=COLORS[attack],label=attack,
          capsize=2,error_kw={'elinewidth':.7,'capthick':.7})
    ax.axhline(10,color='#8B9095',ls='--',lw=.8)
    ax.set_xticks(xx,[f'{v:.3f}' for v in levels])
    upper=max(65,5*np.ceil(100*(a.linkage_rate_mean+a.linkage_rate_std).max()/5))
    ax.set(xlabel='Anchor-noise scale, $v$',ylabel='Identity linkage (%)',ylim=(0,upper))
    ax.legend(frameon=False,ncol=3,loc='upper right')
    polished(ax,'y');fig.subplots_adjust(left=.08,right=.995,top=.97,bottom=.21)
    save(fig,'lfwa_anchor_noise_identity_leakage',out)

def celeba_bars(u,out):
    fig,axes=plt.subplots(1,2,figsize=(6.9,2.2),sharey=True)
    for ax,m,title in zip(axes,[NAA,PRIVATE],['(a) Anchor noise: OP','(b) Private noise: common-transform inversion']):
        d=series(u,m,10);xx=np.arange(len(d))
        ax.bar(xx,d.linkage_rate_mean,yerr=d.linkage_rate_std,color=COLORS[m],width=.72,
             capsize=1.5,error_kw={'elinewidth':.55,'capthick':.55})
        ax.axhline(10,color='#8B9095',ls='--',lw=.8)
        labels=[('0' if v==0 else f'{v:.3f}') if m==NAA else f'{v:g}' for v in d.noise_scale]
        ax.set_xticks(xx,labels,rotation=45,ha='right')
        ax.set(xlabel='Anchor-noise scale, $v$' if m==NAA else r'Private-data noise SD, $\sigma$',ylim=(0,82))
        ax.set_title(title,fontsize=7.7);polished(ax,'y')
    axes[0].set_ylabel('Identity linkage (%)')
    fig.subplots_adjust(left=.08,right=.99,top=.85,bottom=.30,wspace=.13)
    save(fig,'celeba_noise_identity_leakage',out)

def gpm(inputs,raw,out):
    g=pd.read_csv(inputs/'celeba_gpm_diagnostics.csv')
    target=raw[(raw.method==NAA)&(raw.participant_count==10)]
    assert len(g)==200 and set(g.run_id)==set(target.run_id)
    counts={True:int(g.converged.eq(True).sum()),False:int(g.converged.eq(False).sum())}
    fig,axes=plt.subplots(1,3,figsize=(6.9,2.35))
    for ax,field,title,ylabel,log in zip(axes,
       ['iterations','fixed_point_residual','elapsed_seconds'],
       ['(a) Iterations','(b) Relative fixed-point residual','(c) Alignment time'],
       ['GPM iterations','Relative residual','Seconds'],[False,True,True]):
        for converged,color,label in [(True,'#58988F',f'Converged ({counts[True]})'),(False,'#947389',f'Capped ({counts[False]})')]:
            d=g[g.converged==converged]
            # Small deterministic offsets separate the 20 observations at each tested level.
            offsets=(d.outer_seed.to_numpy()*4+d.realization.to_numpy()-9.5)*.000105
            ax.scatter(d.noise_scale+offsets,d[field],s=10,edgecolors=color,
              facecolors=color if converged else 'none',linewidths=.5,alpha=.8,label=label)
        ax.axvline(.0530,color='#9D9FA1',ls='--',lw=.75)
        ax.set(xlabel='Anchor-noise scale, $v$',ylabel=ylabel,xlim=(0,.064))
        ax.set_xticks([0,.02,.04,.06],['0','.02','.04','.06']);ax.set_title(title,fontsize=7.5)
        if log: ax.set_yscale('log')
        polished(ax)
    handles,labels=axes[0].get_legend_handles_labels()
    handles.append(Line2D([],[],color='#9D9FA1',ls='--',label='$v_{PT}=0.0530$'));labels.append('$v_{PT}=0.0530$')
    fig.legend(handles,labels,ncol=3,frameon=False,loc='lower center',bbox_to_anchor=(.53,0))
    fig.subplots_adjust(left=.075,right=.985,bottom=.31,top=.86,wspace=.52)
    save(fig,'celeba_gpm_diagnostics',out)



def main():
    from hashlib import sha256
    from datetime import datetime, timezone
    from validate_results import validate_all, locate_gpm

    current = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=current.parents[0]/'build'/'current_figures')
    parser.add_argument('--results-root', type=Path, default=current/'results', help='Separately supplied saved results directory')
    args = parser.parse_args()
    results_root = args.results_root.resolve()
    try:
        evidence = validate_all(current, results_root=results_root)
    except (FileNotFoundError, ValueError, AssertionError) as exc:
        parser.error(str(exc))
    gpm_path = locate_gpm(results_root)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lfwa = results_root/'lfwa_full_sweep'
    celeba = results_root/'celeba_parallel'/'merged'
    raw_l = pd.read_csv(lfwa/'raw_results.csv')
    raw_c = pd.read_csv(celeba/'raw_results.csv')
    ls, cs = summary(raw_l), summary(raw_c)
    ls.to_csv(out/'lfwa_plot_statistics.csv', index=False)
    cs.to_csv(out/'celeba_plot_statistics.csv', index=False)
    tradeoff(ls, 5, 'LFWA', out)
    tradeoff(cs, 10, 'CelebA', out)
    participants(cs, out)
    lfwa_bars(raw_l, pd.read_csv(lfwa/'attack_results.csv'), out)
    celeba_bars(cs, out)
    gpm(gpm_path.parent, raw_c, out)
    sources = [lfwa/'raw_results.csv', lfwa/'attack_results.csv',
               celeba/'raw_results.csv', gpm_path]
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Six quantitative current-manuscript figures; no fitting or images.',
        'validation': evidence,
        'sources': {str(p.relative_to(results_root)): sha256(p.read_bytes()).hexdigest() for p in sources},
        'renderer_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
        'outputs': {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())
                    if p.suffix in {'.pdf', '.png', '.csv'}},
        'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'matplotlib': matplotlib.__version__},
    }
    (out/'figure_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Rendered six quantitative figures in {out}')


if __name__ == '__main__':
    main()
