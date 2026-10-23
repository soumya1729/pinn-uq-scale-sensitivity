"""Readable figure variant; unchanged selections and values from frozen review CSVs."""
from pathlib import Path
import argparse
import csv
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser(description='Plot manuscript figures from the reviewed round-two CSVs.')
parser.add_argument('--source-root',type=Path,required=True,help='Paper directory containing the frozen review CSVs.')
parser.add_argument('--paper-root',type=Path,required=True,help='Destination paper directory for the readable figures.')
args = parser.parse_args()
root = args.paper_root
review = args.source_root/'analysis/round2/outputs/production_review'
(root/'figures').mkdir(parents=True,exist_ok=True)
rows = list(csv.DictReader((review/'uncertainty_contrast_curves.csv').open()))
settings = list(csv.DictReader((review/'regional_scale_metrics.csv').open()))
plt.rcParams.update({'font.size':15,'axes.labelsize':15,'legend.fontsize':15,
                    'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
                    'axes.spines.right':False})
fig, axes = plt.subplots(1,2,figsize=(9.4,3.8),sharex=True)
colors = plt.get_cmap('tab10').colors
for seed in range(8):
    rs = [r for r in rows if r['tag']==f'burgers_mlp_seed{seed}'
          and r['centre']=='trained' and r['subset']=='full' and r['region']=='sharp']
    selected = {}
    for r in settings:
        if not (r['tag']==f'burgers_mlp_seed{seed}' and r['centre']=='trained'
                and r['subset']=='full' and r['region']=='sharp'): continue
        if r['setting']=='conditional_noise_mle': selected['MLE']=r
        if r['setting'].startswith('affine_evidence_original'): selected['evidence']=r
        if r['setting']=='prior_sweep' and math.isclose(float(r['alpha_multiplier']),1): selected['baseline']=r
    keep = {}
    for r in rs:
        m = float(r['alpha_multiplier'])
        if r['setting']=='prior_sweep' and 1e-7*(1-1e-12)<=m<=100*(1+1e-12):
            keep[float(r['effective_precision'])]=r
    for r in selected.values(): keep[float(r['effective_precision'])]=r
    curve = [keep[g] for g in sorted(keep)]
    for ax, metric in zip(axes,['R_sigma_rms','rho']):
        ax.loglog([float(r['effective_precision']) for r in curve],
                  [float(r[metric]) for r in curve],'-',color=colors[seed],lw=1.6,
                  marker='.',ms=3,label=f'Seed {seed}')
        for name, marker in [('baseline','o'),('MLE','D'),('evidence','s')]:
            r = selected[name]
            ax.plot(float(r['effective_precision']),float(r[metric]),marker,
                    color=colors[seed],ms=6,mec=colors[seed],mew=1.3,
                    mfc='white' if name=='evidence' else colors[seed],zorder=4)
for ax in axes:
    ax.set_xlabel(r'Effective precision $\gamma=\alpha v$')
    ax.grid(True,which='major',alpha=.18)
axes[0].set_ylabel(r'Uncertainty contrast $R_{\sigma}^{\mathrm{rms}}$')
axes[1].set_ylabel(r'Contrast gap $\rho=R_{\mathrm{tar}}/R_{\sigma}^{\mathrm{rms}}$')
axes[1].axhline(1,color='.3',ls='--',lw=1,zorder=0)
handles = [Line2D([],[],color=colors[s],lw=1.6,label=f'{s}') for s in range(8)]
fig.legend(handles=handles,loc='upper center',ncol=8,title='Burgers–MLP seed',
           frameon=False,bbox_to_anchor=(.49,1.11),handlelength=1.3,columnspacing=.8)
markers = [Line2D([],[],color='.25',marker=m,ls='',ms=6,mfc='white' if m=='s' else '.25',label=l)
           for m,l in [('o','Baseline'),('D','Conditional noise MLE'),('s','Evidence boundary')]]
fig.legend(handles=markers,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.10))
fig.subplots_adjust(wspace=.3,bottom=.18,top=.88,left=.085,right=.995)
for suffix in ['pdf','png']:
    fig.savefig(root/f'figures/round2_contrast_and_gap.{suffix}',bbox_inches='tight',dpi=200)
plt.close(fig)
families = [('burgers','mlp','full'),('burgers','fourier','full'),
            ('allen_cahn','mlp','full'),('allen_cahn','fourier','full'),
            ('burgers','mlp','post_formation'),('burgers','fourier','post_formation')]
for pde, arch, subset in families:
    fr = [r for r in rows if r['pde']==pde and r['arch']==arch and r['subset']==subset and r['region']=='sharp']
    tags = sorted({r['tag'] for r in fr})
    fig, ax = plt.subplot_mosaic([['geometry','trained'],['geometry','affine']],figsize=(8,6.4))
    handles = []
    for tag in tags:
        seed = int(tag.rsplit('seed',1)[1]); color=colors[seed]
        handles.append(Line2D([],[],color=color,lw=1.6,label=f'Seed {seed}'))
        for centre in ['trained','affine']:
            cr = sorted([r for r in fr if r['tag']==tag and r['centre']==centre],key=lambda r:float(r['effective_precision']))
            x = [float(r['effective_precision']) for r in cr]
            ax[centre].loglog(x,[float(r['rho']) for r in cr],'.-',color=color,lw=1.4,ms=3)
            if centre=='trained':
                ax['geometry'].loglog(x,[float(r['R_sigma_rms']) for r in cr],'.-',color=color,lw=1.4,ms=3)
                ax['geometry'].axhline(float(cr[0]['R_tar']),color=color,ls='--',lw=.8,alpha=.65)
    ax['geometry'].set_title('Covariance contrast')
    ax['geometry'].set_ylabel(r'$R_{\sigma}^{\mathrm{rms}}$; dashed: trained $R_{\mathrm{tar}}$')
    for centre in ['trained','affine']:
        ax[centre].axhline(1,color='.3',ls='--',lw=1)
        ax[centre].set_title(centre.capitalize()+' centre')
        ax[centre].set_ylabel(r'$\rho=R_{\mathrm{tar}}/R_{\sigma}^{\mathrm{rms}}$')
    for axis in ax.values():
        axis.grid(True,which='major',alpha=.18)
        axis.set_xlabel(r'Effective precision $\gamma=\alpha v$')
    title = ('Burgers' if pde=='burgers' else 'Allen-Cahn')+' / '+('MLP' if arch=='mlp' else 'Fourier')
    title += ' / '+('all times' if subset=='full' else 'post-formation')
    fig.suptitle(title,y=1.01)
    fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,-.16))
    fig.tight_layout()
    fig.savefig(root/f'figures/round2_{pde}_{arch}_{subset}.pdf',bbox_inches='tight')
    plt.close(fig)
print('Saved publication figure and six appendix figures from reviewed CSVs.')

groups=list(csv.DictReader((review/'evidence_slope_groups.csv').open()))
fig,axes=plt.subplots(1,2,figsize=(8,3.25),sharey=True)
group_order=['lambda_below_gamma/100','lambda_near_gamma','lambda_above_100gamma']
for axis,prefix,title in zip(axes,['affine_evidence_original','affine_evidence_extended'],['Original noise grid','Extended noise grid']):
    rs=[r for r in groups if r['tag']=='burgers_mlp_seed2' and r['setting'].startswith(prefix)]
    rs=[next(r for r in rs if r['group']==g) for g in group_order]
    x=np.arange(3);w=.36
    axis.bar(x-w/2,[float(r['mean_projected_data_energy']) for r in rs],w,label=r'Mean $z_i^2$',color='#2979b6')
    axis.bar(x+w/2,[float(r['mean_model_variance']) for r in rs],w,label=r'Mean $v+\lambda_i/\alpha$',color='#e58b2a')
    axis.set_yscale('log');axis.set_title(title)
    axis.set_xticks(x,[f'$<0.01$\n$n={rs[0]["count"]}$',f'$0.01$–$100$\n$n={rs[1]["count"]}$',f'$>100$\n$n={rs[2]["count"]}$'])
    axis.set_xlabel(r'Mode group: $\lambda_i/\gamma$')
    axis.grid(True,axis='y',which='major',alpha=.16)
axes[0].set_ylabel('Mean energy / variance')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,1.07))
fig.tight_layout()
for suffix in ['pdf','png']:
    fig.savefig(root/f'figures/round2_evidence_energy_reference.{suffix}',bbox_inches='tight',dpi=200)
plt.close(fig)
