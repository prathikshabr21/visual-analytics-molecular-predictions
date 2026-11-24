from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='Understanding Molecular Predictions', page_icon='🧪', layout='wide', initial_sidebar_state='expanded')
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'experiments/dashboard_data'
PROPS = ['Molecular Weight','Polar Surface Area','Number of Rings','Number of Rotatable Bonds','Number of H-Bond Donors','Minimum Degree','LogP','Number of H-Bond Acceptors','Number of Aromatic Rings','Heavy Atom Count','Fraction Csp3','Number of Heteroatoms','Molar Refractivity','Number of Atoms','Number of Bonds']
SPLITS = ['Training','Validation','Testing']
COLORS = {'Training':'#2563eb','Validation':'#d97706','Testing':'#0d9488'}
@st.cache_data
def load():
    m = pd.read_csv(DATA/'all_molecules.csv')
    p = pd.read_csv(DATA/'all_split_predictions.csv')
    metrics = pd.read_csv(DATA/'all_split_metrics.csv')
    d = p.merge(m,on=['molecule_id','split'],validate='many_to_one')
    assert len(m)==1128 and len(d)==7896 and len(metrics)==21
    assert d['Compound ID'].notna().all()
    return m,d,metrics
try:
    m,d,metrics=load()
except Exception as e:
    st.error(f'Unable to load dashboard data: {e}'); st.stop()

def chart(fig, **kwargs):
    fig.update_layout(margin=dict(l=15,r=15,t=35,b=15),legend_title_text='')
    return st.plotly_chart(fig,width="stretch",**kwargs)

def select():
    a,b,c=st.columns(3)
    model=a.selectbox('Model',['GCN','GAT','GINE'],index=1)
    seed=b.selectbox('Training seed',sorted(d.loc[d.model==model,'seed'].unique()))
    split=c.selectbox('Dataset partition',SPLITS,index=2)
    return d[(d.model==model)&(d.seed==seed)&(d.split==split)].copy()

def scores(q):
    if q.empty: return
    a,b,c,e=st.columns(4)
    a.metric('Molecules',len(q)); b.metric('RMSE',f'{q.squared_error.mean()**.5:.4f}'); c.metric('MAE',f'{q.absolute_error.mean():.4f}')
    den=((q.actual_logS-q.actual_logS.mean())**2).sum()
    e.metric('R²',f'{1-q.squared_error.sum()/den:.4f}' if den else 'N/A')

def structure(mid):
    path=ROOT/'dashboard/assets/molecules'/f'{int(mid)}.svg'
    if path.exists(): st.image(str(path),width=300)
    else: st.caption('Structure unavailable')

def inspect(q,key):
    if q.empty: return
    ids=q.molecule_id.unique().tolist()
    names=q.drop_duplicates('molecule_id').set_index('molecule_id')['Compound ID'].to_dict()
    mid=st.selectbox('Inspect molecule',ids,format_func=lambda i:f'{i} — {names[i]}',key=key)
    r=q[q.molecule_id==mid].iloc[0]
    a,b=st.columns(2)
    with a: structure(mid)
    with b:
        st.write(r['Compound ID']); st.code(r.smiles)
        st.write(f"Measured log S: {r['Measured logS']:.4f}")
        if 'predicted_logS' in r: st.write(f'Predicted: {r.predicted_logS:.4f} | Absolute error: {r.absolute_error:.4f}')
        with st.expander('Molecular characteristics'): st.dataframe(pd.to_numeric(r[PROPS], errors='coerce').rename('Value').to_frame())
    return mid

def records(q,key):
    order=st.selectbox('Sort predictions',['Highest absolute error','Lowest absolute error','Most overpredicted','Most underpredicted'],key=key+'sort')
    col='absolute_error' if 'absolute' in order else 'error'
    q=q.sort_values(col,ascending=order in ['Lowest absolute error','Most underpredicted'])
    term=st.text_input('Search name, ID or SMILES',key=key+'search')
    if term: q=q[q[['Compound ID','smiles','molecule_id']].astype(str).apply(lambda s:s.str.contains(term,case=False,regex=False)).any(axis=1)]
    st.dataframe(q[['molecule_id','Compound ID','actual_logS','predicted_logS','error','absolute_error']],hide_index=True)
    st.download_button('Download records',q.to_csv(index=False),file_name='predictions.csv',key=key+'download')
    return q

def ranges(q):
    q=q.copy(); q['Range']=pd.cut(q.actual_logS,[-float('inf'),-6,-4,-2,0,float('inf')],labels=['≤ −6','(−6, −4]','(−4, −2]','(−2, 0]','> 0'])
    return q.groupby('Range',observed=True).agg(MAE=('absolute_error','mean'),MSE=('squared_error','mean'),Bias=('error','mean'),Molecules=('molecule_id','size')).reset_index().assign(RMSE=lambda x:x.MSE**.5)

def correlations(q,col):
    return pd.DataFrame([{'Property':p,'Pearson':q[p].corr(q[col]),'Spearman':q[p].rank().corr(q[col].rank())} for p in PROPS]).dropna(subset=['Spearman']).sort_values('Spearman',key=lambda x:x.abs(),ascending=False)

# Compact native radio navigation, grouped by research question.
if "active_page" not in st.session_state:
    st.session_state.active_page = "Project Overview"

NAV_ICONS = {
    "Project Overview": ":material/dashboard:",
    "Dataset Explorer": ":material/database:",
    "Model Performance": ":material/monitoring:",
    "Prediction Analysis": ":material/scatter_plot:",
    "Molecular Property Analysis": ":material/tune:",
    "Molecular Explorer": ":material/biotech:",
    "RQ1 Key Findings": ":material/fact_check:",
    "Experiment Lab": ":material/science:",
}

def nav_label(page):
    return NAV_ICONS[page] + " " + ("Key Findings" if page == "RQ1 Key Findings" else page)

NAV_GROUPS = {
    "nav_overview": ["Project Overview", "Dataset Explorer"],
    "nav_rq1": ["Model Performance", "Prediction Analysis", "Molecular Property Analysis", "Molecular Explorer", "RQ1 Key Findings"],
    "nav_method": ["Experiment Lab"],
}

def change_page(key):
    selected = st.session_state[key]
    if selected is not None:
        st.session_state.active_page = selected
        for other in NAV_GROUPS:
            if other != key:
                st.session_state[other] = None

# Initialize all radio groups before rendering any widget.
for key, options in NAV_GROUPS.items():
    if key not in st.session_state:
        st.session_state[key] = st.session_state.active_page if st.session_state.active_page in options else None

def nav_radio(key):
    st.radio("Navigation", NAV_GROUPS[key], index=None,
             key=key, format_func=nav_label, label_visibility="collapsed",
             on_change=change_page, args=(key,))

with st.sidebar:
    st.markdown("## :material/biotech: Molecular Predictions")
    nav_radio("nav_overview")
    st.markdown("**RQ1 — Where does the model perform well, and where does it fail?**")
    nav_radio("nav_rq1")
    st.markdown("**RQ2 — How well does your assigned explanation method reflect the model’s behavior?**")
    st.caption(":material/schedule: Upcoming phase")
    st.markdown("**RQ3 — What can we learn by investigating individual molecules and their attributions?**")
    st.caption(":material/schedule: Upcoming phase")
    nav_radio("nav_method")
    st.divider()
    st.caption(f":material/database: ESOL · {len(m):,} molecules")
    st.caption(":material/lock: Fixed 80/10/10 random split")

page = st.session_state.active_page

st.title(page)
if page=='Project Overview':
    st.markdown('**Understanding Molecular Predictions** — Where does the model perform well, and where does it fail?')
    t=metrics[metrics.split=='Testing'].groupby('model').agg(RMSE=('rmse','mean'),MAE=('mae','mean'),Runs=('seed','size')).reset_index()
    a,b,c,e=st.columns(4); a.metric('Molecules',len(m)); b.metric('Characteristics',len(PROPS)); c.metric('Candidate model','GAT'); e.metric('GAT mean test RMSE',f"{t.loc[t.model=='GAT','RMSE'].iloc[0]:.4f}")
    st.subheader('Dataset partitions'); counts=m.groupby('split').size().reindex(SPLITS).rename('Molecules').reset_index(); chart(px.bar(counts,x='split',y='Molecules',color='split',color_discrete_map=COLORS,text='Molecules'))
    st.subheader('How accurate are our predictions?'); chart(px.bar(t.melt(id_vars=['model','Runs'],value_vars=['RMSE','MAE'],var_name='Metric',value_name='Error'),x='model',y='Error',color='Metric',barmode='group',hover_data=['Runs']))
    st.caption('Lower is better. GCN: one run; GAT and GINE: three runs each. GAT recommended by mean validation RMSE, pending shared-model discussion.')
    st.write('ESOL → Molecular graphs → GNN training → Evaluation → Visual analytics')
    with st.expander('Methodology and limitations'): st.write('Fixed random split; checkpoint selection by validation RMSE. Alternative splits remain outstanding. Test results are descriptive, not used for further tuning. Associations do not establish causation.')
elif page=='Dataset Explorer':
    splits=st.multiselect('Partitions',SPLITS,default=SPLITS); q=m[m.split.isin(splits)].copy()
    lo,hi=float(m['Measured logS'].min()),float(m['Measured logS'].max()); bounds=st.slider('Measured log S range',lo,hi,(lo,hi)); q=q[q['Measured logS'].between(*bounds)]
    p=st.selectbox('Molecular characteristic',PROPS)
    with st.expander('Advanced property filter'):
        lo,hi=float(m[p].min()),float(m[p].max())
        if lo<hi: q=q[q[p].between(*st.slider(p,lo,hi,(lo,hi)))]
    if q.empty: st.info('No molecules match.'); st.stop()
    a,b=st.columns(2); a.metric('Molecules',len(q)); b.metric('Mean measured log S',f"{q['Measured logS'].mean():.3f}")
    chart(px.histogram(q,x='Measured logS',color='split',barmode='overlay',histnorm='probability',opacity=.6,color_discrete_map=COLORS)); chart(px.box(q,x='split',y='Measured logS',color='split',color_discrete_map=COLORS)); chart(px.histogram(q,x=p,color='split',color_discrete_map=COLORS)); chart(px.scatter(q,x=p,y='Measured logS',color='split',hover_name='Compound ID',color_discrete_map=COLORS))
    term=st.text_input('Search molecules'); q=q[q[['Compound ID','smiles','molecule_id']].astype(str).apply(lambda s:s.str.contains(term,case=False,regex=False)).any(axis=1)]
    st.dataframe(q,hide_index=True); st.download_button('Download dataset selection',q.to_csv(index=False),'molecules.csv'); inspect(q,'dataset')
elif page=='Model Performance':
    metric=st.selectbox('Metric',['rmse','mae','r2']); models=st.multiselect('Models',['GCN','GAT','GINE'],default=['GCN','GAT','GINE']); q=metrics[metrics.model.isin(models)].copy(); q['Run']=q.model+' / '+q.seed.astype(str)
    chart(px.bar(q,x='Run',y=metric,color='split',barmode='group',color_discrete_map=COLORS)); split=st.selectbox('Architecture comparison partition',SPLITS,index=2)
    agg=q[q.split==split].groupby('model')[metric].agg(['mean','std','count']).reset_index(); chart(px.bar(agg,x='model',y='mean',error_y='std',hover_data=['count'])); st.dataframe(agg,hide_index=True)
    st.caption('Error bars show sample standard deviation across seeds; undefined for the single GCN run.')
    with st.expander('Metric definitions'): st.write('RMSE emphasizes large errors; MAE is average absolute error; R² measures improvement over predicting the partition mean. Lower RMSE/MAE and higher R² are better.')
    with st.expander('Complete evaluation metrics'): st.dataframe(q,hide_index=True)
elif page=='Prediction Analysis':
    q=select(); scores(q); threshold=st.slider('Acceptable absolute error (log S)',0.,max(1.,float(q.absolute_error.max())),.5,.05); q['Tolerance']=q.absolute_error.le(threshold).map({True:'Within tolerance',False:'Above tolerance'})
    st.metric('Within tolerance',f'{q.absolute_error.le(threshold).mean():.1%}',f'{q.absolute_error.le(threshold).sum()} of {len(q)} molecules',delta_color='off'); st.caption('Exploratory tolerance, not a universal accuracy standard.')
    fig=px.scatter(q,x='actual_logS',y='predicted_logS',color='Tolerance',hover_name='Compound ID',hover_data=['absolute_error'],custom_data=['molecule_id']); lo=min(q.actual_logS.min(),q.predicted_logS.min()); hi=max(q.actual_logS.max(),q.predicted_logS.max()); fig.add_shape(type='line',x0=lo,y0=lo,x1=hi,y1=hi,line=dict(dash='dash',color='red')); fig.update_yaxes(scaleanchor='x',scaleratio=1)
    event=chart(fig,on_select='rerun',selection_mode=('points','box','lasso'),key='prediction_'+str(q.experiment.iloc[0])+q.split.iloc[0]); ids=[p['customdata'][0] for p in event.selection.points if p.get('customdata')]; inspect(q[q.molecule_id.isin(ids)] if ids else q,'prediction_inspect')
    st.subheader('Underprediction and overprediction distribution'); fig=px.histogram(q,x='error'); fig.add_vline(x=0,line_dash='dash'); chart(fig)
    st.subheader('Prediction bias across solubility levels'); fig=px.scatter(q,x='actual_logS',y='error',hover_name='Compound ID'); fig.add_hline(y=0,line_dash='dash'); chart(fig); st.caption('Signed error = predicted − measured; positive means overprediction.')
    st.subheader('Which solubility ranges are harder to predict?'); r=ranges(q); chart(px.bar(r,x='Range',y='MAE',hover_data=['Molecules','RMSE','Bias'],text='Molecules')); st.dataframe(r.drop(columns='MSE'),hide_index=True)
    st.subheader('Prediction records'); f=st.radio('Tolerance filter',['All','Within tolerance','Above tolerance'],horizontal=True); records(q if f=='All' else q[q.Tolerance==f],'pred')
elif page=='Molecular Property Analysis':
    q=select(); p=st.selectbox('Molecular characteristic',PROPS); col=st.radio('Error measure',['absolute_error','error'],format_func=lambda x:'Absolute error' if x=='absolute_error' else 'Signed error',horizontal=True)
    with st.expander('Advanced filters'):
        lo,hi=float(q[p].min()),float(q[p].max());
        if lo<hi: q=q[q[p].between(*st.slider('Property range',lo,hi,(lo,hi)))]
    if q.empty: st.info('No matching molecules'); st.stop()
    pear=q[p].corr(q[col]); spear=q[p].rank().corr(q[col].rank()); a,b,c=st.columns(3); a.metric('Molecules',len(q)); b.metric('Pearson',f'{pear:.3f}'); c.metric('Spearman',f'{spear:.3f}')
    event=chart(px.scatter(q,x=p,y=col,color='actual_logS',hover_name='Compound ID',custom_data=['molecule_id']),on_select='rerun',key='property_'+p+col+str(q.experiment.iloc[0])+q.split.iloc[0]); ids=[x['customdata'][0] for x in event.selection.points if x.get('customdata')]; inspect(q[q.molecule_id.isin(ids)] if ids else q,'property_inspect')
    st.caption(f'Observed Spearman association: {spear:.3f}. Positive values indicate increasing selected error measure with property rank; correlation is not causation.')
    st.subheader('Average error by property group'); q['Group']=q[p].astype(str) if q[p].nunique()<=8 else pd.qcut(q[p],4,duplicates='drop').astype(str); g=q.groupby('Group',sort=False).agg(Error=(col,'mean'),Molecules=('molecule_id','size')).reset_index(); chart(px.bar(g,x='Group',y='Error',text='Molecules',hover_data=['Molecules'])); st.caption('Bar labels show molecule counts. Continuous properties use approximate quartile groups.')
    st.subheader('Which characteristics are associated with prediction errors?'); r=correlations(q,col); chart(px.bar(r.sort_values('Spearman'),x='Spearman',y='Property',orientation='h',range_x=[-1,1]));
    with st.expander('View correlation values'): st.dataframe(r,hide_index=True)
elif page=='Molecular Explorer':
    q=select(); q=records(q,'explore'); mid=inspect(q,'explore_inspect')
    if mid is not None:
        r=d[d.molecule_id==mid].copy(); r['Run']=r.model+' / '+r.seed.astype(str); st.subheader('Same molecule across checkpoints'); fig=px.bar(r,x='Run',y='predicted_logS'); fig.add_hline(y=r.actual_logS.iloc[0],line_dash='dash'); chart(fig); chart(px.bar(r,x='Run',y='absolute_error')); st.dataframe(r[['Run','actual_logS','predicted_logS','absolute_error']],hide_index=True)
        other=st.selectbox('Compare with another molecule',q.molecule_id.tolist(),format_func=lambda i:f"{i} — {q.loc[q.molecule_id==i,'Compound ID'].iloc[0]}"); a,b=st.columns(2)
        for c,i in [(a,mid),(b,other)]:
            with c:
                r=q[q.molecule_id==i].iloc[0]; st.write(r['Compound ID']); structure(i); st.write(f'Measured {r.actual_logS:.3f} | Predicted {r.predicted_logS:.3f} | Error {r.absolute_error:.3f}')
elif page=='RQ1 Key Findings':
    st.write('Where does the model perform well, and where does it fail?'); q=select(); scores(q)
    st.subheader('1. Prediction accuracy'); st.write(f'{q.absolute_error.le(.5).sum()} of {len(q)} molecules ({q.absolute_error.le(.5).mean():.1%}) are within 0.5 log S; {q.absolute_error.le(1).mean():.1%} within 1 log S. These are exploratory tolerances.')
    st.subheader('2. Solubility ranges'); r=ranges(q); best=r.loc[r.MAE.idxmin()]; worst=r.loc[r.MAE.idxmax()]; st.write(f'Lowest MAE: {best.Range} ({best.MAE:.3f}, n={best.Molecules}); highest MAE: {worst.Range} ({worst.MAE:.3f}, n={worst.Molecules}). Small groups warrant caution.'); chart(px.bar(r,x='Range',y='MAE',text='Molecules'))
    st.subheader('3. Prediction bias'); st.write(f'Mean signed error: {q.error.mean():.4f}. Positive and negative errors can cancel.'); chart(px.bar(r,x='Range',y='Bias',hover_data=['Molecules']))
    st.subheader('4. Molecular characteristics'); r=correlations(q,'absolute_error');
    if not r.empty: st.write(f"Largest observed absolute rank association: {r.iloc[0].Property} (Spearman {r.iloc[0].Spearman:.3f}). Descriptors may be redundant; association is not causation."); chart(px.bar(r.head(8).sort_values('Spearman'),x='Spearman',y='Property',orientation='h',range_x=[-1,1]))
    st.subheader('5. Accurate and inaccurate examples'); a,b=st.columns(2)
    with a: st.write('Lowest errors'); inspect(q.nsmallest(5,'absolute_error'),'best')
    with b: st.write('Highest errors'); inspect(q.nlargest(5,'absolute_error'),'worst')
    st.subheader('6. Consistency across training seeds'); allruns=d[(d.model==q.model.iloc[0])&(d.split==q.split.iloc[0])]; g=allruns.groupby(['molecule_id','Compound ID']).agg(Runs=('seed','nunique'),Mean_error=('absolute_error','mean'),Min_error=('absolute_error','min'),Max_error=('absolute_error','max')).reset_index();
    if g.Runs.max()>1: st.dataframe(g.nlargest(5,'Mean_error'),hide_index=True); st.dataframe(g.nsmallest(5,'Mean_error'),hide_index=True)
    else: st.info('Only one checkpoint available for this architecture.')
    st.caption('Findings describe the selected partition and fixed split; training results are not held-out performance. Shared checkpoint selection remains pending.')
else:
    labels=['Original GCN','GCN comparison','GraphSAGE','GAT baseline','GAT 300 epochs','GAT hidden 128','GAT lower LR','GAT dropout 0.1','GAT 500 epochs','GAT sum pooling','GAT max pooling','GAT combined pooling','GAT enhanced atoms','EdgeGAT','GINE','GAT seed 123','GAT seed 2026','GINE seed 123','GINE seed 2026']
    rm=[1.2395,1.2257,1.1244,.9394,.8799,.9814,.9213,.7986,.7986,.8096,.8721,.7346,.7887,.7719,.7401,.7491,.7275,.7707,.7048]; ma=[.9700,.9477,.8248,.6601,.6168,.7098,.6621,.5891,.5891,.6021,.6322,.5373,.5760,.5741,.5361,.5298,.5230,.5518,.4907]; e=pd.DataFrame({'Experiment':[f'E{i:02}' for i in range(19)],'Description':labels,'Validation RMSE':rm,'Validation MAE':ma})
    a,b,c,f=st.columns(4); a.metric('Architectures',5); b.metric('Experiments',19); c.metric('Best validation RMSE',min(rm)); f.metric('Best validation MAE',min(ma)); chart(px.line(e,x='Experiment',y=['Validation RMSE','Validation MAE'],markers=True,hover_data=['Description'])); st.write('Combined pooling improved the tested GAT configuration. Extra epochs and enhanced atom features did not improve the corresponding comparisons. GAT is the recommended candidate by mean validation RMSE across three seeds, not the best single run.')
    with st.expander('Advanced experiment details'):
        st.dataframe(e,hide_index=True); files=sorted(p for p in (ROOT/'experiments').glob('*_history.csv') if '_e2_' not in p.name)
        if files:
            path=st.selectbox('Training history',files,format_func=lambda p:p.stem.replace('_history','')); h=pd.read_csv(path); cols=[c for c in ['train_rmse','val_rmse','train_mae','val_mae'] if c in h];
            if cols: chart(px.line(h,x='epoch',y=cols))
            summary=path.with_name(path.name.replace('_history.csv','_summary.csv'))
            if summary.exists(): st.dataframe(pd.read_csv(summary).T.astype('string').rename(columns={0: 'Value'}))
    st.caption('Alternative data split comparisons remain outstanding.')
