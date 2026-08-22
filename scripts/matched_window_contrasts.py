import numpy as np, pandas as pd, json, sys
S='/private/tmp/claude-501/-Users-pat-code-ANI-Examination/ede5d137-6350-4382-b68a-12e3be634c88/scratchpad/'
df = pd.read_csv(S+'matched_window_feats.csv')
PROX = ['gzip','lzma','zstd22','brotli','h_char3','h_char5','hrate_k3','hrate_k5','hrate_k6',
        'mattr50','ttr','mean_word_len','mean_sent_words','comma_per_word','subord_per_word','n_words']
B = 2000
rng = np.random.default_rng(7)

def contrast(sub, armA, armB, tag, boot=True):
    """paired armB - armA within item, per generator + pooled (equal gen weight)."""
    out=[]
    a = sub[sub.arm==armA].set_index(['gen','item'])
    b = sub[sub.arm==armB].set_index(['gen','item'])
    common = a.index.intersection(b.index)
    a=a.loc[common]; b=b.loc[common]
    d = (b[PROX]-a[PROX])
    d['gen']=[i[0] for i in common]; d['item']=[i[1] for i in common]
    sdA = a[PROX].groupby(level=0).std()
    gens = sorted(d.gen.unique())
    items = sorted(d.item.unique())
    # per-generator
    per = {}
    for g in gens:
        dg = d[d.gen==g]
        per[g] = dg
    # bootstrap over item clusters (shared across generators)
    idx_by_item = {it: d.index[d.item==it] for it in items} if boot else None
    itemarr = np.array(items)
    # build per-gen arrays keyed by item for fast bootstrap
    mats = {g: per[g].set_index('item')[PROX] for g in gens}
    boots = {g: [] for g in gens}; boots['POOLED']=[]
    if boot:
        for _ in range(B):
            samp = rng.choice(itemarr, size=len(itemarr), replace=True)
            gm=[]
            for g in gens:
                m = mats[g]
                sel = m.reindex(samp).dropna(how='all')
                mu = sel.mean().values
                boots[g].append(mu); gm.append(mu)
            boots['POOLED'].append(np.nanmean(np.vstack(gm),axis=0))
    for g in gens+['POOLED']:
        if g=='POOLED':
            est = np.nanmean(np.vstack([mats[x].mean().values for x in gens]),axis=0)
            n = int(np.mean([len(mats[x]) for x in gens]))
            sd = sdA.mean().values
        else:
            est = mats[g].mean().values; n=len(mats[g]); sd = sdA.loc[g,PROX].values
        bs = np.vstack(boots[g]) if boot else None
        for j,p in enumerate(PROX):
            lo,hi = (np.nanpercentile(bs[:,j],[2.5,97.5]) if boot else (np.nan,np.nan))
            out.append(dict(tag=tag, gen=g, proxy=p, n_items=n, diff=est[j], lo=lo, hi=hi,
                            sd_ref=sd[j], dz=est[j]/sd[j] if sd[j] and not np.isnan(sd[j]) else np.nan))
    return pd.DataFrame(out)

res=[]
for W in sorted(df.W.unique()):
    for pos in ['head','mid']:
        sub = df[(df.W==W)&(df.pos==pos)&(df.arm.isin(['standard_cot','narrative_cot']))]
        r = contrast(sub,'standard_cot','narrative_cot',f'W{W}_{pos}')
        r['W']=W; r['pos']=pos; r['comp']='NoT_vs_CoT'
        res.append(r)
        print('done', W, pos); sys.stdout.flush()
R=pd.concat(res); R.to_csv(S+'mw_contrasts.csv', index=False)

# NoT vs verbose CoT, oeq only
res2=[]
for W in sorted(df.W.unique()):
    for pos in ['head','mid']:
        sub = df[(df.W==W)&(df.pos==pos)&(df.dataset=='oeq')&(df.arm.isin(['standard_cot_verbose','narrative_cot']))]
        if sub.arm.nunique()<2: continue
        r = contrast(sub,'standard_cot_verbose','narrative_cot',f'V_W{W}_{pos}')
        r['W']=W; r['pos']=pos; r['comp']='NoT_vs_verboseCoT'
        res2.append(r)
R2=pd.concat(res2); R2.to_csv(S+'mw_contrasts_verbose.csv', index=False)

# retention
ret = (df[df.arm.isin(['standard_cot','narrative_cot'])]
       .groupby(['W','pos','gen','arm']).item.nunique().unstack('arm'))
ret.to_csv(S+'mw_retention.csv')
print(ret.head(20))
print('OK')
