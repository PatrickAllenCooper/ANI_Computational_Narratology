"""Same matched-window measurement on two text normalisations, to separate
'complexity' from the scaffold's literal markdown template."""
import re, sys, pandas as pd, numpy as np
sys.path.insert(0,'/private/tmp/claude-501/-Users-pat-code-ANI-Examination/ede5d137-6350-4382-b68a-12e3be634c88/scratchpad')
from matched_window import feats, window, P, OUT

_md = re.compile(r'(^[ \t]*(#{1,6}|[-*+>]|\d+[.)])[ \t]*)|[*_`~|]|\[|\]|\(|\)', re.M)
_ws = re.compile(r'\s+')
_alpha = re.compile(r"[A-Za-z']+")

def strip_fmt(s):
    return _ws.sub(' ', _md.sub(' ', s)).strip()

def words_only(s):
    return ' '.join(w.lower() for w in _alpha.findall(s))

def _job(a):
    txt, W, pos, gen, arm, item, ds, var = a
    d = feats(window(txt, W, pos))
    d.update(dict(W=W, pos=pos, gen=gen, arm=arm, item=item, dataset=ds, variant=var))
    return d

def main():
    el = pd.read_csv(P+'elephant_singleagent_raw.csv')
    el = el[el.dataset.isin(['oeq','ss','aita_yta','flip_pairs_free'])]
    el = el[el.arm.isin(['standard_cot','narrative_cot','standard_cot_verbose'])].copy()
    el['response']=el.response.fillna('')
    el = el[~el.empty_response.fillna(False).astype(bool)]
    el = el[el.response.str.len()>0]
    el['item']=el.dataset+'::'+el.item_id.astype(str)
    el['fmt']=el.response.map(strip_fmt)
    el['wrd']=el.response.map(words_only)
    jobs=[]
    for W in (500,800,1200):
        for pos in ('head','mid'):
            for var,col in (('fmt_stripped','fmt'),('words_only','wrd')):
                sub = el[el[col].str.len()>=W]
                for r in sub.itertuples():
                    jobs.append((getattr(r,col), W, pos, r.generator, r.arm, r.item, r.dataset, var))
    print('jobs',len(jobs)); sys.stdout.flush()
    from multiprocessing import Pool
    with Pool(14) as pool:
        recs = pool.map(_job, jobs, chunksize=200)
    pd.DataFrame(recs).to_csv(OUT+'matched_window_feats_norm.csv', index=False)
    print('OKDONE')

if __name__=='__main__':
    main()
