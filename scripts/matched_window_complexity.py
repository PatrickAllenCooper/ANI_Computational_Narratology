"""Matched-character-budget text-complexity contrast, NoT (narrative_cot) vs CoT.

Source: divergence_study_outputs/elephant_singleagent_raw.csv
  arms: standard_cot, narrative_cot (+ standard_cot_verbose on oeq)
  datasets kept: oeq, ss, aita_yta, flip_pairs_free  (flip_pairs dropped: the
  binary-verdict cell has 3-character CoT responses, i.e. no text to measure)
Every proxy is recomputed on a FIXED CHARACTER BUDGET W taken identically from
both arms, at two window positions (head = chars [0,W), mid = centred).
No regression on length anywhere.
"""
import gzip, lzma, zlib, math, sys, json
from collections import Counter
import numpy as np, pandas as pd
import zstandard as zstd, brotli

P = '/Users/pat/code/ANI_Examination/divergence_study_outputs/'
OUT = '/private/tmp/claude-501/-Users-pat-code-ANI-Examination/ede5d137-6350-4382-b68a-12e3be634c88/scratchpad/'
RNG = np.random.default_rng(20260822)

ZC = zstd.ZstdCompressor(level=22, write_content_size=False, write_checksum=False)

def ratios(s: str):
    b = s.encode('utf-8'); n = len(b)
    return dict(
        gzip=len(gzip.compress(b, 9))/n,
        lzma=len(lzma.compress(b, preset=9))/n,
        zstd22=len(ZC.compress(b))/n,
        brotli=len(brotli.compress(b, quality=11))/n,
    )

def ngram_entropy(s, k):
    if len(s) < k: return np.nan
    c = Counter(s[i:i+k] for i in range(len(s)-k+1))
    tot = sum(c.values())
    return -sum((v/tot)*math.log2(v/tot) for v in c.values())

WORD = None
import re
_wre = re.compile(r"[A-Za-z']+")
_sre = re.compile(r'[.!?]+(?:\s|$)')

def mattr(words, w=50):
    if len(words) < w: 
        return len(set(words))/len(words) if words else np.nan
    vals = [len(set(words[i:i+w]))/w for i in range(len(words)-w+1)]
    return float(np.mean(vals))

def feats(s: str):
    f = ratios(s)
    H = {k: ngram_entropy(s, k) for k in (1,2,3,4,5,6)}
    f['h_char1'] = H[1]; f['h_char3'] = H[3]; f['h_char5'] = H[5]
    # conditional entropy rate estimates (bits/char): h_k = H_k - H_{k-1}
    f['hrate_k3'] = H[3]-H[2]
    f['hrate_k5'] = H[5]-H[4]
    f['hrate_k6'] = H[6]-H[5]
    words = _wre.findall(s.lower())
    f['n_words'] = len(words)
    f['mattr50'] = mattr(words, 50)
    f['ttr'] = len(set(words))/len(words) if words else np.nan
    f['mean_word_len'] = float(np.mean([len(w) for w in words])) if words else np.nan
    sents = [x for x in _sre.split(s) if x.strip()]
    f['mean_sent_words'] = (len(words)/len(sents)) if sents else np.nan
    f['comma_per_word'] = s.count(',')/len(words) if words else np.nan
    # cheap clause-embedding proxy: subordinator+relativiser rate per word
    sub = sum(1 for w in words if w in {'that','which','who','whom','whose','because','although',
        'though','while','whereas','if','unless','until','when','whenever','where','since','so','as','before','after','whether'})
    f['subord_per_word'] = sub/len(words) if words else np.nan
    return f

def window(s, W, pos):
    L = len(s)
    if L < W: return None
    if pos == 'head': return s[:W]
    st = (L - W)//2
    return s[st:st+W]

def _job(a):
    resp, W, pos, gen, arm, item, ds, nchar = a
    d = feats(window(resp, W, pos))
    d.update(dict(W=W, pos=pos, gen=gen, arm=arm, item=item, dataset=ds, nchar=nchar))
    return d


def main():
    el = pd.read_csv(P+'elephant_singleagent_raw.csv')
    keep_ds = ['oeq','ss','aita_yta','flip_pairs_free']
    el = el[el.dataset.isin(keep_ds)].copy()
    el = el[el.arm.isin(['standard_cot','narrative_cot','standard_cot_verbose'])]
    el['response'] = el['response'].fillna('')
    el = el[~el.empty_response.fillna(False).astype(bool)]
    el['nchar'] = el.response.str.len()
    el = el[el.nchar > 0]
    el['item'] = el.dataset + '::' + el.item_id.astype(str)
    print('docs after filter:', len(el))
    print(el.groupby(['arm']).size())

    Ws = [300, 500, 800, 1200, 1600, 2400]
    jobs = []
    for W in Ws:
        for pos in ('head','mid'):
            sub = el[el.nchar >= W]
            for r in sub.itertuples():
                jobs.append((r.response, W, pos, r.generator, r.arm, r.item, r.dataset, r.nchar))
    print('jobs', len(jobs)); sys.stdout.flush()
    from multiprocessing import Pool
    with Pool(14) as pool:
        recs = pool.map(_job, jobs, chunksize=200)
    df = pd.DataFrame(recs)
    df.to_parquet(OUT+'matched_window_feats.parquet') if False else df.to_csv(OUT+'matched_window_feats.csv', index=False)
    print('features written', df.shape)

if __name__ == '__main__':
    main()
