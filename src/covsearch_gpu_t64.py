"""GPU weighted local search for covering designs C(v,k,t): find nb k-subsets of {1..v} covering every t-subset.

Move rule (differs from covsearch_gpu_t.py in the focused move, see below):
  one warp (32 GPU threads) = one independent search;
  each step: pick a random uncovered t-set q; among the blocks that share the MOST points with q
  (focused move), evaluate every swap (block b, x out of b, y in q minus b);
  take the best (random tie-break); if it does not improve, +1 weight on every uncovered t-set;
  tabu[b,x] for `tenure` steps; optional degree floor dmin (no point drops below dmin blocks).
  Host side: kicks for searches that stall for `stall_s` seconds.
Limits: 2 <= t <= 7; t < k < v <= 64; nb <= 4000. Coverage counters are int8: a t-set covered more than 127 times
would overflow. This is impossible for nb <= 127; for larger nb, run() checks the GPU counters after every launch and
the CPU recount of every final design, and reports result['cnt_overflow'] (with a warning) if it happens.
Changes from covsearch_gpu_t.py:
  - focused move: only blocks with maximal overlap with the chosen uncovered t-set are considered
    (covsearch_gpu_t.py considers every block);
  - points up to 64 (64-bit block-membership mask), t up to 7, nb up to 4000 with the overflow check above;
  - chunked CPU recount (bounded host memory) of every final design.

Usage
    import covsearch_gpu_t64 as CG
    P = CG.Problem(35, 11, 3)                    # compiles the kernel for v=35, k=11, t=3
    res = CG.run(P, R=2048, nb=52, dmin=16, seconds=60, seed=4)
    res['final'][i]  -> per-search dict(missing, blocks (1-based), signature, start)
All reported 'missing' values are recomputed independently on the CPU from the blocks, not taken from the GPU counters.
Environment: see environment.yml (numpy + CuPy for CUDA 12)."""
import os, glob, site, itertools, json, time
for _sp in site.getsitepackages():
    for _d in glob.glob(os.path.join(_sp, 'nvidia', '*', 'bin')):
        os.add_dll_directory(_d); os.environ['PATH'] = _d + os.pathsep + os.environ['PATH']
import numpy as np
import cupy as cp
from math import comb
from collections import Counter

_KERNEL = r'''
#define V  __V__
#define K  __K__
#define T  __T__
#define NT __NT__
#define NSUB __NSUB__
#define FULL 0xffffffffu
#define BW __BW__
__constant__ int BIN[(V+1)*BW];
__device__ __forceinline__ unsigned xs(unsigned &s){ s^=s<<13; s^=s>>17; s^=s<<5; return s; }
__device__ __forceinline__ int rnk(const int* a0, int extra){
  int a[T]; for(int i=0;i<T-1;i++) a[i]=a0[i]; a[T-1]=extra;
  for(int i=1;i<T;i++){ int x=a[i]; int j=i-1; while(j>=0 && a[j]>x){ a[j+1]=a[j]; j--; } a[j+1]=x; }
  int r=0; for(int i=0;i<T;i++) r+=BIN[a[i]*BW+i+1]; return r;
}
extern "C" __global__ void wls(signed char* cnt, unsigned short* w, signed char* cards, int* deg, int* tabu,
    unsigned* rng, int* nunc_a, int* best_a, signed char* bestcards, int* itc, const signed char* UNR, const unsigned char* SUB,
    int R, int nb, int dmin, int steps, int tenure)
{
  int gw = (blockIdx.x*blockDim.x + threadIdx.x) >> 5; int lane = threadIdx.x & 31;
  if (gw >= R) return;
  signed char* C = cnt + (size_t)gw*NT; unsigned short* W = w + (size_t)gw*NT;
  signed char* KD = cards + (size_t)gw*nb*K; int* D = deg + gw*V; int* TB = tabu + (size_t)gw*nb*V;
  signed char* BC = bestcards + (size_t)gw*nb*K;
  unsigned sh = rng[gw] | 1u; unsigned s = (rng[gw]*2654435761u) ^ (lane*40503u + 12345u); if(!s) s=1;
  int nunc = nunc_a[gw], best = best_a[gw], it0 = itc[gw];
  int t;
  for (t=0; t<steps; t++){
    int it = it0 + t;
    if (nunc==0) break;
    int cz=0; for(int i=lane;i<NT;i+=32) cz += (C[i]==0);
    int pre=cz; for(int o=1;o<32;o<<=1){ int v=__shfl_up_sync(FULL,pre,o); if(lane>=o) pre+=v; }
    int total=__shfl_sync(FULL,pre,31);
    int k = xs(sh) % total;
    int q=-1;
    if (k>=pre-cz && k<pre){ int j=k-(pre-cz); for(int i=lane;i<NT;i+=32){ if(C[i]==0){ if(j==0){q=i;break;} j--; } } }
    unsigned bal=__ballot_sync(FULL,q>=0); q=__shfl_sync(FULL,q,__ffs(bal)-1);
    int qp[T]; for(int i=0;i<T;i++) qp[i]=UNR[T*q+i];
    // focused move: only blocks with maximal overlap with q; lanes share the subset loop
    int myov=-1;
    for(int bb2=lane; bb2<nb; bb2+=32){ const signed char* cd=KD+bb2*K; int o=0; for(int e=0;e<K;e++) for(int i=0;i<T;i++) if(cd[e]==qp[i]) o++; if(o>myov) myov=o; }
    int maxov=myov; for(int o=16;o>0;o>>=1) maxov=max(maxov,__shfl_xor_sync(FULL,maxov,o));
    int bestd=0x7fffffff, bb=-1,bx=-1,by=-1, ties=0;
    for(int b2=0;b2<nb;b2++){
      const signed char* card=KD+b2*K;
      int o=0; unsigned long long m=0; for(int e=0;e<K;e++){ m|=1ull<<card[e]; for(int i=0;i<T;i++) if(card[e]==qp[i]) o++; }
      if(o!=maxov || o>=T) continue;
      int ys[T]; int ny=0; for(int i=0;i<T;i++) if(!((m>>qp[i])&1)) ys[ny++]=qp[i];
      for(int xi=0; xi<K; xi++){
        int x=card[xi]; bool inq=false; for(int i=0;i<T;i++) if(x==qp[i]) inq=true;
        if(inq) continue;
        if(dmin>0 && D[x]-1<dmin) continue;
        int rest[K-1]; int r=0; for(int e=0;e<K;e++) if(e!=xi) rest[r++]=card[e];
        int loss=0; int g[T]; for(int i=0;i<T;i++) g[i]=0; int a[T-1];
        for(int tr=lane; tr<NSUB; tr+=32){
          for(int i=0;i<T-1;i++) a[i]=rest[SUB[(T-1)*tr+i]];
          int ix=rnk(a,x); if(C[ix]==1) loss+=W[ix];
          for(int j=0;j<ny;j++){ int iy=rnk(a,ys[j]); if(C[iy]==0) g[j]+=W[iy]; }
        }
        for(int o2=16;o2>0;o2>>=1){ loss+=__shfl_xor_sync(FULL,loss,o2); for(int j=0;j<T;j++) g[j]+=__shfl_xor_sync(FULL,g[j],o2); }
        bool tb = TB[b2*V+x] > it;
        for(int j=0;j<ny;j++){
          int d=loss-g[j];
          if (tb && d>=0) continue;
          if (d<bestd){ bestd=d; bb=b2; bx=x; by=ys[j]; ties=1; }
          else if (d==bestd){ ties++; unsigned rr=__shfl_sync(FULL,xs(sh),0); if(rr%ties==0){ bb=b2; bx=x; by=ys[j]; } }
        }
      }
    }
    int mind=bestd;
    if (bb<0) continue;
    if (mind>=0){ for(int i=lane;i<NT;i+=32) if(C[i]==0 && W[i]<65000) W[i]++; }
    __syncwarp();
    signed char* card=KD+bb*K;
    int rest[K-1]; int r=0; for(int e=0;e<K;e++) if(card[e]!=bx) rest[r++]=card[e];
    int dn=0; int a[T-1];
    for(int tr=lane; tr<NSUB; tr+=32){
      for(int i=0;i<T-1;i++) a[i]=rest[SUB[(T-1)*tr+i]];
      int ix=rnk(a,bx); int c=C[ix]-1; C[ix]=(signed char)c; if(c==0) dn++;
      int iy=rnk(a,by); if(C[iy]==0) dn--; C[iy]=(signed char)(C[iy]+1);
    }
    for(int o=16;o>0;o>>=1) dn+=__shfl_xor_sync(FULL,dn,o);
    nunc+=dn;
    __syncwarp();
    if(lane==0){ for(int e=0;e<K;e++) if(card[e]==bx) card[e]=(signed char)by; D[bx]--; D[by]++; TB[bb*V+bx]=it+tenure; }
    __syncwarp();
    if (nunc<best){ best=nunc; for(int i=lane;i<nb*K;i+=32) BC[i]=KD[i]; }
    __syncwarp();
  }
  if(lane==0){ nunc_a[gw]=nunc; best_a[gw]=best; itc[gw]=it0+t; rng[gw]=sh; }
}
'''


class Problem:
    """Tables, compiled kernel and exact CPU helpers for C(v, k, t), 2 <= t <= 7, v <= 64."""
    def __init__(self, v, k, t=4):
        assert 2 <= t <= 7 and t < k < v <= 64, "needs 2<=t<=7, t<k<v<=64"
        self.v, self.k, self.t, self.nt = v, k, t, comb(v, t)
        self.BW = t + 1
        self.BIN = np.array([[comb(a, c) if c <= a else 0 for c in range(self.BW)] for a in range(v + 1)], dtype=np.int32)
        self.SUB = np.array(list(itertools.combinations(range(k - 1), t - 1)), dtype=np.uint8)
        self.POS = np.array(list(itertools.combinations(range(k), t)), dtype=np.int64)
        self.UNR = np.zeros((self.nt, t), dtype=np.int8)
        for q in itertools.combinations(range(v), t):
            self.UNR[self.rank(np.array(q))] = q
        code = (_KERNEL.replace('__V__', str(v)).replace('__K__', str(k)).replace('__T__', str(t))
                .replace('__NT__', str(self.nt)).replace('__BW__', str(self.BW)).replace('__NSUB__', str(len(self.SUB))))
        mod = cp.RawModule(code=code, options=('-std=c++11',))
        b = np.ascontiguousarray(self.BIN.ravel())
        cp.cuda.runtime.memcpy(mod.get_global('BIN').ptr, b.ctypes.data, b.nbytes, cp.cuda.runtime.memcpyHostToDevice)
        self.mod = mod; self.kern = mod.get_function('wls'); self.UNR_d = cp.asarray(self.UNR.ravel())
        self.SUB_d = cp.asarray(np.ascontiguousarray(self.SUB.ravel()))

    def rank(self, a):  # (..., t) ascending, 0-based -> colex rank
        return sum(self.BIN[a[..., i], i + 1] for i in range(self.t))

    def counts(self, cards):  # chunked (v<=64 variant) to bound host memory; int16 counts
        R = cards.shape[0]; out = np.empty((R, self.nt), np.int16)
        per = cards.shape[1] * len(self.POS) * 8 * 3 + self.nt * 8
        ch = max(1, int(3e8 // per))
        for a in range(0, R, ch):
            c = cards[a:a + ch]; r = c.shape[0]
            idx = self.rank(np.sort(c, axis=2)[:, :, self.POS]).reshape(r, -1) + (np.arange(r) * self.nt)[:, None]
            out[a:a + r] = np.bincount(idx.ravel(), minlength=r * self.nt).reshape(r, self.nt)
        return out

    def verify(self, blocks0):
        cov = set()
        for b in blocks0:
            cov.update(itertools.combinations(sorted(int(e) for e in b), self.t))
        return self.nt - len(cov)

    def holes(self, blocks0):
        m = self.counts(np.asarray(blocks0)[None])[0]
        return [tuple(int(x) for x in self.UNR[i]) for i in np.where(m == 0)[0]]

    def drop_one(self, blocks0):
        c = np.asarray(blocks0); m = self.counts(c[None])[0]
        idx = self.rank(np.sort(c, axis=1)[:, self.POS])
        return (m[idx] == 1).sum(1)

    def signature(self, blocks0):
        c = np.asarray(blocks0)
        deg = tuple(sorted(np.bincount(c.ravel(), minlength=self.v).tolist(), reverse=True))
        sets = [set(int(e) for e in x) for x in c]
        ov = tuple(sorted(Counter(len(a & b) for a, b in itertools.combinations(sets, 2)).items()))
        m = self.counts(c[None])[0]; mult = tuple(sorted(Counter(m.tolist()).items()))
        h = [tuple(int(x) for x in self.UNR[i]) for i in np.where(m == 0)[0]]
        hp = tuple(sorted(Counter(x for q in h for x in q).values(), reverse=True))
        hpp = tuple(sorted(Counter(p for q in h for p in itertools.combinations(q, 2)).values(), reverse=True))
        return dict(missing=len(h), degrees=deg, overlaps=ov, multiplicity=mult, hole_points=hp, hole_pairs=hpp)

    def balanced_start(self, nb, rng):
        deg = np.zeros(self.v, dtype=np.int64); blocks = []
        for _ in range(nb):
            order = np.lexsort((rng.random(self.v), deg)); blk = np.sort(order[:self.k]); deg[blk] += 1; blocks.append(blk)
        return np.array(blocks)

    def kick(self, blocks, dmin, ncards, nswap, rng):
        blocks = blocks.copy(); deg = np.bincount(blocks.ravel(), minlength=self.v)
        for b in rng.choice(blocks.shape[0], size=min(ncards, blocks.shape[0]), replace=False):
            card = set(int(e) for e in blocks[b]); outside = [e for e in range(self.v) if e not in card]
            removable = [e for e in card if dmin == 0 or deg[e] - 1 >= dmin]
            rng.shuffle(removable); rng.shuffle(outside)
            for x, y in zip(removable[:nswap], outside[:nswap]):
                card.remove(x); card.add(y); deg[x] -= 1; deg[y] += 1
            blocks[b] = np.array(sorted(card))
        return blocks


def sig_key(s):
    """Hashable key of a signature dict (used to group results into shapes)."""
    return json.dumps([s['missing'], s['degrees'], s['hole_points'], s['hole_pairs'], s['overlaps'], s['multiplicity']])


def degstr(deg):
    return ' + '.join(f"{n}x{d}" for d, n in sorted(Counter(deg).items()))


class Swarm:
    def __init__(self, P, R, nb, dmin, seed, tenure=40, starts=None):
        assert nb <= 4000  # cnt is int8: overflow is impossible for nb <= 127; for larger nb run() checks it
        self.P, self.R, self.nb, self.dmin, self.tenure = P, R, nb, dmin, tenure
        self.rng = np.random.default_rng(seed)
        cards = (np.stack([P.balanced_start(nb, self.rng) for _ in range(R)]) if starts is None
                 else np.stack([np.asarray(starts[i % len(starts)]) for i in range(R)]))
        self._upload(cards)
        self.rng_d = cp.asarray(self.rng.integers(1, 2**32 - 1, R, dtype=np.uint64).astype(np.uint32))
        self.itc = cp.zeros(R, dtype=cp.int32)
        self.best_all = np.full(R, 10**9); self.best_cards = cards.copy(); self.last_imp = np.zeros(R)
        self.kicks = np.zeros(R, int); self.kick_cards = np.full(R, 2)

    def _upload(self, cards):
        P, R = self.P, self.R
        cnt = P.counts(cards)
        self.cards = cp.asarray(cards.astype(np.int8)); self.cnt = cp.asarray(cnt.astype(np.int8))
        self.w = cp.ones((R, P.nt), dtype=cp.uint16)
        self.deg = cp.asarray(np.stack([np.bincount(c.ravel(), minlength=P.v) for c in cards]).astype(np.int32))
        self.tabu = cp.zeros((R, self.nb, P.v), dtype=cp.int32)
        nunc = (cnt == 0).sum(1).astype(np.int32)
        self.nunc = cp.asarray(nunc); self.best = cp.asarray(nunc.copy()); self.bestcards = self.cards.copy()

    def reset_rows(self, rows, cards_rows):
        P = self.P; cnt = P.counts(cards_rows); r = cp.asarray(rows)
        self.cards[r] = cp.asarray(cards_rows.astype(np.int8)); self.bestcards[r] = self.cards[r]
        self.cnt[r] = cp.asarray(cnt.astype(np.int8)); self.w[r] = 1; self.tabu[r] = 0; self.itc[r] = 0
        self.deg[r] = cp.asarray(np.stack([np.bincount(c.ravel(), minlength=P.v) for c in cards_rows]).astype(np.int32))
        nu = cp.asarray((cnt == 0).sum(1).astype(np.int32)); self.nunc[r] = nu; self.best[r] = nu

    def step(self, steps):
        threads = 128; blocks = (self.R * 32 + threads - 1) // threads
        self.P.kern((blocks,), (threads,), (self.cnt, self.w, self.cards, self.deg, self.tabu, self.rng_d, self.nunc,
                    self.best, self.bestcards, self.itc, self.P.UNR_d, self.P.SUB_d, np.int32(self.R), np.int32(self.nb),
                    np.int32(self.dmin), np.int32(steps), np.int32(self.tenure)))
        cp.cuda.Device().synchronize()


def run(P, R, nb, dmin, seconds, seed, starts=None, start_labels=None, stall_s=60.0, launch_steps=200,
        stop_at_zero=True, progress_file=None, log_every=30.0, kicks=True):
    """Run R searches for `seconds`. Returns a dict with per-search results (independently verified).
    Each GPU launch is kept short (< ~1 s) because Windows resets the GPU if one kernel runs > ~2 s."""
    S = Swarm(P, R, nb, dmin, seed, starts=starts)
    t0 = time.time(); S.last_imp[:] = t0; hist = []; last = t0; moves = 0
    first_solve = np.full(R, np.nan); overflow = False
    while time.time() - t0 < seconds:
        S.step(launch_steps); moves += launch_steps * R; now = time.time()
        if nb > 127 and not overflow and int(S.cnt.min()) < 0: overflow = True  # int8 wrap-around (count > 127)
        best = cp.asnumpy(S.best); imp = best < S.best_all
        if imp.any():
            S.best_cards[imp] = cp.asnumpy(S.bestcards[cp.asarray(np.where(imp)[0])])
            S.best_all[imp] = best[imp]; S.last_imp[imp] = now; S.kick_cards[imp] = 2
        newz = (S.best_all == 0) & np.isnan(first_solve); first_solve[newz] = now - t0
        if stop_at_zero and (S.best_all == 0).any(): break
        if not stop_at_zero and (S.best_all == 0).all(): break
        stalled = np.where((now - S.last_imp > stall_s) & (S.best_all > 0))[0] if kicks else []
        if len(stalled):
            cur = cp.asnumpy(S.cards[cp.asarray(stalled)]); new = []
            for j, i in enumerate(stalled):
                S.kicks[i] += 1
                if S.kicks[i] % 3 == 0 and S.kick_cards[i] < 5: S.kick_cards[i] += 1
                base = S.best_cards[i] if S.kicks[i] % 2 == 0 else cur[j]
                new.append(P.kick(base, dmin, int(S.kick_cards[i]), 4, S.rng))
            S.reset_rows(stalled, np.stack(new)); S.last_imp[stalled] = now
        if now - last > log_every:
            hist.append(dict(t=round(now - t0), best=int(S.best_all.min()), median=float(np.median(S.best_all))))
            last = now
            if progress_file: json.dump(dict(hist=hist, moves=moves), open(progress_file, 'w'))
    el = time.time() - t0
    final = []
    _cnt = P.counts(np.asarray(S.best_cards))  # CPU recount from the blocks (chunked)
    _missall = (_cnt == 0).sum(1); max_cov = int(_cnt.max())
    if max_cov > 127: overflow = True
    if overflow:
        import warnings
        warnings.warn('a t-set was covered more than 127 times: int8 GPU counters overflowed; GPU results are unreliable')
    for i in range(R):
        bl = S.best_cards[i]; miss = int(_missall[i])
        final.append(dict(search=i, start=(start_labels[i % len(start_labels)] if start_labels else 'fresh'),
                          missing=miss, gpu_missing=int(S.best_all[i]), kicks=int(S.kicks[i]),
                          solve_time_s=(None if np.isnan(first_solve[i]) else float(first_solve[i])),
                          blocks=[[int(e) + 1 for e in sorted(b)] for b in bl]))
    return dict(params=dict(v=P.v, k=P.k, t=P.t, R=R, nb=nb, dmin=dmin, seconds=seconds, seed=seed, stall_s=stall_s,
                            tenure=S.tenure, kicks=kicks),
                elapsed_s=el, moves=moves, moves_per_s=moves / max(el, 1e-9), hist=hist,
                gpu_matches_cpu=all(f['missing'] == f['gpu_missing'] for f in final),
                max_coverage_final=max_cov, cnt_overflow=overflow, final=final)


def to0(blocks1):
    return np.array([[e - 1 for e in b] for b in blocks1])
