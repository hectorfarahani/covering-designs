"""GPU weighted local search for covering designs C(v,k,4): find nb k-subsets of {1..v} covering every 4-subset.

Generic version of gpu_wls.py (same move rule as the CPU full_ils.py):
  one warp (32 GPU threads) = one independent search ("replica");
  each step: pick a random uncovered 4-set q, evaluate every swap (card b, x out of b, y in q\\b),
  take the best (random tie-break); if it does not improve, +1 weight on every uncovered 4-set;
  tabu[b,x] for `tenure` steps; optional degree floor dmin (no point drops below dmin cards).
  Host side: iterated-local-search kicks for searches that stall for `stall_s` seconds.
Limits: t = 4 only; v <= 32; nb <= 127.

Usage
    import covsearch_gpu as CG
    P = CG.Problem(20, 12)                       # compiles the kernel for v=20, k=12
    res = CG.run(P, R=512, nb=19, dmin=11, seconds=600, seed=1, starts=None)
    res['final'][i]  -> per-search dict(missing, blocks (1-based), signature, start)
All reported 'missing' values are recomputed independently on the CPU (P.verify), not taken from the GPU counters.
Environment: conda env 'gpu' (numpy + cupy-cuda12x + nvidia-cuda-nvrtc-cu12 + nvidia-cuda-runtime-cu12)."""
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
#define NT __NT__
#define NTRI __NTRI__
#define FULL 0xffffffffu
__constant__ int BIN[(V+1)*5];
__constant__ unsigned char TRI[NTRI*3];
__device__ __forceinline__ unsigned xs(unsigned &s){ s^=s<<13; s^=s>>17; s^=s<<5; return s; }
__device__ __forceinline__ int r4(int a,int b,int c,int d){
  int t;
  if(a>b){t=a;a=b;b=t;} if(c>d){t=c;c=d;d=t;} if(a>c){t=a;a=c;c=t;} if(b>d){t=b;b=d;d=t;} if(b>c){t=b;b=c;c=t;}
  return BIN[a*5+1]+BIN[b*5+2]+BIN[c*5+3]+BIN[d*5+4];
}
extern "C" __global__ void wls(signed char* cnt, unsigned short* w, signed char* cards, int* deg, int* tabu,
    unsigned* rng, int* nunc_a, int* best_a, signed char* bestcards, int* itc, const signed char* UNR,
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
    int q0=UNR[4*q],q1=UNR[4*q+1],q2=UNR[4*q+2],q3=UNR[4*q+3];
    int bestd=0x7fffffff, bb=-1,bx=-1,by=-1, ties=0;
    for (int p=lane; p<nb*K; p+=32){
      int b=p/K, xi=p%K; const signed char* card=KD+b*K;
      int x=card[xi];
      if (x==q0||x==q1||x==q2||x==q3) continue;
      if (dmin>0 && D[x]-1<dmin) continue;
      unsigned m=0; for(int e=0;e<K;e++) m|=1u<<card[e];
      int ys[4]; int ny=0;
      if(!((m>>q0)&1)) ys[ny++]=q0; if(!((m>>q1)&1)) ys[ny++]=q1; if(!((m>>q2)&1)) ys[ny++]=q2; if(!((m>>q3)&1)) ys[ny++]=q3;
      if(ny==0) continue;
      int rest[K-1]; int r=0; for(int e=0;e<K;e++) if(e!=xi) rest[r++]=card[e];
      int loss=0; int g0=0,g1=0,g2=0,g3=0;
      for(int tr=0; tr<NTRI; tr++){
        int a1=rest[TRI[3*tr]], a2=rest[TRI[3*tr+1]], a3=rest[TRI[3*tr+2]];
        int ix=r4(a1,a2,a3,x); if(C[ix]==1) loss+=W[ix];
        int iy;
        iy=r4(a1,a2,a3,ys[0]); if(C[iy]==0) g0+=W[iy];
        if(ny>1){ iy=r4(a1,a2,a3,ys[1]); if(C[iy]==0) g1+=W[iy]; }
        if(ny>2){ iy=r4(a1,a2,a3,ys[2]); if(C[iy]==0) g2+=W[iy]; }
        if(ny>3){ iy=r4(a1,a2,a3,ys[3]); if(C[iy]==0) g3+=W[iy]; }
      }
      int gs[4]={g0,g1,g2,g3};
      bool tb = TB[b*V+x] > it;
      for(int j=0;j<ny;j++){
        int d=loss-gs[j];
        if (tb && d>=0) continue;
        if (d<bestd){ bestd=d; bb=b; bx=x; by=ys[j]; ties=1; }
        else if (d==bestd){ ties++; if(xs(s)%ties==0){ bb=b; bx=x; by=ys[j]; } }
      }
    }
    int mind=bestd; for(int o=16;o>0;o>>=1) mind=min(mind,__shfl_xor_sync(FULL,mind,o));
    unsigned cand=__ballot_sync(FULL, bb>=0 && bestd==mind);
    if (cand==0) continue;
    int nc=__popc(cand); int pick=xs(sh)%nc; int src=0;
    for(int l=0;l<32;l++){ if((cand>>l)&1){ if(pick==0){src=l;break;} pick--; } }
    bb=__shfl_sync(FULL,bb,src); bx=__shfl_sync(FULL,bx,src); by=__shfl_sync(FULL,by,src);
    if (mind>=0){ for(int i=lane;i<NT;i+=32) if(C[i]==0 && W[i]<65000) W[i]++; }
    __syncwarp();
    signed char* card=KD+bb*K;
    int rest[K-1]; int r=0; for(int e=0;e<K;e++) if(card[e]!=bx) rest[r++]=card[e];
    int dn=0;
    for(int tr=lane; tr<NTRI; tr+=32){
      int a1=rest[TRI[3*tr]], a2=rest[TRI[3*tr+1]], a3=rest[TRI[3*tr+2]];
      int ix=r4(a1,a2,a3,bx); int c=C[ix]-1; C[ix]=(signed char)c; if(c==0) dn++;
      int iy=r4(a1,a2,a3,by); if(C[iy]==0) dn--; C[iy]=(signed char)(C[iy]+1);
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
    """Tables, compiled kernel and exact CPU helpers for C(v, k, 4)."""
    def __init__(self, v, k):
        assert 5 <= k < v <= 32, "needs 5 <= k < v <= 32"
        self.v, self.k, self.nt = v, k, comb(v, 4)
        self.BIN = np.array([[comb(a, c) if c <= a else 0 for c in range(5)] for a in range(v + 1)], dtype=np.int32)
        self.TRI = np.array(list(itertools.combinations(range(k - 1), 3)), dtype=np.int32)
        self.POS4 = np.array(list(itertools.combinations(range(k), 4)), dtype=np.int64)
        self.UNR = np.zeros((self.nt, 4), dtype=np.int8)
        for q in itertools.combinations(range(v), 4):
            self.UNR[self.rank4(np.array(q))] = q
        code = (_KERNEL.replace('__V__', str(v)).replace('__K__', str(k))
                .replace('__NT__', str(self.nt)).replace('__NTRI__', str(len(self.TRI))))
        mod = cp.RawModule(code=code, options=('-std=c++11',))
        b = np.ascontiguousarray(self.BIN.ravel()); t8 = np.ascontiguousarray(self.TRI.astype(np.uint8).ravel())
        cp.cuda.runtime.memcpy(mod.get_global('BIN').ptr, b.ctypes.data, b.nbytes, cp.cuda.runtime.memcpyHostToDevice)
        cp.cuda.runtime.memcpy(mod.get_global('TRI').ptr, t8.ctypes.data, t8.nbytes, cp.cuda.runtime.memcpyHostToDevice)
        self.mod = mod; self.kern = mod.get_function('wls'); self.UNR_d = cp.asarray(self.UNR.ravel())

    def rank4(self, a):  # (..., 4) ascending, 0-based -> colex rank
        B = self.BIN
        return B[a[..., 0], 1] + B[a[..., 1], 2] + B[a[..., 2], 3] + B[a[..., 3], 4]

    def counts(self, cards):  # (R, nb, k) 0-based -> (R, nt) coverage multiplicity
        R = cards.shape[0]
        idx = self.rank4(np.sort(cards, axis=2)[:, :, self.POS4]).reshape(R, -1) + (np.arange(R) * self.nt)[:, None]
        return np.bincount(idx.ravel(), minlength=R * self.nt).reshape(R, self.nt)

    def verify(self, blocks0):
        """Independent check (plain itertools, no shared ranking code path with the GPU): number of uncovered 4-sets."""
        cov = set()
        for b in blocks0:
            cov.update(itertools.combinations(sorted(int(e) for e in b), 4))
        return self.nt - len(cov)

    def holes(self, blocks0):
        m = self.counts(np.asarray(blocks0)[None])[0]
        return [tuple(int(x) for x in self.UNR[i]) for i in np.where(m == 0)[0]]

    def drop_one(self, blocks0):
        """For each card: how many 4-sets become uncovered if that card is removed."""
        c = np.asarray(blocks0); m = self.counts(c[None])[0]
        idx = self.rank4(np.sort(c, axis=1)[:, self.POS4])
        return (m[idx] == 1).sum(1)

    def signature(self, blocks0):
        """Relabelling-invariant fingerprint. Different signatures => certainly different shapes
        (not relabellings of each other). Equal signatures => very likely the same shape (not proven)."""
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
        assert nb <= 127
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
                    self.best, self.bestcards, self.itc, self.P.UNR_d, np.int32(self.R), np.int32(self.nb),
                    np.int32(self.dmin), np.int32(steps), np.int32(self.tenure)))
        cp.cuda.Device().synchronize()


def run(P, R, nb, dmin, seconds, seed, starts=None, start_labels=None, stall_s=60.0, launch_steps=200,
        stop_at_zero=True, progress_file=None, log_every=30.0, kicks=True):
    """Run R searches for `seconds`. Returns a dict with per-search results (independently verified).
    Each GPU launch is kept short (< ~1 s) because Windows resets the GPU if one kernel runs > ~2 s."""
    S = Swarm(P, R, nb, dmin, seed, starts=starts)
    t0 = time.time(); S.last_imp[:] = t0; hist = []; last = t0; moves = 0
    first_solve = np.full(R, np.nan)
    while time.time() - t0 < seconds:
        S.step(launch_steps); moves += launch_steps * R; now = time.time()
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
    for i in range(R):
        bl = S.best_cards[i]; miss = P.verify(bl)
        final.append(dict(search=i, start=(start_labels[i % len(start_labels)] if start_labels else 'fresh'),
                          missing=miss, gpu_missing=int(S.best_all[i]), kicks=int(S.kicks[i]),
                          solve_time_s=(None if np.isnan(first_solve[i]) else float(first_solve[i])),
                          blocks=[[int(e) + 1 for e in sorted(b)] for b in bl]))
    return dict(params=dict(v=P.v, k=P.k, t=4, R=R, nb=nb, dmin=dmin, seconds=seconds, seed=seed, stall_s=stall_s,
                            tenure=S.tenure, kicks=kicks),
                elapsed_s=el, moves=moves, moves_per_s=moves / el, hist=hist,
                gpu_matches_cpu=all(f['missing'] == f['gpu_missing'] for f in final), final=final)


def to0(blocks1):
    return np.array([[e - 1 for e in b] for b in blocks1])
