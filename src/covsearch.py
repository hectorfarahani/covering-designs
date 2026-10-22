"""GPU weighted local search for covering designs C(v,k,t): find nb k-subsets of {1..v} covering every t-subset.
One module for every case: the search of covsearch_gpu_t64.py extended to v <= 127, with the speed-ups listed below.

Move rule:
  one warp (32 GPU threads) = one independent search;
  each step: pick a random uncovered t-set q and evaluate swaps (block b, x out of b, y in q minus b):
    move='focused': only blocks with maximal overlap with q (as covsearch_gpu_t64.py);
    move='full'   : every block (as covsearch_gpu_t.py);
  take the best (random tie-break); if it does not improve, +1 weight on every uncovered t-set;
  tabu[b,x] for `tenure` steps; degree floor dmin and ceiling dmax (int or one value per point).
  dmin/dmax are heuristic search restrictions, not proven requirements.
  Host side: kicks for searches that stall for `stall_s` seconds.
Speed-ups over covsearch_gpu_t64.py (same move rule):
  - each search keeps a list of its uncovered t-sets, so picking q and adding weight cost O(uncovered), not O(C(v,t)).
    The list holds at most `list_cap` entries; above that the search falls back to the full scan and rebuilds the list
    once the uncovered count drops to list_cap/2;
  - binomial table, dmin and dmax in shared memory (the old __constant__ table serialised divergent reads);
  - score='block' (default for the focused move): each candidate block is scored in one pass (loss of every x from the
    block's t-subsets, gain of every y from its (t-1)-subsets) instead of once per removed point x. Identical scores and
    moves to score='point'; about 5-6x fewer counter lookups.
Limits: 2 <= t <= 7; t < k < v <= 127; C(v,t) < 2^31; nb <= 4000. Coverage counters are int8 (overflow above 127 covers,
impossible for nb <= 127; checked otherwise, see run()). Every reported 'missing' is recounted from the blocks
(on the GPU, independent of the kernel's counters); every complete design is also recounted on the CPU.

Starting designs (Swarm/run argument `starts`, CLI --start):
  None or 'balanced' : greedy degree-balanced random designs (default);
  'random'           : uniformly random blocks;
  a file path / glob, or a list of them : designs (1-based, one block per line); cycled over the searches;
                       a design with more than nb blocks loses random blocks, one with fewer gets random extra blocks;
  a list of 0-based (nb, k) arrays : used as given (cycled).
  start_kick=(ncards, nswap) perturbs every search's start so searches from one design differ.

Usage
    import covsearch as CS
    P = CS.Problem(35, 11, 3)                       # move='focused' by default
    res = CS.run(P, R=2048, nb=52, dmin=16, seconds=60, seed=4)
CLI
    python covsearch.py 35 11 3 52 --dmin 16 --seconds 60 --seed 4 --start C35_11_3_53blocks.txt --start-kick 2,4
    python covsearch.py --config run.json           # same keys as the CLI options (v, k, t, nb, dmin, ...)
Environment: see environment.yml (numpy + CuPy for CUDA 12)."""
import os, glob, site, itertools, json, time, argparse, warnings
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
#define BW __BW__
#define CAP __CAP__
#define REBUILD __REBUILD__
#define FOCUSED __FOCUSED__
#define BLOCKSCORE __BLOCKSCORE__
#define NPOS __NPOS__
#define NSUBK __NSUBK__
#define WPB 4
#define FULL 0xffffffffu
__constant__ int BIN[(V+1)*BW];
__shared__ int sBIN[(V+1)*BW];
__shared__ int sDMN[V];
__shared__ int sDMX[V];
#if BLOCKSCORE
__shared__ int sACC[WPB][K*(T+1)];   // per warp: loss per block position, then gain correction per (y, position)
#endif
// per t-set state in one 4-byte word, so count and weight come in one memory read
struct __align__(4) CWE { signed char c; unsigned char f; unsigned short w; };   // count, in-list flag, weight
__device__ __forceinline__ unsigned xs(unsigned &s){ s^=s<<13; s^=s>>17; s^=s<<5; return s; }
__device__ __forceinline__ unsigned below(int lane){ return (1u<<lane)-1u; }
__device__ __forceinline__ int rnk(const int* a0, int extra){
  int a[T]; for(int i=0;i<T-1;i++) a[i]=a0[i]; a[T-1]=extra;
  for(int i=1;i<T;i++){ int x=a[i]; int j=i-1; while(j>=0 && a[j]>x){ a[j+1]=a[j]; j--; } a[j+1]=x; }
  int r=0; for(int i=0;i<T;i++) r+=sBIN[a[i]*BW+i+1]; return r;
}
__device__ __forceinline__ int rnkT(const int* a0){
  int a[T]; for(int i=0;i<T;i++) a[i]=a0[i];
  for(int i=1;i<T;i++){ int x=a[i]; int j=i-1; while(j>=0 && a[j]>x){ a[j+1]=a[j]; j--; } a[j+1]=x; }
  int r=0; for(int i=0;i<T;i++) r+=sBIN[a[i]*BW+i+1]; return r;
}
// Keep only the uncovered entries of the list L[0..ln) (clear the flag of the dropped ones); bump: +1 weight on each kept one.
__device__ int compact(int* L, int ln, CWE* E, bool bump, int lane){
  int w0=0;
  for(int base=0; base<ln; base+=32){
    int i=base+lane; int c=-1; bool keep=false;
    if(i<ln){ c=L[i]; CWE e=E[c]; keep=(e.c==0); if(!keep) E[c].f=0; else if(bump && e.w<65000) E[c].w=e.w+1; }
    unsigned bal=__ballot_sync(FULL,keep);
    if(keep) L[w0+__popc(bal&below(lane))]=c;
    w0+=__popc(bal);
    __syncwarp();
  }
  return w0;
}
extern "C" __global__ void wls(CWE* cw, int* ulist, int* uln_a, int* uovf_a,
    signed char* cards, int* deg, int* tabu, unsigned* rng, int* nunc_a, int* best_a, signed char* bestcards, int* itc,
    const signed char* UNR, const unsigned char* SUB, const unsigned char* POSK, const unsigned char* SUBK,
    const int* DMN, const int* DMX, int R, int nb, int steps, int tenure)
{
  for(int i=threadIdx.x;i<(V+1)*BW;i+=blockDim.x) sBIN[i]=BIN[i];
  for(int i=threadIdx.x;i<V;i+=blockDim.x){ sDMN[i]=DMN[i]; sDMX[i]=DMX[i]; }
  __syncthreads();
  int gw = (blockIdx.x*blockDim.x + threadIdx.x) >> 5; int lane = threadIdx.x & 31;
  if (gw >= R) return;
  CWE* E = cw + (size_t)gw*NT;
  int* L = ulist + (size_t)gw*CAP;
  signed char* KD = cards + (size_t)gw*nb*K; int* D = deg + gw*V; int* TB = tabu + (size_t)gw*nb*V;
  signed char* BC = bestcards + (size_t)gw*nb*K;
  unsigned sh = rng[gw] | 1u; unsigned s = (rng[gw]*2654435761u) ^ (lane*40503u + 12345u); if(!s) s=1;
  int nunc = nunc_a[gw], best = best_a[gw], it0 = itc[gw], ln = uln_a[gw], ovf = uovf_a[gw];
  int t;
  for (t=0; t<steps; t++){
    int it = it0 + t;
    if (nunc==0) break;
    // (re)build the uncovered list once it fits again
    if (ovf && nunc<=REBUILD){
      ln=0;
      for(int base=0; base<NT; base+=32){
        int i=base+lane; bool u=false;
        if(i<NT){ u=(E[i].c==0); E[i].f=(unsigned char)u; }
        unsigned bal=__ballot_sync(FULL,u);
        if(u) L[ln+__popc(bal&below(lane))]=i;
        ln+=__popc(bal);
      }
      ovf=0; __syncwarp();
    }
    // pick a uniformly random uncovered t-set q
    int q=-1;
    if (!ovf){
      // every uncovered t-set is in L exactly once (plus stale covered entries): 32 uniform probes, first valid one wins
      for(int round=0; round<2 && q<0; round++){
        int c=L[xs(s)%ln];
        unsigned bal=__ballot_sync(FULL, E[c].c==0);
        if(bal) q=__shfl_sync(FULL,c,__ffs(bal)-1);
        else ln=compact(L,ln,E,false,lane);   // afterwards every entry is valid
      }
    } else {
      int cz=0; for(int i=lane;i<NT;i+=32) cz += (E[i].c==0);
      int pre=cz; for(int o=1;o<32;o<<=1){ int v=__shfl_up_sync(FULL,pre,o); if(lane>=o) pre+=v; }
      int total=__shfl_sync(FULL,pre,31);
      int k = xs(sh) % total;
      if (k>=pre-cz && k<pre){ int j=k-(pre-cz); for(int i=lane;i<NT;i+=32){ if(E[i].c==0){ if(j==0){q=i;break;} j--; } } }
      unsigned bal=__ballot_sync(FULL,q>=0); q=__shfl_sync(FULL,q,__ffs(bal)-1);
    }
    int qp[T]; for(int i=0;i<T;i++) qp[i]=UNR[T*q+i];
    int bestd=0x7fffffff, bb=-1,bx=-1,by=-1, ties=0;
#if FOCUSED
    // focused move: only blocks with maximal overlap with q; lanes share the subset loop
    int myov=-1;
    for(int bb2=lane; bb2<nb; bb2+=32){ const signed char* cd=KD+bb2*K; int o=0; for(int e=0;e<K;e++) for(int i=0;i<T;i++) if(cd[e]==qp[i]) o++; if(o>myov) myov=o; }
    int maxov=myov; for(int o=16;o>0;o>>=1) maxov=max(maxov,__shfl_xor_sync(FULL,maxov,o));
    // candidate blocks (overlap == maxov): each lane tests one block of every 32, then the warp visits the
    // candidates in ascending order (same order, so same tie-breaks, as testing every block on every lane)
    for(int base=0; base<nb; base+=32){
     int bl=base+lane; bool cnd=false;
     if(bl<nb){ const signed char* cd=KD+bl*K; int o=0; for(int e=0;e<K;e++) for(int i=0;i<T;i++) if(cd[e]==qp[i]) o++; cnd=(o==maxov && o<T); }
     unsigned cmask=__ballot_sync(FULL,cnd);
     while(cmask){
      int b2=base+__ffs(cmask)-1; cmask&=cmask-1;
      const signed char* card=KD+b2*K;
      bool inb[T]; for(int i=0;i<T;i++) inb[i]=false;
      for(int e=0;e<K;e++){ int pc=card[e]; for(int i=0;i<T;i++) if(pc==qp[i]) inb[i]=true; }
      int ys[T]; int ny=0; for(int i=0;i<T;i++) if(!inb[i] && D[qp[i]]<sDMX[qp[i]]) ys[ny++]=qp[i];
      if(ny==0) continue;
#if BLOCKSCORE
      // one pass over the block: loss of every x from its t-subsets, gain of every y from its (t-1)-subsets
      // (gain of y with x removed = gain over all (t-1)-subsets minus those containing x). Same scores as below.
      int* LS=sACC[threadIdx.x>>5]; int* GS=LS+K;
      for(int i=lane;i<K*(T+1);i+=32) LS[i]=0;
      __syncwarp();
      int gt[T]; for(int j=0;j<T;j++) gt[j]=0;
      {
        int a[T];
        for(int p=lane;p<NPOS;p+=32){
          const unsigned char* ps=POSK+T*p;
          for(int i=0;i<T;i++) a[i]=card[ps[i]];
          int ix=rnkT(a); CWE ex=E[ix]; if(ex.c==1){ int wv=ex.w; for(int i=0;i<T;i++) atomicAdd(&LS[ps[i]],wv); }
        }
        for(int p=lane;p<NSUBK;p+=32){
          const unsigned char* ps=SUBK+(T-1)*p;
          for(int i=0;i<T-1;i++) a[i]=card[ps[i]];
          for(int j=0;j<ny;j++){
            int iy=rnk(a,ys[j]);
            CWE ey=E[iy]; if(ey.c==0){ int wv=ey.w; gt[j]+=wv; for(int i=0;i<T-1;i++) atomicAdd(&GS[j*K+ps[i]],wv); }
          }
        }
      }
      for(int o2=16;o2>0;o2>>=1) for(int j=0;j<T;j++) gt[j]+=__shfl_xor_sync(FULL,gt[j],o2);
      __syncwarp();
#endif
      for(int xi=0; xi<K; xi++){
        int x=card[xi]; bool inq=false; for(int i=0;i<T;i++) if(x==qp[i]) inq=true;
        if(inq) continue;
        if(D[x]-1<sDMN[x]) continue;
        int loss=0; int g[T]; for(int i=0;i<T;i++) g[i]=0;
#if BLOCKSCORE
        loss=LS[xi]; for(int j=0;j<ny;j++) g[j]=gt[j]-GS[j*K+xi];
#else
        int rest[K-1]; int r=0; for(int e=0;e<K;e++) if(e!=xi) rest[r++]=card[e];
        int a[T-1];
        for(int tr=lane; tr<NSUB; tr+=32){
          for(int i=0;i<T-1;i++) a[i]=rest[SUB[(T-1)*tr+i]];
          int ix=rnk(a,x); CWE ex=E[ix]; if(ex.c==1) loss+=ex.w;
          for(int j=0;j<ny;j++){ int iy=rnk(a,ys[j]); CWE ey=E[iy]; if(ey.c==0) g[j]+=ey.w; }
        }
        for(int o2=16;o2>0;o2>>=1){ loss+=__shfl_xor_sync(FULL,loss,o2); for(int j=0;j<T;j++) g[j]+=__shfl_xor_sync(FULL,g[j],o2); }
#endif
        bool tb = TB[b2*V+x] > it;
        for(int j=0;j<ny;j++){
          int d=loss-g[j];
          if (tb && d>=0) continue;
          if (d<bestd){ bestd=d; bb=b2; bx=x; by=ys[j]; ties=1; }
          else if (d==bestd){ ties++; if(xs(sh)%ties==0){ bb=b2; bx=x; by=ys[j]; } }
        }
      }
#if BLOCKSCORE
      __syncwarp();   // all lanes done reading sACC before the next block clears it
#endif
     }
    }
    int mind=bestd;
    if (bb<0) continue;
#else
    // full move: every (block, x) pair; lanes split the pairs
    for (int p=lane; p<nb*K; p+=32){
      int b=p/K, xi=p%K; const signed char* card=KD+b*K;
      int x=card[xi];
      bool inq=false; for(int i=0;i<T;i++) if(x==qp[i]) inq=true;
      if (inq) continue;
      if (D[x]-1<sDMN[x]) continue;
      bool inb[T]; for(int i=0;i<T;i++) inb[i]=false;
      for(int e=0;e<K;e++) for(int i=0;i<T;i++) if(card[e]==qp[i]) inb[i]=true;
      int ys[T]; int ny=0; for(int i=0;i<T;i++) if(!inb[i] && D[qp[i]]<sDMX[qp[i]]) ys[ny++]=qp[i];
      if(ny==0) continue;
      int rest[K-1]; int r=0; for(int e=0;e<K;e++) if(e!=xi) rest[r++]=card[e];
      int loss=0; int g[T]; for(int i=0;i<T;i++) g[i]=0;
      int a[T-1];
      for(int tr=0; tr<NSUB; tr++){
        for(int i=0;i<T-1;i++) a[i]=rest[SUB[(T-1)*tr+i]];
        int ix=rnk(a,x); CWE ex=E[ix]; if(ex.c==1) loss+=ex.w;
        for(int j=0;j<ny;j++){ int iy=rnk(a,ys[j]); CWE ey=E[iy]; if(ey.c==0) g[j]+=ey.w; }
      }
      bool tb = TB[b*V+x] > it;
      for(int j=0;j<ny;j++){
        int d=loss-g[j];
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
#endif
    // no improving swap: +1 weight on every uncovered t-set
    if (mind>=0){
      if (!ovf) ln=compact(L,ln,E,true,lane);
      else for(int i=lane;i<NT;i+=32){ CWE e=E[i]; if(e.c==0 && e.w<65000) E[i].w=e.w+1; }
    }
    __syncwarp();
    // apply the swap bx -> by in block bb; newly uncovered t-sets join the list
    signed char* card=KD+bb*K;
    int rest[K-1]; int r=0; for(int e=0;e<K;e++) if(card[e]!=bx) rest[r++]=card[e];
    int dn=0; int a[T-1];
    for(int base=0; base<NSUB; base+=32){
      int tr=base+lane; int nw=-1;
      if (tr<NSUB){
        for(int i=0;i<T-1;i++) a[i]=rest[SUB[(T-1)*tr+i]];
        int ix=rnk(a,bx); CWE ex=E[ix]; int c=ex.c-1; E[ix].c=(signed char)c; if(c==0){ dn++; if(!ex.f) nw=ix; }
        int iy=rnk(a,by); int cy=E[iy].c; if(cy==0) dn--; E[iy].c=(signed char)(cy+1);
      }
      if (!ovf){
        unsigned bal=__ballot_sync(FULL,nw>=0);
        if (bal){
          int n=__popc(bal);
          if (ln+n>CAP) ovf=1;
          else { if(nw>=0){ L[ln+__popc(bal&below(lane))]=nw; E[nw].f=1; } ln+=n; }
        }
      }
    }
    for(int o=16;o>0;o>>=1) dn+=__shfl_xor_sync(FULL,dn,o);
    nunc+=dn;
    __syncwarp();
    if(lane==0){ for(int e=0;e<K;e++) if(card[e]==bx) card[e]=(signed char)by; D[bx]--; D[by]++; TB[bb*V+bx]=it+tenure; }
    __syncwarp();
    if (nunc<best){ best=nunc; for(int i=lane;i<nb*K;i+=32) BC[i]=KD[i]; }
    __syncwarp();
  }
  if(lane==0){ nunc_a[gw]=nunc; best_a[gw]=best; itc[gw]=it0+t; rng[gw]=sh; uln_a[gw]=ln; uovf_a[gw]=ovf; }
}
'''

MOVES = ('focused', 'full')
SCORES = ('block', 'point')   # swap scoring: once per candidate block (faster) or once per removed point (legacy)
NO_DMAX = 10**6


def per_point(x, v, default):
    """int, None or length-v sequence -> int32 array of length v."""
    return np.broadcast_to(np.asarray(default if x is None else x, dtype=np.int32), (v,)).copy()


class Problem:
    """Tables, compiled kernel and exact CPU helpers for C(v, k, t)."""
    def __init__(self, v, k, t, move='focused', list_cap=1 << 15, score='auto'):
        assert move in MOVES, f"move must be one of {MOVES}"
        if score == 'auto': score = 'block' if move == 'focused' else 'point'
        assert score in SCORES and not (score == 'block' and move == 'full'), "score='block' needs move='focused'"
        self.score = score
        assert 2 <= t <= 7 and t < k < v <= 127 and comb(v, t) < 2**31, "needs 2<=t<=7, t<k<v<=127, C(v,t) < 2^31"
        self.v, self.k, self.t, self.nt, self.move = v, k, t, comb(v, t), move
        self.cap = int(min(self.nt, list_cap))
        rebuild = self.cap if self.cap >= self.nt else self.cap // 2
        self.BW = t + 1
        self.BIN = np.array([[comb(a, c) if c <= a else 0 for c in range(self.BW)] for a in range(v + 1)], dtype=np.int32)
        self.SUB = np.array(list(itertools.combinations(range(k - 1), t - 1)), dtype=np.uint8)
        self.POS = np.array(list(itertools.combinations(range(k), t)), dtype=np.int64)
        self.SUBK = np.array(list(itertools.combinations(range(k), t - 1)), dtype=np.uint8)
        Q = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(v), t)), dtype=np.int8,
                        count=self.nt * t).reshape(self.nt, t)
        self.UNR = np.zeros((self.nt, t), dtype=np.int8); self.UNR[self.rank(Q)] = Q
        code = _KERNEL
        for key, val in dict(V=v, K=k, T=t, NT=self.nt, BW=self.BW, NSUB=len(self.SUB), CAP=self.cap, REBUILD=rebuild,
                             FOCUSED=int(move == 'focused'), BLOCKSCORE=int(score == 'block'),
                             NPOS=len(self.POS), NSUBK=len(self.SUBK)).items():
            code = code.replace(f'__{key}__', str(val))
        mod = cp.RawModule(code=code, options=('-std=c++11',))
        b = np.ascontiguousarray(self.BIN.ravel())
        cp.cuda.runtime.memcpy(mod.get_global('BIN').ptr, b.ctypes.data, b.nbytes, cp.cuda.runtime.memcpyHostToDevice)
        self.mod = mod; self.kern = mod.get_function('wls'); self.UNR_d = cp.asarray(self.UNR.ravel())
        self.SUB_d = cp.asarray(np.ascontiguousarray(self.SUB.ravel()))
        self.POSK_d = cp.asarray(np.ascontiguousarray(self.POS.astype(np.uint8).ravel()))
        self.SUBK_d = cp.asarray(np.ascontiguousarray(self.SUBK.ravel()))
        self.BIN_d = cp.asarray(self.BIN); self.POS_d = cp.asarray(self.POS); self.COL_d = cp.arange(1, t + 1)

    def bytes_per_search(self, nb):
        return self.nt * 4 + self.cap * 4 + nb * self.v * 4 + 2 * nb * self.k + 64

    def max_searches(self, nb, budget=6e9, cap=2048):
        """Largest R (multiple of 32, <= cap) whose GPU state fits in `budget` bytes."""
        return max(32, int(min(cap, budget // self.bytes_per_search(nb))) // 32 * 32)

    def rank(self, a):  # (..., t) ascending, 0-based -> colex rank
        return sum(self.BIN[a[..., i], i + 1] for i in range(self.t))

    def counts(self, cards):  # chunked to bound host memory; int16 counts
        R = cards.shape[0]; out = np.empty((R, self.nt), np.int16)
        per = cards.shape[1] * len(self.POS) * 8 * 3 + self.nt * 8
        ch = max(1, int(3e8 // per))
        for a in range(0, R, ch):
            c = cards[a:a + ch]; r = c.shape[0]
            idx = self.rank(np.sort(c, axis=2)[:, :, self.POS]).reshape(r, -1) + (np.arange(r) * self.nt)[:, None]
            out[a:a + r] = np.bincount(idx.ravel(), minlength=r * self.nt).reshape(r, self.nt)
        return out

    def counts_gpu(self, cards, max_elems=1 << 25):
        """GPU version of counts() for (R, nb, k) cards (numpy or CuPy) -> (int8 counts, uncovered per search, max count).
        int8 wraps above 127 (as the kernel's counters); the returned max count is exact."""
        cards = cp.asarray(cards); R, nb = cards.shape[:2]
        out = cp.empty((R, self.nt), cp.int8); nunc = cp.empty(R, cp.int32); mx = 0
        ch = max(1, max_elems // (nb * len(self.POS) * self.t))
        for a in range(0, R, ch):
            c = cp.sort(cards[a:a + ch].astype(cp.int32), axis=2); r = c.shape[0]
            idx = self.BIN_d[c[:, :, self.POS_d], self.COL_d].sum(-1, dtype=cp.int64).reshape(r, -1)
            idx += (cp.arange(r, dtype=cp.int64) * self.nt)[:, None]
            cnt = cp.bincount(idx.ravel(), minlength=r * self.nt).reshape(r, self.nt)
            mx = max(mx, int(cnt.max())); nunc[a:a + r] = (cnt == 0).sum(1); out[a:a + r] = cnt.astype(cp.int8)
            del idx, cnt
        return out, nunc, mx

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

    def balanced_starts(self, R, nb, rng):
        """R balanced_start designs at once (same rule: each block takes the k least-used points, ties at random)."""
        deg = np.zeros((R, self.v)); out = np.empty((R, nb, self.k), np.int64)
        for b in range(nb):
            blk = np.sort(np.argsort(deg + rng.random((R, self.v)), axis=1)[:, :self.k], axis=1)
            out[:, b] = blk; np.put_along_axis(deg, blk, np.take_along_axis(deg, blk, 1) + 1, 1)
        return out

    def random_start(self, nb, rng):
        return np.sort(np.argsort(rng.random((nb, self.v)), axis=1)[:, :self.k], axis=1)

    def fit(self, blocks, nb, rng):
        """Design with exactly nb blocks: drop random blocks, or add random ones."""
        blocks = np.asarray(blocks)
        if len(blocks) > nb: blocks = blocks[np.sort(rng.choice(len(blocks), nb, replace=False))]
        if len(blocks) < nb: blocks = np.concatenate([blocks, self.random_start(nb - len(blocks), rng)])
        return blocks

    def kick(self, blocks, dmin, ncards, nswap, rng, dmax=None):
        dmn = per_point(dmin, self.v, 0); dmx = per_point(dmax, self.v, NO_DMAX)
        blocks = blocks.copy(); deg = np.bincount(blocks.ravel(), minlength=self.v)
        for b in rng.choice(blocks.shape[0], size=min(ncards, blocks.shape[0]), replace=False):
            card = set(int(e) for e in blocks[b])
            outside = [e for e in range(self.v) if e not in card and deg[e] + 1 <= dmx[e]]
            removable = [e for e in card if deg[e] - 1 >= dmn[e]]
            rng.shuffle(removable); rng.shuffle(outside)
            for x, y in zip(removable[:nswap], outside[:nswap]):
                card.remove(x); card.add(y); deg[x] -= 1; deg[y] += 1
            blocks[b] = np.array(sorted(card))
        return blocks


def load_design(path):
    """Design file (one block per line, 1-based) -> 0-based int array."""
    return np.array([[int(e) - 1 for e in line.split()] for line in open(path) if line.strip()])


def make_starts(P, R, nb, rng, starts=None, dmin=0, dmax=None, start_kick=None):
    """(R, nb, k) starting designs and one label per search; see the module docstring for `starts`."""
    if starts is None or (isinstance(starts, str) and starts in ('balanced', 'random')):
        if starts == 'random': return np.stack([P.random_start(nb, rng) for _ in range(R)]), ['random']
        return P.balanced_starts(R, nb, rng), ['balanced']
    if isinstance(starts, str) or (isinstance(starts, (list, tuple)) and starts and isinstance(starts[0], str)):
        paths = sorted(p for s in ([starts] if isinstance(starts, str) else starts) for p in glob.glob(s))
        assert paths, f"no design file matches {starts}"
        designs, labels = [load_design(p) for p in paths], [os.path.basename(p) for p in paths]
    else:
        designs, labels = [np.asarray(d) for d in starts], None
    cards = np.stack([P.fit(designs[i % len(designs)], nb, rng) for i in range(R)])
    if start_kick:
        cards = np.stack([P.kick(c, dmin, start_kick[0], start_kick[1], rng, dmax) for c in cards])
    return cards, labels


def sig_key(s):
    """Hashable key of a signature dict (used to group results into shapes)."""
    return json.dumps([s['missing'], s['degrees'], s['hole_points'], s['hole_pairs'], s['overlaps'], s['multiplicity']])


def degstr(deg):
    return ' + '.join(f"{n}x{d}" for d, n in sorted(Counter(deg).items()))


class Swarm:
    def __init__(self, P, R, nb, dmin, seed, tenure=40, starts=None, dmax=None, start_kick=None):
        assert nb <= 4000  # cnt is int8: overflow is impossible for nb <= 127; for larger nb run() checks it
        self.P, self.R, self.nb, self.dmin, self.dmax, self.tenure = P, R, nb, dmin, dmax, tenure
        self.rng = np.random.default_rng(seed)
        self.DMN = cp.asarray(per_point(dmin, P.v, 0)); self.DMX = cp.asarray(per_point(dmax, P.v, NO_DMAX))
        cards, self.start_labels = make_starts(P, R, nb, self.rng, starts, dmin, dmax, start_kick)
        self._upload(cards)
        self.rng_d = cp.asarray(self.rng.integers(1, 2**32 - 1, R, dtype=np.uint64).astype(np.uint32))
        self.itc = cp.zeros(R, dtype=cp.int32)
        self.best_all = np.full(R, 10**9); self.best_cards = cards.copy(); self.last_imp = np.zeros(R)
        self.kicks = np.zeros(R, int); self.kick_cards = np.full(R, 2)

    def _upload(self, cards):
        P, R = self.P, self.R
        self.cards = cp.asarray(cards.astype(np.int8)); cnt, nunc, _ = P.counts_gpu(self.cards)
        # kernel's CWE word per t-set: byte 0 count (int8), byte 1 in-list flag, bytes 2-3 weight; cnt/flag/w are views
        self.cw = cp.zeros((R, P.nt, 2), dtype=cp.uint16)
        self.cnt = self.cw.view(cp.int8)[..., 0]; self.flag = self.cw.view(cp.uint8)[..., 1]; self.w = self.cw[..., 1]
        self.cnt[...] = cnt; self.w[...] = 1; del cnt
        self.ulist = cp.zeros((R, P.cap), dtype=cp.int32)
        self.uln = cp.zeros(R, dtype=cp.int32); self.uovf = cp.ones(R, dtype=cp.int32)  # list built by the kernel
        self.deg = cp.asarray(np.stack([np.bincount(c.ravel(), minlength=P.v) for c in cards]).astype(np.int32))
        self.tabu = cp.zeros((R, self.nb, P.v), dtype=cp.int32)
        self.nunc = nunc; self.best = nunc.copy(); self.bestcards = self.cards.copy()

    def reset_rows(self, rows, cards_rows):
        P = self.P; r = cp.asarray(rows); c8 = cp.asarray(cards_rows.astype(np.int8)); cnt, nu, _ = P.counts_gpu(c8)
        self.cards[r] = c8; self.bestcards[r] = c8
        self.cnt[r] = cnt; self.w[r] = 1; self.tabu[r] = 0; self.itc[r] = 0
        self.flag[r] = 0; self.uln[r] = 0; self.uovf[r] = 1
        self.deg[r] = cp.asarray(np.stack([np.bincount(c.ravel(), minlength=P.v) for c in cards_rows]).astype(np.int32))
        self.nunc[r] = nu; self.best[r] = nu

    def step(self, steps):
        threads = 128; blocks = (self.R * 32 + threads - 1) // threads
        self.P.kern((blocks,), (threads,), (self.cw, self.ulist, self.uln, self.uovf, self.cards, self.deg,
                    self.tabu, self.rng_d, self.nunc, self.best, self.bestcards, self.itc, self.P.UNR_d, self.P.SUB_d,
                    self.P.POSK_d, self.P.SUBK_d, self.DMN, self.DMX, np.int32(self.R), np.int32(self.nb), np.int32(steps), np.int32(self.tenure)))
        cp.cuda.Device().synchronize()


def run(P, R, nb, dmin, seconds, seed, starts=None, start_labels=None, stall_s=60.0, launch_steps=200,
        stop_at_zero=True, progress_file=None, log_every=30.0, kicks=True, dmax=None, start_kick=None, tenure=40,
        swarm=None):
    """Run R searches for `seconds`. Returns a dict with per-search results (independently verified).
    Each GPU launch is kept short (< ~1 s) because Windows resets the GPU if one kernel runs > ~2 s.
    `moves` counts the steps actually taken (searches that stopped at 0 missing take none).
    swarm: a Swarm already built with these arguments (e.g. by launch_plan), used instead of building a new one.
    Final counts: every design is recounted from its blocks on the GPU (independent of the kernel's counters); designs
    with 0 missing, or whose recount differs from the kernel, are also recounted on the CPU (`cpu_checked`)."""
    S = swarm or Swarm(P, R, nb, dmin, seed, tenure=tenure, starts=starts, dmax=dmax, start_kick=start_kick)
    assert S.R == R and S.nb == nb
    labels = start_labels or S.start_labels
    t0 = time.time(); S.last_imp[:] = t0; hist = []; last = t0; moves = 0; launches = 0
    it_prev = np.zeros(R, np.int64)
    first_solve = np.full(R, np.nan); overflow = False
    while time.time() - t0 < seconds:
        S.step(launch_steps); launches += 1; now = time.time()
        it_now = cp.asnumpy(S.itc).astype(np.int64); moves += int((it_now - it_prev).sum()); it_prev = it_now
        if nb > 127 and not overflow and launches % 16 == 0 and bool((S.cnt < 0).any()): overflow = True  # int8 wrap
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
                new.append(P.kick(base, dmin, int(S.kick_cards[i]), 4, S.rng, dmax))
            S.reset_rows(stalled, np.stack(new)); S.last_imp[stalled] = now; it_prev[stalled] = 0
        if now - last > log_every:
            hist.append(dict(t=round(now - t0), best=int(S.best_all.min()), median=float(np.median(S.best_all))))
            last = now
            if progress_file: json.dump(dict(hist=hist, moves=moves), open(progress_file, 'w'))
    el = time.time() - t0
    final = []
    _, _nu, max_cov = P.counts_gpu(S.best_cards); del _  # GPU recount from the blocks
    _missall = cp.asnumpy(_nu).astype(int)
    chk = np.where((_missall == 0) | (_missall != S.best_all))[0]; chk_set = set(chk.tolist())
    cpu_miss = (P.counts(np.asarray(S.best_cards[chk])) == 0).sum(1) if len(chk) else np.zeros(0, int)
    cpu_ok = bool((cpu_miss == _missall[chk]).all()); _missall[chk] = cpu_miss  # the CPU count wins
    if max_cov > 127: overflow = True
    if overflow:
        warnings.warn('a t-set was covered more than 127 times: int8 GPU counters overflowed; GPU results are unreliable')
    for i in range(R):
        bl = S.best_cards[i]; miss = int(_missall[i])
        final.append(dict(search=i, start=(labels[i % len(labels)] if labels else 'given'),
                          missing=miss, gpu_missing=int(S.best_all[i]), cpu_checked=i in chk_set, kicks=int(S.kicks[i]),
                          solve_time_s=(None if np.isnan(first_solve[i]) else float(first_solve[i])),
                          blocks=[[int(e) + 1 for e in sorted(b)] for b in bl]))
    return dict(params=dict(v=P.v, k=P.k, t=P.t, move=P.move, score=P.score, R=R, nb=nb, dmin=np.asarray(dmin).tolist(),
                            dmax=None if dmax is None else np.asarray(dmax).tolist(), seconds=seconds, seed=seed,
                            stall_s=stall_s, tenure=tenure, kicks=kicks, start_kick=start_kick,
                            starts=starts if isinstance(starts, (str, type(None))) else 'given'),
                elapsed_s=el, moves=moves, moves_per_s=moves / max(el, 1e-9), hist=hist,
                gpu_matches_cpu=cpu_ok and all(f['missing'] == f['gpu_missing'] for f in final),
                max_coverage_final=max_cov, cnt_overflow=overflow, final=final)


def to0(blocks1):
    return np.array([[e - 1 for e in b] for b in blocks1])


def launch_plan(P, nb, dmin, R=None, budget=6e9, target_s=0.8, seed=0, **swarm_kw):
    """Pick R (by memory, halved while one step takes > 0.5 s) and launch_steps (one launch ~ target_s).
    Returns (R, launch_steps, swarm); pass the swarm (built with `seed` and `swarm_kw`) to run() to avoid a second build."""
    R = R or P.max_searches(nb, budget)
    while True:
        S = Swarm(P, R, nb, dmin, seed, **swarm_kw); S.step(1); t1 = time.time(); S.step(1); dt = time.time() - t1
        if dt <= 0.5 or R <= 32: break
        del S; cp.get_default_memory_pool().free_all_blocks()
        R = max(32, R // 2 // 32 * 32)
    return R, max(1, min(400, int(target_s / max(dt, 1e-4)))), S


def _ints(s):
    if s is None or isinstance(s, (int, list)): return s
    x = [int(e) for e in str(s).split(',')]
    return x[0] if len(x) == 1 else x


def main():
    ap = argparse.ArgumentParser(description='GPU search for covering designs C(v,k,t) with nb blocks.')
    ap.add_argument('v', type=int, nargs='?'); ap.add_argument('k', type=int, nargs='?')
    ap.add_argument('t', type=int, nargs='?'); ap.add_argument('nb', type=int, nargs='?')
    ap.add_argument('--config', help='JSON file with any of the options below (command line wins)')
    ap.add_argument('--dmin', default=None, help='degree floor: int or comma list per point (heuristic restriction)')
    ap.add_argument('--dmax', default=None, help='degree ceiling: int or comma list per point')
    ap.add_argument('--move', default=None, choices=MOVES)
    ap.add_argument('--score', default=None, choices=('auto',) + SCORES, help="swap scoring (default auto: block for focused)")
    ap.add_argument('--R', type=int, default=None, help='parallel searches (default: by memory and step time)')
    ap.add_argument('--seconds', type=float, default=None); ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--start', default=None, help="'balanced', 'random', or design file / glob (1-based)")
    ap.add_argument('--start-kick', default=None, help='ncards,nswap perturbation of every start')
    ap.add_argument('--launch-steps', type=int, default=None); ap.add_argument('--stall-s', type=float, default=None)
    ap.add_argument('--tenure', type=int, default=None)
    ap.add_argument('--out', default=None, help='write the result JSON here')
    ap.add_argument('--save-dir', default=None, help='write every complete design here as C{v}_{k}_{t}_{nb}blocks*.txt')
    a = vars(ap.parse_args())
    cfg = dict(dmin=0, move='focused', seconds=60.0, seed=1, start='balanced', stall_s=60.0, tenure=40)
    if a['config']: cfg.update({k.replace('-', '_'): v for k, v in json.load(open(a['config'])).items()})
    cfg.update({k: v for k, v in a.items() if v is not None})
    v, k, t, nb = cfg['v'], cfg['k'], cfg['t'], cfg['nb']
    dmin, dmax, kick = _ints(cfg['dmin']), _ints(cfg.get('dmax')), _ints(cfg.get('start_kick'))
    P = Problem(v, k, t, move=cfg['move'], score=cfg.get('score', 'auto'))
    R, ls, S = launch_plan(P, nb, dmin, cfg.get('R'), seed=cfg['seed'], tenure=cfg['tenure'], starts=cfg['start'],
                           dmax=dmax, start_kick=kick)
    ls = cfg.get('launch_steps') or ls
    print(f"C({v},{k},{t}) nb={nb} move={P.move} score={P.score} R={R} launch_steps={ls} dmin={dmin} start={cfg['start']}", flush=True)
    res = run(P, R, nb, dmin, cfg['seconds'], cfg['seed'], starts=cfg['start'], stall_s=cfg['stall_s'],
              launch_steps=ls, dmax=dmax, start_kick=kick, tenure=cfg['tenure'], log_every=1e9, swarm=S)
    sols = [f for f in res['final'] if f['missing'] == 0]
    miss = [f['missing'] for f in res['final']]
    print(f"solved {len(sols)}/{R}; best {min(miss)} missing; median {np.median(miss)}; "
          f"{res['moves_per_s']:.0f} moves/s; gpu_matches_cpu {res['gpu_matches_cpu']}", flush=True)
    if cfg.get('save_dir') and sols:
        os.makedirs(cfg['save_dir'], exist_ok=True)
        for j, f in enumerate(sols[:5]):
            assert P.verify(to0(f['blocks'])) == 0
            p = os.path.join(cfg['save_dir'], f"C{v}_{k}_{t}_{nb}blocks" + (f"_{j}" if j else "") + ".txt")
            open(p, 'w').write('\n'.join(' '.join(map(str, b)) for b in f['blocks']) + '\n'); print('saved', p)
    if cfg.get('out'):
        json.dump(res, open(cfg['out'], 'w'))


if __name__ == '__main__':
    main()
