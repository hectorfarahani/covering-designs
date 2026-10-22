"""Calibration tests (need an NVIDIA GPU + CuPy). Run: python tests/test_calibration.py
Proven values: C(10,6,4)=20, C(8,4,3)=14, C(7,3,2)=7. The search must solve at the value and never below it."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import covsearch_gpu_t, covsearch_gpu_t64, covsearch
for CT in (covsearch_gpu_t, covsearch_gpu_t64, covsearch):
  print(CT.__name__)
  for (v, k, t, val) in [(10, 6, 4, 20), (8, 4, 3, 14), (7, 3, 2, 7)]:
      P = CT.Problem(v, k, t)
      a = CT.run(P, 128, val, 0, 30, seed=1, stop_at_zero=False)
      b = CT.run(P, 128, val - 1, 0, 30, seed=2)
      ok_at = sum(f['missing'] == 0 for f in a['final']); best_below = min(f['missing'] for f in b['final'])
      print(f"C({v},{k},{t})={val}: solved at {val}: {ok_at}/128; best at {val-1}: {best_below} missing;",
            "OK" if ok_at > 0 and best_below > 0 and a['gpu_matches_cpu'] and b['gpu_matches_cpu'] else "FAIL")
