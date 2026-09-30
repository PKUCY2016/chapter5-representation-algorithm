"""Bounded wall-time benchmark for MSSA; execution is explicit and opt-in."""
import argparse, json, pathlib, time, tracemalloc
import numpy as np
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from reference import mssa, tssa, dense_attention, orthogonal_subspaces

def flop_estimate(operator, d, p, k, n):
    # Multiply-add counted as two FLOPs; this is an auditable analytic estimate,
    # separate from wall-clock measurements and BLAS kernel counters.
    projection = 2 * k * d * p * n
    if operator == 'TSSA':
        return int(projection + 8 * k * p * n + 5 * k * n)
    if operator == 'MSSA':
        return int(projection + 4 * k * p * n * n + 2 * k * d * p * n)
    if operator == 'DENSE_ATTENTION':
        return int(projection + 4 * k * p * n * n + 2 * k * d * p * n)
    raise ValueError(operator)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--sizes', nargs='+', type=int, default=[128,256,512,1024,2048])
    args = ap.parse_args()
    rows=[]
    for n in args.sizes:
        rng=np.random.default_rng(0); d=128; U=orthogonal_subspaces(d,4,32,rng); Z=rng.normal(size=(d,n))
        for operator, fn in [('MSSA', lambda: mssa(Z,U,.75,.1)), ('TSSA', lambda: tssa(Z,U,.75,.1,.5)), ('DENSE_ATTENTION', lambda: dense_attention(Z,U,.75))]:
            tracemalloc.start(); t=time.perf_counter(); _,diag=fn(); elapsed=time.perf_counter()-t
            _, peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
            rows.append({'N':n,'operator':operator,'wall_seconds':elapsed,'peak_memory_bytes':peak,'FLOPs_estimate':flop_estimate(operator, d, 32, 4, n),'flops_method':'analytic_multiply_add_count','A_norms':diag.get('A_norms'),'status':'PASS'})
    pathlib.Path(args.out).write_text(json.dumps({'status':'PASS','rows':rows},indent=2)+'\n')
if __name__=='__main__': main()
