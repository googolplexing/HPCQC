import json, tempfile, os, sys
import numpy as np
from lumi_hpc_qc.sweep import byo_sweep as B

P = F = 0
def ok(label, cond):
    global P, F
    if cond: P += 1; print(f"PASS  {label}")
    else:    F += 1; print(f"FAIL  {label}")
def raises(label, fn, msg_sub=None):
    global P, F
    try:
        fn(); F += 1; print(f"FAIL  {label}: no error")
    except Exception as e:
        if msg_sub and msg_sub not in str(e):
            F += 1; print(f"FAIL  {label}: wrong msg -> {e}")
        else:
            P += 1; print(f"PASS  {label}")

# ── desugar_axis ──
ok("list passthrough", B.desugar_axis("k", [0,1,2,59]) == [0,1,2,59])
ok("range [0,60] -> 0..59 (stop-excl)", B.desugar_axis("k", {"range":[0,60]}) == list(range(60)))
ok("range step", B.desugar_axis("k", {"range":[0,60,5]}) == [0,5,10,15,20,25,30,35,40,45,50,55])
ok("str list preserved", B.desugar_axis("m", ["x","xy"]) == ["x","xy"])
raises("range float rejected", lambda: B.desugar_axis("k", {"range":[0,60.0]}), "must be int")
raises("range step 0 rejected", lambda: B.desugar_axis("k", {"range":[0,10,0]}), "step must not be 0")
raises("empty range rejected", lambda: B.desugar_axis("k", {"range":[10,0]}), "is empty")
raises("empty list rejected", lambda: B.desugar_axis("k", []), "is empty")

# ── expand_circuit_grid ──
ok("empty grid -> [{}]", B.expand_circuit_grid({}) == [{}])
ok("single axis", B.expand_circuit_grid({"num_kicks":{"range":[0,3]}}) == [{"num_kicks":0},{"num_kicks":1},{"num_kicks":2}])
mg = B.expand_circuit_grid({"p":[1,2,3],"mixer":["x","xy"]})
ok("multi-axis count 3x2=6", len(mg) == 6)
ok("multi-axis order (first slowest)", mg[0]=={"p":1,"mixer":"x"} and mg[1]=={"p":1,"mixer":"xy"} and mg[2]=={"p":2,"mixer":"x"})

# ── validate_factory_signature ──
def good(*, num_kicks, epsilon, num_qubits, hz_angles, Jzz_angles, init_bit_array): pass
B.validate_factory_signature(good, grid_keys={"num_kicks"}, fixed_keys={"epsilon","num_qubits"},
                             disorder_keys={"hz_angles","Jzz_angles","init_bit_array"})
ok("good signature accepted", True)
raises("missing required flagged",
       lambda: B.validate_factory_signature(good, grid_keys={"num_kicks"}, fixed_keys={"num_qubits"},
                                            disorder_keys={"hz_angles","Jzz_angles","init_bit_array"}),
       "requires ['epsilon']")
raises("unknown key flagged",
       lambda: B.validate_factory_signature(good, grid_keys={"num_kicks","bogus"}, fixed_keys={"epsilon","num_qubits"},
                                            disorder_keys={"hz_angles","Jzz_angles","init_bit_array"}),
       "does not accept")
raises("duplicate across blocks flagged",
       lambda: B.validate_factory_signature(good, grid_keys={"num_kicks","epsilon"}, fixed_keys={"epsilon","num_qubits"},
                                            disorder_keys={"hz_angles","Jzz_angles","init_bit_array"}),
       "both 'grid' and 'fixed'")
def kw(**kwargs): pass
raises("**kwargs rejected by default",
       lambda: B.validate_factory_signature(kw, grid_keys={"a"}, fixed_keys=set(), disorder_keys=set()),
       "**kwargs")
B.validate_factory_signature(kw, grid_keys={"a"}, fixed_keys=set(), disorder_keys=set(), allow_kwargs=True)
ok("**kwargs allowed with opt-out", True)
def poso(a, /, b): pass
raises("positional-only rejected",
       lambda: B.validate_factory_signature(poso, grid_keys={"a","b"}, fixed_keys=set(), disorder_keys=set()),
       "positional-only")
def withdef(*, num_kicks, epsilon=0.03): pass
B.validate_factory_signature(withdef, grid_keys={"num_kicks"}, fixed_keys=set(), disorder_keys=set())
ok("defaulted param optional (unsupplied is OK)", True)

# ── resolve_disorder: file path ──
doc = {"_meta":{"generator":"legacy_npr","master_seed":0,"num_qubits":4,"initial_state":3},
       "instances":{str(s):{"hz_angles":[0.1]*4,"Jzz_angles":[0.2]*4,"init_bit_array":[0,0,0,0]} for s in range(5)}}
fd, path = tempfile.mkstemp(suffix=".json"); os.write(fd, json.dumps(doc).encode()); os.close(fd)
res, meta = B.resolve_disorder({"source":"file","file":path}, [0,1,2,3,4], num_qubits=4, configured_initial_state=3)
ok("file load covers seeds", set(res.keys())=={0,1,2,3,4})
raises("seed coverage fail",
       lambda: B.resolve_disorder({"source":"file","file":path}, [0,1,42], num_qubits=4),
       "missing instances for seed(s) [42]")
raises("num_qubits mismatch",
       lambda: B.resolve_disorder({"source":"file","file":path}, [0], num_qubits=10),
       "num_qubits")
raises("initial_state mismatch",
       lambda: B.resolve_disorder({"source":"file","file":path}, [0], num_qubits=4, configured_initial_state=1),
       "initial_state")
# array-length mismatch
doc2 = {"_meta":{"num_qubits":4},"instances":{"0":{"hz_angles":[0.1,0.2]}}}
fd2, path2 = tempfile.mkstemp(suffix=".json"); os.write(fd2, json.dumps(doc2).encode()); os.close(fd2)
raises("array length mismatch",
       lambda: B.resolve_disorder({"source":"file","file":path2}, [0], num_qubits=4),
       "length 2 != num_qubits 4")

# ── resolve_disorder: generate path (pcg64) ──
def sampler(rng, n):
    return {"hz_angles": rng.uniform(-np.pi, np.pi, n).tolist()}
g1,_ = B.resolve_disorder({"source":"generate","generator":"pcg64","master_seed":0}, [0,1,2], num_qubits=4, sampler=sampler)
g2,_ = B.resolve_disorder({"source":"generate","generator":"pcg64","master_seed":0}, [0,1,2], num_qubits=4, sampler=sampler)
ok("generate pcg64 deterministic (same master_seed)", g1 == g2)
ok("generate distinct per seed", g1[0] != g1[1])
g3,_ = B.resolve_disorder({"source":"generate","generator":"pcg64","master_seed":1}, [0,1,2], num_qubits=4, sampler=sampler)
ok("generate different master_seed -> different ensemble", g1[0] != g3[0])

# ── cross_grid_identity_check ──
# Correct factory: ignores grid for disorder; "circuit" = disorder echo.
def build_good(**kw):  return {"disorder_params": kw["hz_angles"], "kicks": kw["num_kicks"]}
def extract(c):        return c["disorder_params"]
inst = {"hz_angles":[0.1,0.2,0.3,0.4]}
fixed = {"num_qubits":4}
pts = B.expand_circuit_grid({"num_kicks":{"range":[0,60]}})
B.cross_grid_identity_check(build_good, fixed=fixed, instance=inst, grid_points=pts,
                            extract_disorder_params=extract, primary_axis="num_kicks")
ok("cross-grid check passes for pure factory", True)
# Buggy factory: leaks grid (num_kicks) into the disorder params -> drift.
def build_bad(**kw):   return {"disorder_params": [a + kw["num_kicks"] for a in kw["hz_angles"]]}
raises("cross-grid check CATCHES drift (the Q2 bug)",
       lambda: B.cross_grid_identity_check(build_bad, fixed=fixed, instance=inst, grid_points=pts,
                                           extract_disorder_params=extract, primary_axis="num_kicks"),
       "FAILED")

print()
print(f"{'ALL PASS' if F==0 else 'SOME FAILED'}  ({P} passed, {F} failed)")
sys.exit(0 if F == 0 else 1)
