import os
aff = sorted(os.sched_getaffinity(0))
def g(c, f):
    p = f"/sys/devices/system/cpu/cpu{c}/topology/{f}"
    return open(p).read().strip() if os.path.exists(p) else None
sibs = {g(c, "thread_siblings_list") for c in aff}
pkgcore = {(g(c, "physical_package_id"), g(c, "core_id")) for c in aff}
print("n_logical_in_affinity:", len(aff))
print("distinct_physical_cores_by_siblings:", len([s for s in sibs if s]))
print("distinct_physical_cores_by_pkg_core:", len(pkgcore))
print("current_code_div2_says:", len(aff) // 2)
print("sample_sibling_lists:", sorted([s for s in sibs if s])[:6])
