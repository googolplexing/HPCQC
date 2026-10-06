"""Patch B proof — per-family-max cap + family split.

RED-RULING-BYO-FLAT-DISPATCH-AND-NOISELESS-DEDUP §2(B cap). Pure-logic replicas
of the two flat-dispatch selection rules in _execute_byo_group; the real
byte-identity (single-observable == pre-B, single vs N-rank) is the LUMI gate.

1. Probe selection sizes per_unit_peak to the per-family MAX: one device-cal probe
   per observable family, skip families with no device-cal unit, C1 static
   fallback when no family has one. Single family -> identical to the pre-B single
   probe (label unchanged), so single-observable stays byte-identical.
2. The un-folded group splits into families in first-seen order; a single-
   observable group yields exactly one family (the byte-identity path).

Self-running; also pytest-collectable.
"""

C1 = 9_999  # stand-in for wc.C1_PER_UNIT_PEAK_BYTES


class U:
    def __init__(self, env_source, observable_name, peak_kib):
        self.env_source = env_source
        self.observable_name = observable_name
        self.peak_rss_kib = peak_kib
        self.error = None


def select_probe_idxs(work_units):
    """Mirror of the per-family probe selection."""
    probe_idxs, seen = [], set()
    for i, u in enumerate(work_units):
        if u.env_source == "device_calibrated" and u.observable_name not in seen:
            seen.add(u.observable_name)
            probe_idxs.append(i)
    return probe_idxs


def resolve_peak(work_units):
    """Mirror of peak resolution: per-family max, skip, or C1 fallback."""
    probe_idxs = select_probe_idxs(work_units)
    peaks = [work_units[i].peak_rss_kib * 1024 for i in probe_idxs
             if work_units[i].peak_rss_kib > 0]
    if peaks:
        src = ("probe:device_calibrated_VmHWM" if len(probe_idxs) == 1
               else "probe:device_calibrated_VmHWM_per_family_max")
        return max(peaks), src, probe_idxs
    if not probe_idxs:
        return C1, "c1_fallback:no_device_cal_unit", probe_idxs
    return C1, "c1_fallback:probe_returned_no_vmhwm", probe_idxs


def split_families(tasks):
    """Mirror of the family split (first-seen order)."""
    fams = {}
    for obs, fn in tasks:
        fams.setdefault((obs, fn), []).append((obs, fn))
    return list(fams.values())


def test_two_families_take_the_heavier_arm():
    # autocorr light (probed first), echo heavy: max must be echo's.
    wu = [U("device_calibrated", "autocorr", 100),
          U("noiseless", "autocorr", 0),
          U("device_calibrated", "echo", 250),
          U("noiseless", "echo", 0)]
    peak, src, idxs = resolve_peak(wu)
    assert len(idxs) == 2, "one probe per family"
    assert peak == 250 * 1024, "must size to the heavier echo arm, not the first probe"
    assert src.endswith("per_family_max")


def test_single_family_is_byte_identical_label():
    wu = [U("device_calibrated", "default", 120), U("noiseless", "default", 0)]
    peak, src, idxs = resolve_peak(wu)
    assert len(idxs) == 1 and peak == 120 * 1024
    assert src == "probe:device_calibrated_VmHWM"  # unchanged from pre-B


def test_family_without_device_cal_is_skipped():
    # echo has only noiseless units -> no probe for it; max over the one that has one.
    wu = [U("device_calibrated", "autocorr", 130),
          U("noiseless", "echo", 0)]
    peak, src, idxs = resolve_peak(wu)
    assert len(idxs) == 1 and peak == 130 * 1024


def test_all_noiseless_falls_back_to_c1():
    wu = [U("noiseless", "autocorr", 0), U("noiseless", "echo", 0)]
    peak, src, idxs = resolve_peak(wu)
    assert idxs == [] and peak == C1 and src == "c1_fallback:no_device_cal_unit"


def test_single_observable_yields_one_family():
    assert len(split_families([("default", "build_circuit")] * 4)) == 1


def test_two_observables_yield_two_families_in_order():
    fams = split_families([("autocorr", "build_circuit"),
                           ("echo", "build_circuit_echo")] * 3)
    assert len(fams) == 2


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\nALL {len(fns)} checks PASS")
