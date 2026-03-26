# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Phase-level timing instrumentation.

Carried forward from lumi_vqa vqa_utils.py TimingTracker, extended with
benchmark JSON and AI/ML training record export formats.

Usage:
    timer = TimingTracker()
    timer.mark("hamiltonian_build")
    # ... build hamiltonian ...
    timer.mark("ansatz_build")
    # ... build ansatz ...
    breakdown = timer.finish()
    print(breakdown.to_human_readable())
"""

from __future__ import annotations

import time

from lumi_hpc_qc.types import TimingBreakdown


class TimingTracker:
    """Records wall-clock time between phases of execution.

    Call mark() at each phase transition. The duration of phase N
    is the time between mark(N) and mark(N+1). Call finish() to
    compute the final breakdown.

    Output formats:
      - to_human_readable(): pretty-printed table (stdout logging)
      - to_benchmark_json(): structured dict for HPC benchmarking
      - to_training_record(): flat dict for AI/ML training datasets
    """

    def __init__(self) -> None:
        self._start_time = time.time()
        self._marks: list[tuple[str, float]] = []
        self._current_phase: str | None = None

    def mark(self, phase_name: str) -> None:
        """Record a phase transition.

        The time since the last mark (or since __init__) is attributed
        to the previous phase. The new phase starts now.
        """
        now = time.time()
        self._marks.append((phase_name, now))
        self._current_phase = phase_name

    def finish(self) -> TimingBreakdown:
        """Finalize timing and compute all durations and percentages.

        Must be called after the last phase completes. The time since
        the last mark() is attributed to the final phase.
        """
        now = time.time()
        phases: dict[str, float] = {}

        if not self._marks:
            return TimingBreakdown(
                phases={},
                total_s=now - self._start_time,
                percentages={},
            )

        # Duration of first phase = time from start to first mark
        # (usually near-zero, represents setup before first mark)

        # Duration between consecutive marks
        for i in range(len(self._marks)):
            phase_name = self._marks[i][0]
            if i == 0:
                start = self._start_time
            else:
                start = self._marks[i - 1][1]
            end = self._marks[i][1]
            # Accumulate if same phase name appears multiple times
            phases[phase_name] = phases.get(phase_name, 0.0) + (end - start)

        # Last phase runs until now
        last_name = self._marks[-1][0]
        # The mark records the END of the phase, so nothing to add
        # But if mark is called at the START, we need the remaining time
        # Convention: mark("X") means "X just completed"
        # So finish() just computes total

        total_s = now - self._start_time
        percentages = {
            name: (dur / total_s * 100.0 if total_s > 0 else 0.0)
            for name, dur in phases.items()
        }

        return TimingBreakdown(
            phases=phases,
            total_s=total_s,
            percentages=percentages,
        )
