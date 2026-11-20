"""The export parity gate's tolerance check (no models needed)."""

from __future__ import annotations

from tsd.infer.export import PARITY_TOLERANCE, ParityReport


def test_parity_gate_reports_mismatches() -> None:
    good = ParityReport("a", "b", 8, (640, 640), 0.04, 0.0, 1e-5, 0.0, 50, 50, 50, 0, 1e-6)
    assert good.failures() == []
    bad = ParityReport("a", "b", 8, (640, 640), 2.0, 0.1, 0.2, 0.01, 50, 40, 60, 20, 0.1)
    assert len(bad.failures()) == 4
    assert bad.as_dict()["failures"] == bad.failures()
    assert PARITY_TOLERANCE["min_match_rate"] == 0.98
