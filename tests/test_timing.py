"""Central wall-clock profiling (broadway.timing.TimingReport) unit tests."""

from __future__ import annotations

import time

from broadway.timing import TimingReport


def test_timing_report_accumulates() -> None:
    r = TimingReport()
    r.add("load", 1.0)
    r.add("load", 2.0)
    r.add("encode", 0.5)
    assert r.elapsed("load") == 3.0
    assert r.elapsed("encode") == 0.5
    assert r.elapsed("missing") == 0.0
    assert r.total() == 3.5
    d = r.as_dict()
    assert d["load"]["seconds"] == 3.0
    assert d["load"]["calls"] == 2
    assert d["load"]["last"] == 2.0
    assert d["encode"]["calls"] == 1


def test_timing_record_context_manager() -> None:
    r = TimingReport()
    with r.record("blk"):
        time.sleep(0.001)
    assert r.elapsed("blk") > 0
    assert r.as_dict()["blk"]["calls"] == 1


def test_timing_as_dict_is_sorted_and_plain() -> None:
    r = TimingReport()
    r.add("z", 0.1)
    r.add("a", 0.2)
    keys = list(r.as_dict())
    assert keys == ["a", "z"]
    # plain data: no nested live objects, all values are numbers
    for entry in r.as_dict().values():
        assert set(entry) == {"seconds", "calls", "last"}
        assert all(isinstance(v, (int, float)) for v in entry.values())
