"""Batched archive commits.

These tests exercise the "many error traces at once" path: pending traces are
committed in batches of at most 8 in one SQLite transaction, while per-trace
bad data and whole-batch failures keep their existing guarantees.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import threading
import time
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import SpanRecord  # noqa: E402
from app.sampler import ARCHIVE_BATCH_SIZE, TraceSampler  # noqa: E402
from app.storage import ArchiveStore  # noqa: E402


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FlakyBatchStore(ArchiveStore):
    """Real store whose batch transactions can be made to fail on demand."""

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.fail_next_batches = 0
        self.batch_calls = 0

    def write_batch(self, items):
        self.batch_calls += 1
        if self.fail_next_batches:
            self.fail_next_batches -= 1
            raise sqlite3.OperationalError("database is locked")
        return super().write_batch(items)


class BlockingBatchStore(ArchiveStore):
    """Batch store that parks inside the first write until released."""

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.batch_calls = 0
        self._calls_lock = threading.Lock()

    def write_batch(self, items):
        with self._calls_lock:
            self.batch_calls += 1
        self.entered.set()
        self.release.wait(timeout=5)
        return super().write_batch(items)


def span(tid: str, error: bool = True, start_time_ns: str = "1700000000000000000"):
    return SpanRecord(
        trace_id=tid,
        span_id=uuid.uuid4().hex[:16],
        parent_span_id=None,
        name="op",
        kind="SERVER",
        status_code="ERROR" if error else "UNSET",
        status_message="boom" if error else "",
        is_error=error,
        start_time_ns=start_time_ns,
        end_time_ns="1700000000500000000",
        duration_ns=500_000_000,
        service_name="svc",
        event_count=0,
        link_count=0,
        resource_json="{}",
        scope_json="{}",
        attributes_json="{}",
        raw_json="{}",
    )


def new_tid() -> str:
    return uuid.uuid4().hex


def make_sampler(store, clock, **kwargs):
    kwargs.setdefault("window_seconds", 2.0)
    kwargs.setdefault("capacity", 32)
    kwargs.setdefault("max_spans_per_trace", 64)
    return TraceSampler(store, clock=clock, **kwargs)


def expire_all(sampler, clock, count):
    """Ingest ``count`` error traces and let their windows close."""
    tids = [new_tid() for _ in range(count)]
    for tid in tids:
        sampler.ingest([span(tid)])
    clock.advance(2.0)
    sampler.tick()
    return tids


@pytest.fixture()
def clock():
    return FakeClock()


@pytest.fixture()
def store(tmp_path):
    store = ArchiveStore(str(tmp_path / "traces.db"))
    yield store
    store.close()


def test_batch_size_is_eight():
    assert ARCHIVE_BATCH_SIZE == 8


def test_eight_pending_traces_commit_in_one_transaction(store, clock):
    sampler = make_sampler(store, clock)
    sampler.pause()
    tids = [new_tid() for _ in range(8)]
    for tid in tids:
        sampler.ingest([span(tid)])
    clock.advance(2.0)
    sampler.tick()  # expire while paused: nothing written yet
    assert store.count() == 0
    assert len(sampler.snapshot()["pendingTraces"]) == 8

    sampler.resume()
    sampler.tick()

    status = sampler.snapshot()
    assert store.count() == 8
    assert status["totals"]["traces_archived"] == 8
    assert status["totals"]["archive_commits"] == 1
    assert status["pendingTraces"] == []
    assert status["slotsUsed"] == 0
    for tid in tids:
        assert store.get_trace(tid) is not None


def test_more_than_eight_traces_use_multiple_batches(store, clock):
    sampler = make_sampler(store, clock)
    tids = expire_all(sampler, clock, 10)
    status = sampler.snapshot()
    assert store.count() == 10
    assert status["totals"]["traces_archived"] == 10
    assert status["totals"]["archive_commits"] == 2
    assert {t for t in tids if store.get_trace(t)} == set(tids)


def test_short_batch_is_written_immediately(store, clock):
    sampler = make_sampler(store, clock)
    expire_all(sampler, clock, 3)
    status = sampler.snapshot()
    assert status["totals"]["traces_archived"] == 3
    assert status["totals"]["archive_commits"] == 1
    assert status["pendingTraces"] == []


def test_bad_trace_is_isolated_but_rest_of_batch_commits(store, clock):
    sampler = make_sampler(store, clock)
    sampler.pause()
    good = [new_tid() for _ in range(3)]
    bad = new_tid()
    for tid in good:
        sampler.ingest([span(tid)])
    sampler.ingest([span(bad, start_time_ns="not-a-nanosecond")])
    clock.advance(2.0)
    sampler.tick()
    assert len(sampler.snapshot()["pendingTraces"]) == 4

    sampler.resume()
    sampler.tick()

    status = sampler.snapshot()
    # The three healthy traces landed in the same transaction...
    assert status["totals"]["traces_archived"] == 3
    assert status["totals"]["archive_commits"] == 1
    for tid in good:
        assert store.get_trace(tid) is not None
    # ...while the corrupt one kept its pending snapshot and its slot.
    assert store.get_trace(bad) is None
    pending = status["pendingTraces"]
    assert len(pending) == 1
    assert pending[0]["traceId"] == bad
    assert pending[0]["attempts"] == 1
    assert "ValueError" in pending[0]["lastError"]
    assert status["slotsUsed"] == 1

    # Fixing the data lets the normal retry backoff archive it.
    sampler._pending[bad].spans = [span(bad)]
    clock.advance(0.6)
    sampler.tick()
    assert store.get_trace(bad) is not None
    status = sampler.snapshot()
    assert status["totals"]["traces_archived"] == 4
    assert status["totals"]["archive_commits"] == 2
    assert status["pendingTraces"] == []


def test_bad_trace_keeps_its_previous_archive(store, clock):
    tid = new_tid()
    store.write_trace(tid, [span(tid), span(tid, error=False)])

    other = new_tid()
    result = store.write_batch(
        [
            (other, [span(other)]),
            (tid, [span(tid, start_time_ns="broken")]),
        ]
    )

    assert result.failed and tid in result.failed
    assert result.committed == [other]
    assert result.commits == 1
    # The old archived version is untouched by the failed re-write.
    archived = store.get_trace(tid)
    assert archived is not None
    assert archived["span_count"] == 2


def test_whole_batch_failure_keeps_everything_for_retry(tmp_path, clock):
    flaky = FlakyBatchStore(str(tmp_path / "traces.db"))
    try:
        sampler = make_sampler(flaky, clock)
        old = new_tid()
        flaky.write_trace(old, [span(old)])

        flaky.fail_next_batches = 1
        tids = expire_all(sampler, clock, 3)

        status = sampler.snapshot()
        assert flaky.batch_calls == 1
        assert status["totals"]["traces_archived"] == 0
        assert status["totals"]["archive_commits"] == 0
        assert status["totals"]["traces_write_failed"] == 3
        assert len(status["pendingTraces"]) == 3
        assert status["slotsUsed"] == 3
        for tid in tids:
            assert flaky.get_trace(tid) is None
        # Existing archives are still readable after the failed batch.
        assert flaky.get_trace(old) is not None

        clock.advance(1.0)  # past the first backoff
        sampler.tick()
        status = sampler.snapshot()
        assert status["totals"]["traces_archived"] == 3
        assert status["totals"]["archive_commits"] == 1
        assert status["pendingTraces"] == []
        assert all(flaky.get_trace(tid) for tid in tids)
    finally:
        flaky.close()


def test_overlapping_ticks_archive_each_pending_trace_once(tmp_path, clock):
    blocking = BlockingBatchStore(str(tmp_path / "traces.db"))
    try:
        sampler = make_sampler(blocking, clock)
        tids = [new_tid() for _ in range(4)]
        for tid in tids:
            sampler.ingest([span(tid)])
        clock.advance(2.0)

        worker = threading.Thread(target=sampler.tick, name="bg-tick")
        worker.start()
        assert blocking.entered.wait(timeout=5)

        # This is the inline tick the resume page performs while the worker is
        # already committing the same pending traces.
        started = time.monotonic()
        sampler.tick()
        assert time.monotonic() - started < 2.0

        blocking.release.set()
        worker.join(timeout=5)
        assert not worker.is_alive()

        status = sampler.snapshot()
        assert blocking.batch_calls == 1
        assert status["totals"]["traces_archived"] == 4
        assert status["totals"]["archive_commits"] == 1
        assert len({t for t in tids}) == 4
        assert all(blocking.get_trace(tid) for tid in tids)
        assert status["slotsUsed"] == 0
    finally:
        blocking.release.set()
        blocking.close()


def test_sqlite_durability_pragmas_are_unchanged(store):
    assert store._conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    # synchronous=FULL (2)
    assert store._conn.execute("PRAGMA synchronous").fetchone()[0] == 2


def test_write_batch_empty_is_a_noop(store):
    result = store.write_batch([])
    assert result.committed == [] and result.failed == {} and result.commits == 0
