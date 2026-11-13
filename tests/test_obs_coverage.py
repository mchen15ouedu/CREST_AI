"""obs.get_series coverage: an empty NWIS answer is cached as covered only
when stored rows bracket that window on BOTH sides (NWIS also answers 200
with an empty series when it fails quietly — a gap open towards now is what
that looks like — and a covered window is never asked again); a dry site's
zero rows are an answer; `refresh` asks a covered window again. No network:
fetch_usgs_discharge is faked."""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import pytest

T0, T1 = datetime(2020, 6, 1), datetime(2020, 6, 11)       # long before the provisional tail
SITE = "01234567"


@pytest.fixture
def obs(tmp_path, monkeypatch):
    from hf_data import obs, statecache
    monkeypatch.setattr(statecache, "CACHE_DIR", str(tmp_path))
    return obs


def hourly(a: datetime, b: datetime, q: float = 1.0) -> list:
    return [(a + timedelta(hours=i), q) for i in range(int((b - a).total_seconds() // 3600) + 1)]


def _nwis(monkeypatch, obs, answer) -> list:
    """fetch_usgs_discharge -> answer(a, b); returns the windows asked."""
    asked = []

    def fetch(site, a, b):
        asked.append((a, b))
        return answer(a, b)
    monkeypatch.setattr(obs, "fetch_usgs_discharge", fetch)
    return asked


def test_empty_answer_without_rows_is_asked_again(obs, monkeypatch):
    asked = _nwis(monkeypatch, obs, lambda a, b: [])           # NWIS 200, empty series
    info = {}
    assert obs.get_series(SITE, T0, T1, info=info) == []
    assert info == {"cached": False, "fetched_windows": 1, "empty_windows": 1,
                    "from_store": False}
    assert not os.path.exists(obs._store_path(SITE))           # nothing to keep
    assert obs.get_series(SITE, T0, T1) == [] and asked == [(T0, T1)] * 2
    # NWIS is back: the same window is answered, stored and covered
    _nwis(monkeypatch, obs, hourly)
    assert len(obs.get_series(SITE, T0, T1)) == 241
    assert obs._load_store(SITE)[1] == [[T0, T1]]


def test_dry_site_zero_rows_are_an_answer(obs, monkeypatch):
    asked = _nwis(monkeypatch, obs, lambda a, b: hourly(a, b, 0.0))
    s = obs.get_series(SITE, T0, T1)
    assert len(s) == 241 and {q for _, q in s} == {0.0}
    info = {}
    assert obs.get_series(SITE, T0, T1, info=info) == s and len(asked) == 1
    assert info["cached"] is True and info["from_store"] is True


def test_empty_answer_with_rows_on_one_side_is_asked_again(obs, monkeypatch):
    """Rows before the window and none after it: a gap open towards now, what
    NWIS failing quietly looks like (the site may just as well have stopped
    reporting, but then asking again costs one request)."""
    mid = T0 + timedelta(days=4)
    _nwis(monkeypatch, obs, hourly)
    obs.get_series(SITE, T0, mid)
    asked = _nwis(monkeypatch, obs, lambda a, b: [])           # NWIS 200, empty series
    info = {}
    assert len(obs.get_series(SITE, T0, T1, info=info)) == 97
    assert asked == [(mid, T1)] and info["from_store"] is True
    assert obs._load_store(SITE)[1] == [[T0, mid]]              # not trusted: one side only
    obs.get_series(SITE, T0, T1)
    assert asked == [(mid, T1)] * 2                             # asked again
    # NWIS is back: the hours are answered and covered
    _nwis(monkeypatch, obs, hourly)
    assert len(obs.get_series(SITE, T0, T1)) == 241
    assert obs._load_store(SITE)[1] == [[T0, T1]]
    # the same on the other side: rows after the window, none before it
    obs._save_store(SITE, {t: 1.0 for t, _ in hourly(T1, T1 + timedelta(days=3))},
                    [[T1, T1 + timedelta(days=3)]])
    asked = _nwis(monkeypatch, obs, lambda a, b: [])
    obs.get_series(SITE, T0, T1)
    obs.get_series(SITE, T0, T1)
    assert asked == [(T0, T1)] * 2
    assert obs._load_store(SITE)[1] == [[T1, T1 + timedelta(days=3)]]


def test_empty_answer_between_rows_is_a_gap_in_the_record(obs, monkeypatch):
    """Rows on BOTH sides (within the fetch pad plus a day): the site reports
    around the window, so no row in it is its record — covered, not asked
    again."""
    a, b = T0 + timedelta(days=5), T0 + timedelta(days=8)
    rows = {t: 1.0 for t, _ in hourly(T0, T1) if not a < t < b}    # no record in between
    obs._save_store(SITE, rows, [[T0, a], [b, T1]])
    asked = _nwis(monkeypatch, obs, lambda a_, b_: [])
    assert len(obs.get_series(SITE, T0, T1)) == 241 - 71
    assert asked == [(a, b)]
    assert obs._load_store(SITE)[1] == [[T0, T1]]               # trusted: bracketed
    obs.get_series(SITE, T0, T1)
    assert len(asked) == 1                                      # not asked again
    # the last row before it too far (more than the pad plus a day): not
    # bracketed, asked again
    obs._save_store(SITE, {t: q for t, q in rows.items() if t <= a - timedelta(days=3)
                           or t >= b}, [[T0, a], [b, T1]])
    obs.get_series(SITE, T0, T1)
    obs.get_series(SITE, T0, T1)
    assert asked[1:] == [(a, b)] * 2
    assert obs._load_store(SITE)[1] == [[T0, a], [b, T1]]


def test_answer_with_rows_only_in_one_pad_is_asked_again(obs, monkeypatch):
    """An answer whose only rows lie in the ±1-day pad on one side carries no
    row of the window: covered only when bracketed, as _trusted keeps it —
    no window is marked covered that the next load would drop again."""
    mid = T0 + timedelta(days=4)
    asked = _nwis(monkeypatch, obs, lambda a, b: [(t, q) for t, q in hourly(T0, mid)
                                                  if a - timedelta(days=1) <= t < a])
    obs._save_store(SITE, {t: 1.0 for t, _ in hourly(T0, mid - timedelta(days=1))},
                    [[T0, mid - timedelta(days=1)]])
    obs.get_series(SITE, T0, T1)
    assert asked == [(mid - timedelta(days=1), T1)]
    assert obs._load_store(SITE)[1] == [[T0, mid - timedelta(days=1)]]


def test_quiet_failure_towards_now_leaves_no_hour_behind(obs, monkeypatch):
    """An hourly caller (a 72 h lookback ending now) through 48 h of NWIS
    answering 200 with an empty series: once NWIS is back, every hour of the
    lookback is there — none was covered as answered while it failed."""
    import datetime as _dt
    now0 = datetime(2026, 11, 1, 12)
    clock = {"h": 0}

    class FakeDT(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            t = now0 + timedelta(hours=clock["h"])
            return t.replace(tzinfo=_dt.timezone.utc) if tz else t
    monkeypatch.setattr(obs, "datetime", FakeDT)

    def fetch(site, a, b):
        if 1 <= clock["h"] <= 48:
            return []
        now = now0 + timedelta(hours=clock["h"])
        return hourly(a - timedelta(days=1), min(b + timedelta(days=1), now))
    monkeypatch.setattr(obs, "fetch_usgs_discharge", fetch)
    for h in range(52):
        clock["h"] = h
        now = now0 + timedelta(hours=h)
        s = obs.get_series(SITE, now - timedelta(hours=72), now)
    want = {now - timedelta(hours=72 - i) for i in range(73)}
    assert want <= {t for t, _ in s}


def test_rowless_coverage_of_an_older_store_is_asked_again(obs, monkeypatch):
    obs._save_store(SITE, {}, [[T0, T1]])        # an empty answer the old rule trusted
    asked = _nwis(monkeypatch, obs, hourly)
    info = {}
    assert len(obs.get_series(SITE, T0, T1, info=info)) == 241
    assert asked == [(T0, T1)] and info["from_store"] is False
    rows, cov = obs._load_store(SITE)
    assert len(rows) == 241 and cov == [[T0, T1]]
    # coverage far from any row is dropped too; coverage with rows stands
    far = [T0 - timedelta(days=60), T0 - timedelta(days=50)]
    obs._save_store(SITE, rows, [far, [T0, T1]])
    asked.clear()
    obs.get_series(SITE, far[0], T1)
    assert asked == [(far[0], T0)]
    # rowless coverage next to rows on one side only: dropped (asked again);
    # bracketed by rows on both sides: it stands
    after = [T1 + timedelta(hours=2), T1 + timedelta(days=1)]
    obs._save_store(SITE, rows, [[T0, T1], after])
    asked.clear()
    obs.get_series(SITE, T0, after[1])
    assert asked == [(T1, after[1])]
    late = after[1] + timedelta(hours=3)
    obs._save_store(SITE, {**rows, **{t: 1.0 for t, _ in hourly(late, late + timedelta(days=1))}},
                    [[T0, T1], after, [late, late + timedelta(days=1)]])
    asked.clear()
    obs.get_series(SITE, T0, after[1])
    assert asked == [(T1, after[0])]                            # only the 2 h sliver


def test_refresh_asks_a_covered_window_again(obs, monkeypatch):
    q = {"v": 1.0}
    asked = _nwis(monkeypatch, obs, lambda a, b: hourly(a, b, q["v"]))
    obs.get_series(SITE, T0, T1)
    q["v"] = 2.0
    assert {v for _, v in obs.get_series(SITE, T0, T1)} == {1.0} and len(asked) == 1
    info = {}
    s = obs.get_series(SITE, T0, T1, info=info, refresh=True)
    assert {v for _, v in s} == {2.0} and asked[-1] == (T0, T1)
    assert info["from_store"] is False and info["cached"] is False
    # an empty refresh answer (NWIS failing quietly) drops no covered row
    _nwis(monkeypatch, obs, lambda a, b: [])
    assert obs.get_series(SITE, T0, T1, refresh=True) == s
    assert obs._load_store(SITE)[1] == [[T0, T1]]


def test_fresh_answer_still_replaces_uncovered_rows(obs, monkeypatch):
    phantom = T0 + timedelta(hours=5)
    rows = {t: 1.0 for t, _ in hourly(T0, T1)}
    rows[phantom] = 1087.0                       # a scrubbed provisional value
    obs._save_store(SITE, rows, [])              # uncovered (provisional tail)
    _nwis(monkeypatch, obs, lambda a, b: [(t, q) for t, q in hourly(a, b) if t != phantom])
    s = dict(obs.get_series(SITE, T0, T1))
    assert phantom not in s and len(s) == 240


def test_fetch_error_is_reported_with_the_exception(obs, monkeypatch):
    def boom(site, a, b):
        raise IndexError("list index out of range")             # an answer that fails to parse
    monkeypatch.setattr(obs, "fetch_usgs_discharge", boom)
    info = {}
    assert obs.get_series(SITE, T0, T1, info=info) == []
    assert info["fetch_error"] == "list index out of range"
    assert isinstance(info["fetch_exc"], IndexError) and info["fetched_windows"] == 0
    assert not os.path.exists(obs._store_path(SITE))
