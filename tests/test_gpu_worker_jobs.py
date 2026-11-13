"""The GPU worker leaves protocol-2 and engine-pinned jobs to a compatible worker.

gpu_worker_space/app.py boots (git clones, Gradio) at import, so the pure
`incompatible_job` is lifted out of its source and exercised alone."""
import ast
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def incompatible_job():
    src = open(os.path.join(ROOT, "gpu_worker_space", "app.py"), encoding="utf-8").read()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "incompatible_job")
    ns: dict = {}
    exec(compile(ast.Module([fn], []), "app.py", "exec"), ns)
    return ns["incompatible_job"]


ZERO = "zerogpu:vincewin/CREST_gpu_worker:7"


@pytest.mark.parametrize("spec, why", [
    ({}, None),
    ({"worker_protocol": 1}, None),
    ({"coupling_protocol": 1}, None),
    ({"worker_protocol": 2}, "unsupported_protocol"),
    ({"coupling_protocol": "3"}, "unsupported_protocol"),
    ({"worker_protocol": "x"}, "unsupported_protocol"),
    ({"required_engine_prefix": "jetstream2:"}, "engine_mismatch"),
    ({"required_engine_prefix": "zerogpu:"}, None),
])
def test_incompatible_job(incompatible_job, spec, why):
    assert incompatible_job(spec, ZERO) == why


def test_eventsim_boundary_jobs_are_never_claimed_by_the_gpu_worker(incompatible_job):
    # the spec shape eventsim enqueues for observed-boundary runs
    spec = {"worker_protocol": 2, "required_engine_prefix": "jetstream2:", "dem_res": "1"}
    assert incompatible_job(spec, ZERO) is not None


def test_scan_applies_the_guard():
    src = open(os.path.join(ROOT, "gpu_worker_space", "app.py"), encoding="utf-8").read()
    scan = src[src.index("def _scan():"):src.index("def _gpu_chunk_impl")]
    assert "incompatible_job(spec, IDENT)" in scan
