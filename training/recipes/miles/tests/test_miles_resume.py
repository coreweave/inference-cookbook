# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Native recovery may reset transport numbering only with a fresh serving chain."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from cwtrain.errors import CWTrainError

spec = importlib.util.spec_from_file_location(
    "miles_resume", Path(__file__).resolve().parents[1] / "coreweave_resume.py")
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


@pytest.fixture
def setup_resume(monkeypatch):
    for key, value in {
        "DEPLOYMENT_ID": "deployment", "RUN_ID": "fresh", "WAIT": "1", "VERIFY": "1",
        "PUBLISH_BASELINE": "1", "MIRROR": "1", "PROMPT_CACHE_POLICY": "reset_all",
    }.items():
        monkeypatch.setenv(f"CWTRAIN_HOTLOAD_{key}", value)
    history, objects, requests = [], [], []

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def list_hot_loads(self, deployment):
            assert deployment == "deployment"
            return iter(history)

    def list_objects(**kwargs):
        requests.append(kwargs)
        return {"Contents": objects}

    loader = SimpleNamespace(
        deployment=SimpleNamespace(hot_load=SimpleNamespace(bucket="bucket", path_prefix="snapshots/")),
        s3=SimpleNamespace(list_objects_v2=list_objects))
    monkeypatch.setattr(resume, "InferenceClient", lambda **kwargs: Client())
    monkeypatch.setattr(resume, "HotLoader", lambda *args: loader)
    args = SimpleNamespace(rollout_num_gpus=0,
                           custom_update_weight_post_write_path="cwtrain.integrations.slime.hook")
    return args, history, objects, requests


def test_resume_accepts_fresh_chain_and_ignores_other_runs(setup_resume):
    args, history, _, requests = setup_resume
    history.append(SimpleNamespace(identity="another-v000000"))
    resume.validate_resume(args)
    assert requests == [{"Bucket": "bucket", "Prefix": "snapshots/fresh-v", "MaxKeys": 1}]


@pytest.mark.parametrize("existing", ["history", "objects"])
def test_resume_refuses_reused_transport_identity(setup_resume, existing):
    args, history, objects, _ = setup_resume
    if existing == "history":
        history.append(SimpleNamespace(identity="fresh-v000002"))
    else:
        objects.append({"Key": "snapshots/fresh-v000000/config.json"})
    with pytest.raises(CWTrainError, match="fresh CWTRAIN_HOTLOAD_RUN_ID"):
        resume.validate_resume(args)


@pytest.mark.parametrize("key", ["WAIT", "VERIFY", "PUBLISH_BASELINE", "MIRROR"])
def test_resume_requires_checked_synchronous_publication(setup_resume, monkeypatch, key):
    monkeypatch.setenv(f"CWTRAIN_HOTLOAD_{key}", "0")
    if key == "MIRROR":
        monkeypatch.setenv("CWTRAIN_HOTLOAD_FULL_EVERY", "0")
    with pytest.raises(CWTrainError, match="baseline publication"):
        resume.validate_resume(setup_resume[0])


def test_resume_requires_cache_reset(setup_resume, monkeypatch):
    monkeypatch.setenv("CWTRAIN_HOTLOAD_PROMPT_CACHE_POLICY", "preserve")
    with pytest.raises(CWTrainError, match="reset_all"):
        resume.validate_resume(setup_resume[0])


@pytest.mark.parametrize("field,value", [("rollout_num_gpus", 2),
                                         ("custom_update_weight_post_write_path", "other.hook")])
def test_resume_keeps_other_topologies_unsupported(setup_resume, field, value):
    setattr(setup_resume[0], field, value)
    with pytest.raises(CWTrainError, match="zero local rollout GPUs"):
        resume.validate_resume(setup_resume[0])


@pytest.mark.parametrize("path_prefix,expected", [("", "fresh-v"), ("/snapshots/", "snapshots/fresh-v")])
def test_resume_checks_canonical_storage_prefix(setup_resume, path_prefix, expected):
    args, _, _, requests = setup_resume
    resume.HotLoader(None, None).deployment.hot_load.path_prefix = path_prefix
    resume.validate_resume(args)
    assert requests[0]["Prefix"] == expected
