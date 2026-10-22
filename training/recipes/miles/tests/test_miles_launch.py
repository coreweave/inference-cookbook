# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Unit tests for the miles launcher (no miles, Ray or GPUs needed)."""

from __future__ import annotations

import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

# run_hotload.py imports miles' command helpers at module level; empty stand-ins are enough here
for name in ("miles", "miles.utils", "miles.utils.external_utils", "miles.utils.external_utils.command_utils"):
    sys.modules.setdefault(name, types.ModuleType(name))

spec = importlib.util.spec_from_file_location(
    "miles_launch", Path(__file__).resolve().parents[1] / "run_hotload.py"
)
launch = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = launch
spec.loader.exec_module(launch)


@pytest.mark.parametrize(
    "script,given,expected",
    [
        ("train.py", None, "reset_all"),
        ("train.py", "", "reset_all"),
        ("train.py", "preserve", "preserve"),
        ("train_async.py", None, None),
        ("train_async.py", "reset_all", "reset_all"),  # an explicit choice still goes through
    ],
)
def test_worker_env_resets_the_prompt_cache_for_the_synchronous_loop(script, given, expected) -> None:
    environ = {"CWTRAIN_HOTLOAD_RUN_ID": "r", "HOME": "/root"}
    if given is not None:
        environ["CWTRAIN_HOTLOAD_PROMPT_CACHE_POLICY"] = given
    env = launch.worker_env(script, environ)
    assert env.get("CWTRAIN_HOTLOAD_PROMPT_CACHE_POLICY") == expected
    assert env["CWTRAIN_HOTLOAD_RUN_ID"] == "r" and "HOME" not in env


LAUNCHER = Path(__file__).resolve().parents[1] / "run_hotload.py"
K8S = Path(__file__).resolve().parents[1] / "k8s"
UTILS = "miles.utils.external_utils.command_utils"


def helpers_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    for helper in launch.HELPERS:
        setattr(module, helper, object())
    return module


def load_launcher(monkeypatch: pytest.MonkeyPatch, command_utils: types.ModuleType | None) -> types.ModuleType:
    """Import run_hotload.py against a stand-in for one miles release's command_utils (None: none)."""
    for name in ("miles", "miles.utils", "miles.utils.external_utils"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    # a None entry makes `import` raise ImportError, like a miles that moved the module away
    monkeypatch.setitem(sys.modules, UTILS, command_utils)
    legacy = getattr(command_utils, "legacy", None)
    if legacy is not None:
        monkeypatch.setitem(sys.modules, f"{UTILS}.legacy", legacy)
    else:
        monkeypatch.delitem(sys.modules, f"{UTILS}.legacy", raising=False)
    variant = importlib.util.spec_from_file_location("miles_launch_variant", LAUNCHER)
    module = importlib.util.module_from_spec(variant)
    variant.loader.exec_module(module)
    return module


def test_the_launcher_takes_the_helpers_from_legacy_where_newer_miles_moved_them(monkeypatch) -> None:
    package = types.ModuleType(UTILS)  # newer miles: a package without the helpers, plus .legacy
    package.__path__ = []
    package.legacy = helpers_module(f"{UTILS}.legacy")
    variant = load_launcher(monkeypatch, package)
    assert variant.U is package.legacy and variant.U_PROBLEM == ""


def test_the_launcher_takes_the_helpers_from_command_utils_on_miles_v0_1(monkeypatch) -> None:
    module = helpers_module(UTILS)  # v0.1.x: one module that holds the helpers, no .legacy
    variant = load_launcher(monkeypatch, module)
    assert variant.U is module and variant.U_PROBLEM == ""


def stop_message(variant: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> str:
    """main()'s stop, with proof that it came before any download or conversion."""
    started = []  # checkpoint_paths runs right after the check, before any download or conversion
    monkeypatch.setattr(variant, "checkpoint_paths", lambda *a: started.append(a) or ("r", "m", "/c"))
    monkeypatch.setattr(sys, "argv", ["run_hotload.py"])
    with pytest.raises(SystemExit) as stop:
        variant.main()
    assert started == []
    return str(stop.value)


def test_the_launcher_stops_before_any_download_when_no_module_has_the_helpers(monkeypatch) -> None:
    package = types.ModuleType(UTILS)
    package.__path__ = []
    package.legacy = types.ModuleType(f"{UTILS}.legacy")  # moved again: nothing where we look
    message = stop_message(load_launcher(monkeypatch, package), monkeypatch)
    assert re.search(r"no exec_command_cpu, .*ExecuteTrainConfig.*radixark/miles:v0\.1\.1", message)


def test_the_launcher_stops_with_the_pinned_image_when_command_utils_is_gone(monkeypatch) -> None:
    variant = load_launcher(monkeypatch, None)  # neither import works: no raw traceback
    message = stop_message(variant, monkeypatch)
    assert "cannot be imported" in message and launch.PINNED_IMAGE in message


def test_the_stop_names_a_failed_legacy_import_only_where_miles_has_a_legacy_module(monkeypatch) -> None:
    package = types.ModuleType(UTILS)  # newer layout, but command_utils.legacy cannot be imported
    package.__path__ = []
    variant = load_launcher(monkeypatch, package)
    with pytest.raises(SystemExit, match=r"importing command_utils\.legacy failed"):
        variant.check_helpers(variant.U, variant.U_PROBLEM)
    module = helpers_module(UTILS)  # v0.1.x missing one helper: no legacy module is expected there
    del module.execute_train
    variant = load_launcher(monkeypatch, module)
    with pytest.raises(SystemExit) as stop:
        variant.check_helpers(variant.U, variant.U_PROBLEM)
    assert "execute_train" in str(stop.value) and "legacy" not in str(stop.value)


@pytest.mark.parametrize("name", ["Qwen/Qwen3-4B/", " Qwen/Qwen3-4B ", "Qwen3-4B/"])
def test_trailing_slashes_and_spaces_in_the_model_name_are_ignored(name) -> None:
    assert launch.checkpoint_paths(name, "/work/models") == (
        "Qwen/Qwen3-4B", "Qwen3-4B", "/work/models/Qwen3-4B"
    )


@pytest.mark.parametrize("base_s3_uri", ["", "s3://bucket/models/Llama-3.1-8B"])
@pytest.mark.parametrize("hf_model", ["Qwen/Qwen3-4B", "meta-llama/Llama-3.1-8B/", "Qwen3-8B"])
def test_the_launcher_stages_the_model_where_every_later_step_reads_it(
    tmp_path, monkeypatch, hf_model, base_s3_uri
) -> None:
    module = helpers_module(UTILS)
    calls: list[tuple] = []
    module.exec_command_cpu = lambda cmd: calls.append(("cmd", cmd))
    module.hf_download_dataset = lambda *a, **kw: None
    module.convert_checkpoint = lambda **kw: calls.append(("convert", kw["hf_checkpoint"]))
    module.execute_train = lambda **kw: calls.append(("train", kw["train_args"]))
    module.ExecuteTrainConfig = lambda **kw: kw
    variant = load_launcher(monkeypatch, module)

    def stage_from_s3(uri: str, dest: str) -> None:  # like the real one: a checkpoint lands in dest
        Path(dest).mkdir(parents=True, exist_ok=True)
        Path(dest, "config.json").write_text("{}")
        calls.append(("stage", uri, str(dest)))

    monkeypatch.setattr(variant, "stage_from_s3", stage_from_s3)
    monkeypatch.setattr(variant, "megatron_model_type", lambda name: name)
    models = tmp_path / "models"
    monkeypatch.setattr(sys, "argv", [
        "run_hotload.py", "--model-name", hf_model, "--model-dir", str(models),
        "--data-dir", str(tmp_path / "data"), "--prompt-data", str(tmp_path / "prompts.jsonl"),
        "--work-dir", str(tmp_path), "--base-s3-uri", base_s3_uri,
    ])  # fmt: skip
    revision = "1cfa9a7208912126459214e8b04321603b3df60c"
    monkeypatch.setattr(variant.os, "environ", {"HF_REVISION": revision})
    variant.main()
    checkpoint = launch.checkpoint_paths(hf_model, str(models))[2]
    staged = [c for c in calls if c[0] == "stage"]
    assert staged == ([("stage", base_s3_uri, checkpoint)] if base_s3_uri else [])
    downloads = [c[1] for c in calls if c[0] == "cmd" and c[1].startswith("hf download")]
    repo = launch.checkpoint_paths(hf_model, str(models))[0]
    assert downloads == ([] if base_s3_uri else [
        f"hf download {repo} --revision {revision} --local-dir {checkpoint}"
    ])
    assert ("convert", checkpoint) in calls
    train_args = next(c[1] for c in calls if c[0] == "train")
    assert f"--hf-checkpoint {checkpoint} " in train_args


def launched_train_args(tmp_path, monkeypatch, *options: str) -> str:
    """train.py's arguments from a launcher run with ``options``, against stand-in helpers."""
    module = helpers_module(UTILS)
    calls: list[str] = []
    module.exec_command_cpu = lambda cmd: None
    module.hf_download_dataset = lambda *a, **kw: None
    module.convert_checkpoint = lambda **kw: None
    module.execute_train = lambda **kw: calls.append(kw["train_args"])
    module.ExecuteTrainConfig = lambda **kw: kw
    variant = load_launcher(monkeypatch, module)
    monkeypatch.setattr(variant, "megatron_model_type", lambda name: name)
    monkeypatch.setattr(sys, "argv", [
        "run_hotload.py", "--model-name", "Qwen/Qwen3-4B", "--skip-download",
        "--model-dir", str(tmp_path / "models"), "--data-dir", str(tmp_path / "data"),
        "--prompt-data", str(tmp_path / "prompts.jsonl"), "--work-dir", str(tmp_path), *options,
    ])  # fmt: skip
    monkeypatch.setattr(variant.os, "environ", {})
    variant.main()
    return calls[0]


@pytest.mark.parametrize(
    ("options", "tis", "old_actor"),
    [
        ((), True, False),
        (("--tis", "0"), False, False),
        (("--train-script", "train_async.py"), True, True),
        (("--train-script", "train_async.py", "--async-correction", "none"), True, False),
    ],
)
def test_tis_is_on_in_both_loops_unless_turned_off(
    tmp_path, monkeypatch, options, tis, old_actor
) -> None:
    args = launched_train_args(tmp_path, monkeypatch, *options)
    assert ("--use-tis " in args) is tis
    assert ("--keep-old-actor " in args) is old_actor


def test_the_old_tis_async_correction_says_what_to_set_instead() -> None:
    with pytest.raises(launch.argparse.ArgumentTypeError, match="set ASYNC_CORRECTION=none"):
        launch.async_correction("tis")
    with pytest.raises(launch.argparse.ArgumentTypeError, match="keep_old_actor or none"):
        launch.async_correction("sometimes")
    assert launch.async_correction("none") == "none"


def test_extra_args_that_name_a_generate_function_stop_the_launcher() -> None:
    agent = "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate"
    with pytest.raises(launch.argparse.ArgumentTypeError, match="is not supported"):
        launch.extra_args(f"--save-interval 50 {agent}")
    assert launch.extra_args("--save-interval 50") == "--save-interval 50"


def test_tis_takes_cwtrains_boolean_values_from_run_env() -> None:
    assert launch.on_off("off") is False and launch.on_off("yes") is True
    with pytest.raises(launch.argparse.ArgumentTypeError, match="--tis must be a boolean"):
        launch.on_off("sometimes")
    script = (K8S / "run-trainer.sh").read_text()
    assert '--tis "${USE_TIS:-1}"' in script
    assert "{name: USE_TIS," in (K8S / "trainer-job.yaml").read_text()


def test_run_trainer_hands_the_model_and_its_source_to_the_launcher() -> None:
    script = (K8S / "run-trainer.sh").read_text()
    assert '--model-name "$HF_MODEL"' in script, "the launcher must train the model run.env names"
    assert '--base-s3-uri "${BASE_S3_URI:-}"' in script
    # one place derives the checkpoint directory: the launcher
    assert "MODEL_DIR" not in script and "MODEL_SLUG" not in script
    # MODEL_SLUG only names the staged copy in the bucket; the trainer reads HF_MODEL alone
    job = (K8S / "trainer-job.yaml").read_text()
    assert "{name: HF_MODEL," in job and "{name: BASE_S3_URI," in job and "MODEL_SLUG" not in job
