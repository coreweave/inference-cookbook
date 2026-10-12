# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""The Kubernetes steps an external user follows: run.env renders every manifest, nothing is left
unset or half filled in, the Job cannot start without its Secrets, and the kit script explains
itself."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

RECIPE = Path(__file__).resolve().parents[1]
K8S = RECIPE / "k8s"
PLACEHOLDER = re.compile(r"<[A-Za-z][A-Za-z0-9 _-]*>")
EXTRA = "EXTRA_TRAIN_ARGS"
needs_envsubst = pytest.mark.skipif(shutil.which("envsubst") is None, reason="render.sh needs envsubst")


def filled_run_env(tmp_path: Path, **overrides: str) -> Path:
    """run.env.example with every <placeholder> filled in, as a user's run.env would be."""
    text = PLACEHOLDER.sub("filled", (K8S / "run.env.example").read_text())
    path = tmp_path / "run.env"
    path.write_text(text + "".join(f"{key}={value}\n" for key, value in overrides.items()))
    return path


def render(env_file: Path, manifest: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(K8S / "render.sh"), str(env_file), str(K8S / manifest)],
        env={"PATH": os.environ["PATH"]}, capture_output=True, text=True, check=False,
    )


def test_run_env_example_can_be_sourced_and_names_a_new_run_each_time() -> None:
    # quoted placeholders are values to bash, not redirections
    result = subprocess.run(
        ["bash", "-c", 'set -euo pipefail; set -a; . "$1"; printf "%s" "$RUN_ID"', "-",
         str(K8S / "run.env.example")],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert re.fullmatch(r"qwen3-4b-grpo-\d{8}T\d{6}Z-\d+", result.stdout)


@needs_envsubst
@pytest.mark.parametrize("manifest", sorted(path.name for path in K8S.glob("*.yaml")))
def test_every_manifest_renders_from_a_filled_in_run_env(tmp_path: Path, manifest: str) -> None:
    result = render(filled_run_env(tmp_path), manifest)
    assert result.returncode == 0, result.stderr
    assert "${" not in result.stdout


@needs_envsubst
def test_render_refuses_a_placeholder_left_in_place(tmp_path: Path) -> None:
    env = tmp_path / "run.env"
    shutil.copy(K8S / "run.env.example", env)
    result = render(env, "trainer-job.yaml")
    assert result.returncode == 1 and result.stdout == ""
    assert "fill in" in result.stderr and "ROLLOUT_BASE_URL" in result.stderr
    # step 1 needs only NAMESPACE, so it works before the rest is filled in
    assert render(env, "namespace.yaml").returncode == 0


@needs_envsubst
def test_render_refuses_a_variable_the_env_file_does_not_set(tmp_path: Path) -> None:
    env = tmp_path / "run.env"
    env.write_text("NAMESPACE=ns\n")
    result = render(env, "trainer-job.yaml")
    assert result.returncode == 1 and result.stdout == ""
    assert "does not set" in result.stderr and "DEPLOYMENT_ID" in result.stderr


@needs_envsubst
def test_free_form_arguments_may_carry_tags(tmp_path: Path) -> None:
    # "<qwen size>" reads as a run.env placeholder anywhere but in EXTRA_* arguments
    value = "'--tag <qwen size> --stop </answer>'"
    result = render(filled_run_env(tmp_path, **{EXTRA: value}), "trainer-job.yaml")
    assert result.returncode == 0, result.stderr
    assert "--tag <qwen size> --stop </answer>" in result.stdout


@needs_envsubst
def test_the_hot_load_settings_come_from_run_env(tmp_path: Path) -> None:
    settings = {"WAIT": "0", "TIMEOUT": "3600", "VERIFY": "1", "PROMPT_CACHE_POLICY": "preserve"}
    env = filled_run_env(tmp_path, **{f"HOTLOAD_{key}": value for key, value in settings.items()})
    rendered = render(env, "trainer-job.yaml").stdout
    for key, value in settings.items():
        assert f'{{name: CWTRAIN_HOTLOAD_{key}, value: "{value}"}}' in rendered


def test_the_trainer_does_not_start_without_the_api_token() -> None:
    job = (K8S / "trainer-job.yaml").read_text()
    token = job[job.index("- name: COREWEAVE_API_TOKEN"):].splitlines()[1]
    assert "secretKeyRef" in token and "optional" not in token


def test_the_example_deployment_sets_no_engine_env_or_internal_options() -> None:
    runtime = json.loads((RECIPE / "deployment.example.json").read_text())["runtime"]
    assert "engineEnv" not in runtime
    assert not {"enable-rl", "load-format"} & set(runtime["engineConfig"])


def kit_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A copy of make-kit.sh in a repository layout, an empty wheel dir, and a recording kubectl."""
    script = tmp_path / "recipes" / "training" / RECIPE.name / "k8s" / "make-kit.sh"
    script.parent.mkdir(parents=True)
    shutil.copy(K8S / "make-kit.sh", script)
    dist = tmp_path / "libs" / "cwtrain" / "dist"
    dist.mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    kubectl = bin_dir / "kubectl"
    kubectl.write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> {calls}\ncase "$*" in *"-f -"*) cat >/dev/null;; esac\n')
    kubectl.chmod(0o755)
    return script, dist, calls


def run_kit(script: Path, tmp_path: Path, **extra_env: str) -> subprocess.CompletedProcess[str]:
    env = {"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin", **extra_env}
    return subprocess.run(
        ["bash", str(script), "ctx", "ns", "kit"], env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, check=False,
    )


def test_make_kit_says_how_to_build_the_missing_wheel(tmp_path: Path) -> None:
    script, _, _ = kit_tree(tmp_path)
    result = run_kit(script, tmp_path)
    assert result.returncode == 1
    assert "uv build --wheel" in result.stderr


def test_make_kit_bundles_the_newest_wheel(tmp_path: Path) -> None:
    script, dist, calls = kit_tree(tmp_path)
    older, newer = dist / "cwtrain-0.10.0-py3-none-any.whl", dist / "cwtrain-0.2.0-py3-none-any.whl"
    for wheel, mtime in ((older, 1_700_000_000), (newer, 1_800_000_000)):
        wheel.write_bytes(b"")
        os.utime(wheel, (mtime, mtime))
    result = run_kit(script, tmp_path)
    assert result.returncode == 0, result.stderr
    assert f"--from-file={newer.name}=" in calls.read_text() and older.name not in calls.read_text()


SECRET = "never-printed-4f2c"


def test_storage_secret_refuses_a_placeholder_region(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    kubectl = bin_dir / "kubectl"
    kubectl.write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> {calls}\ncat >/dev/null\n')
    kubectl.chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "AWS_ACCESS_KEY_ID": "id", "AWS_SECRET_ACCESS_KEY": SECRET}
    script = [str(K8S / "make-storage-secret.sh"), "ctx", "ns", "secret"]

    def run(region: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", *script], env={**env, "AWS_REGION": region}, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, check=False,
        )

    result = run("<availability zone>")
    assert result.returncode == 1 and "placeholder" in result.stderr
    assert not calls.exists()  # refused before kubectl ran
    env["AWS_ACCESS_KEY_ID"] = "<access key>"
    result = run("US-EAST-04A")
    assert result.returncode == 1 and "AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY" in result.stderr
    assert "<access key>" not in result.stderr and not calls.exists()
    env["AWS_ACCESS_KEY_ID"] = "id"
    result = run("US-EAST-04A")
    assert result.returncode == 0, result.stderr
    assert "create secret generic secret" in calls.read_text()
    assert SECRET not in result.stdout + result.stderr


def readme_block(heading: str, index: int = 0) -> str:
    """The index-th bash block under a README heading."""
    text = (RECIPE / "README.md").read_text()
    section = text[text.index(heading):]
    start = -1
    for _ in range(index + 1):
        start = section.index("```bash\n", start + 1)
    start += len("```bash\n")
    return section[start : section.index("\n```", start)]


# two runs of this deployment, as `cwtrain hotload list` prints them (newest first)
HOT_LOADS = ["run-b-v000000", "run-a-v000001", "run-a-v000000"]


def run_cleanup(tmp_path: Path, *, cwtrain_exit: int = 0, hotload_prefix_file: bool = True,
                from_k8s: bool = False, git_repo: bool = True, shell: dict[str, str] | None = None,
                deployment_json: str | None = None, hot_loads: list[str] = HOT_LOADS,
                **run_env: str) -> tuple[subprocess.CompletedProcess[str], str]:
    """Run the README's cleanup block in a repository with recording cwtrain, kubectl and aws.

    shell holds variables already exported in the user's shell, run_env the run.env overrides,
    hot_loads the identities that the deployment hot-loaded."""
    k8s = tmp_path / "recipes" / "training" / RECIPE.name / "k8s"
    k8s.mkdir(parents=True)
    if git_repo:
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    defaults = {"BUCKET": "bkt", "BUCKET_ZONE": "US-EAST-04A", "NAMESPACE": "ns", "DEPLOYMENT_ID": "dep-1",
                "JOB_NAME": "job", "STORAGE_SECRET": "store", "ADMIN_STORAGE_SECRET": "admin",
                "INFERENCE_KEY_SECRET": "ikey", "API_TOKEN_SECRET": "token", "KIT_NAME": "kit"}
    filled_run_env(k8s, **{**defaults, **run_env})  # writes k8s/run.env
    if hotload_prefix_file:
        good = json.dumps({"hotLoad": {"pathPrefix": "hotload"}})
        (k8s / "deployment.run.json").write_text(good if deployment_json is None else deployment_json)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    # a failing cwtrain says why on stderr, as it does for a refused delete or a missing token
    refuse = f'echo "error: permission_denied: not allowed" >&2; exit {cwtrain_exit}' if cwtrain_exit else ":"
    listed = tmp_path / "hot_loads.json"
    listed.write_text(json.dumps([{"spec": {"identity": identity}} for identity in hot_loads]))
    cwtrain = f'case "$1 $2" in "hotload list") cat {listed};; "deployment delete") {refuse};; esac'
    for tool, reply in (("cwtrain", cwtrain), ("kubectl", ":"), ("aws", ":")):
        (bin_dir / tool).write_text(f'#!/bin/sh\nprintf "{tool} %s\\n" "$*" >> {calls}\n{reply}\n')
        (bin_dir / tool).chmod(0o755)
    (bin_dir / "python3").symlink_to(sys.executable)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), **(shell or {})}
    result = subprocess.run(["bash", "-c", readme_block("### Cleaning up")], cwd=k8s if from_k8s else tmp_path,
                            env=env, capture_output=True, text=True, check=False)  # fmt: skip
    return result, calls.read_text() if calls.exists() else ""


@pytest.mark.parametrize("from_k8s", [False, True], ids=["from the repository root", "from k8s"])
def test_the_cleanup_block_deletes_the_deployment_recipe_objects_and_prefixes(tmp_path: Path, from_k8s) -> None:
    result, calls = run_cleanup(tmp_path, from_k8s=from_k8s)
    assert result.returncode == 0, result.stderr
    lines = calls.splitlines()
    assert lines[:2] == [  # the runs are listed while the deployment exists; a second cleanup still passes
        "cwtrain hotload list --deployment dep-1 --limit 100000",
        "cwtrain deployment delete dep-1 --ignore-not-found",
    ]
    assert [line for line in lines if line.startswith("kubectl")] == [
        "kubectl -n ns delete job cwtrain-job cwtrain-stage-job cwtrain-make-bucket --ignore-not-found",
        "kubectl -n ns delete secret store admin ikey token --ignore-not-found",
        "kubectl -n ns delete configmap kit --ignore-not-found",
        "kubectl -n ns delete serviceaccount cwtrain-trainer --ignore-not-found",
    ]  # only the recipe's objects; the namespace may hold other work
    removed = [line.split(" --endpoint-url")[0] for line in lines if line.startswith("aws s3 rm")]
    assert removed == [  # only this deployment's runs: every deployment shares the hot-load prefix
        "aws s3 rm --recursive s3://bkt/hotload/ --exclude * --include run-a-v* --include _manifests/run-a-v*",
        "aws s3 rm --recursive s3://bkt/hotload/ --exclude * --include run-b-v* --include _manifests/run-b-v*",
        "aws s3 rm --recursive s3://bkt/models/Qwen3-4B/",
        "aws s3 rm --recursive s3://bkt/models/_probe/",
    ]


@pytest.mark.parametrize(
    ("case", "error"),
    [({"BUCKET_ZONE": "'<availability zone>'"}, "fill in run.env"),
     ({"hotload_prefix_file": False, "HOTLOAD_PREFIX": ""}, "set HOTLOAD_PREFIX"),
     ({"BUCKET": "'<bucket>'"}, "fill in run.env"),
     ({"DEPLOYMENT_ID": "'<deployment id>'"}, "fill in run.env")],
    ids=["zone placeholder", "no hot-load prefix", "bucket placeholder", "deployment placeholder"],
)  # fmt: skip
def test_the_cleanup_block_touches_nothing_until_everything_is_set(tmp_path: Path, case, error) -> None:
    result, calls = run_cleanup(tmp_path, **case)
    assert result.returncode != 0 and error in result.stderr, result.stderr
    assert calls == ""


def test_the_cleanup_block_is_valid_bash() -> None:
    assert subprocess.run(["bash", "-n", "-c", readme_block("### Cleaning up")]).returncode == 0


def test_the_cleanup_block_stops_when_the_deployment_is_not_deleted(tmp_path: Path) -> None:
    result, calls = run_cleanup(tmp_path, cwtrain_exit=1)
    assert result.returncode != 0 and "permission_denied" in result.stderr
    assert calls.splitlines() == [  # nothing else
        "cwtrain hotload list --deployment dep-1 --limit 100000",
        "cwtrain deployment delete dep-1 --ignore-not-found",
    ]


def test_the_cleanup_block_leaves_the_shared_prefix_alone_without_hot_loads(tmp_path: Path) -> None:
    result, calls = run_cleanup(tmp_path, hot_loads=[])
    assert result.returncode == 0, result.stderr
    assert "found no hot-loads of dep-1" in result.stderr
    assert "s3://bkt/hotload/" not in calls  # never the whole prefix that other deployments share


@pytest.mark.parametrize("prefix", ["hotload//", "//hotload", "/hotload/"])
def test_the_cleanup_block_strips_every_outer_slash(tmp_path: Path, prefix: str) -> None:
    # the bucket Job and the API strip them all, so cleanup must target the same objects
    result, calls = run_cleanup(tmp_path, hotload_prefix_file=False, HOTLOAD_PREFIX=prefix)
    assert result.returncode == 0, result.stderr
    assert [line.split()[4] for line in calls.splitlines() if line.startswith("aws s3 rm")][0] == "s3://bkt/hotload/"


@pytest.mark.parametrize("prefix", ["/", "///"])
def test_the_cleanup_block_refuses_a_prefix_of_slashes(tmp_path: Path, prefix: str) -> None:
    result, calls = run_cleanup(tmp_path, hotload_prefix_file=False, HOTLOAD_PREFIX=prefix)
    assert result.returncode != 0 and "HOTLOAD_PREFIX" in result.stderr
    assert calls == ""


def test_run_trainer_checks_snapshot_access_before_training() -> None:
    script = (K8S / "run-trainer.sh").read_text()
    check = re.search(r"^(?!\s*#).*-m cwtrain\.integrations\.preflight\s*$", script, re.M)
    launch = re.search(r"^\s*exec\b", script, re.M)
    assert check and launch and check.start() < launch.start(), "the access check must run before training"


@needs_envsubst
def test_the_trainer_job_can_run_as_the_trainer_service_account(tmp_path: Path) -> None:
    result = render(filled_run_env(tmp_path, TRAINER_SERVICE_ACCOUNT="trainer-sa", ORG_ID="org1"), "trainer-job.yaml")
    assert result.returncode == 0, result.stderr
    assert 'serviceAccountName: "trainer-sa"' in result.stdout
    assert "https://api.coreweave.com/v1/cwobject/temporary-credentials/oidc/org1" in result.stdout
    assert 'audience: "https://coreweave.com/iam"' in result.stdout


def test_the_create_block_creates_the_deployment_and_waits(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    (bin_dir / "cwtrain").write_text(f'#!/bin/sh\nprintf "cwtrain %s\\n" "$*" >> {calls}\n')
    (bin_dir / "cwtrain").chmod(0o755)
    result = subprocess.run(["bash", "-c", readme_block("### Creating the deployment", 1)], cwd=tmp_path,
                            env={"PATH": f"{bin_dir}:/usr/bin:/bin"}, capture_output=True, text=True,
                            check=False)  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert calls.read_text() == "cwtrain deployment create --file deployment.run.json --wait\n"


@pytest.mark.parametrize(("value", "seen"), [("", "unset"), (" ", "unset"), ("900", "900")])
def test_run_trainer_leaves_an_empty_hot_load_timeout_to_cwtrain(value: str, seen: str) -> None:
    script = (K8S / "run-trainer.sh").read_text()
    guard = re.search(r"^hotload_timeout=.*\n.*unset CWTRAIN_HOTLOAD_TIMEOUT$", script, re.M)
    assert guard, "run-trainer.sh must drop a blank CWTRAIN_HOTLOAD_TIMEOUT"
    out = subprocess.run(
        ["bash", "-c", guard.group(0) + '\nprintf %s "${CWTRAIN_HOTLOAD_TIMEOUT-unset}"'],
        env={"PATH": "/usr/bin:/bin", "CWTRAIN_HOTLOAD_TIMEOUT": value},
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    assert out == seen
    launch = re.search(r"^(?!\s*#).*(?:\bexec\b|run_hotload\.py)", script, re.M)  # first non-comment launch line
    assert launch and guard.start() < launch.start(), "the guard must run before the launcher"


def test_the_bucket_policy_command_applies_the_git_ignored_copy() -> None:
    text = (RECIPE / "README.md").read_text()
    assert "file://bucket-policy.json" not in text
    assert "file://k8s/bucket-policy.run.json" in text
    assert "recipes/**/k8s/*.run.json" in (RECIPE.parents[2] / ".gitignore").read_text()


def test_make_kit_skips_an_empty_patches_directory(tmp_path: Path) -> None:
    if not (RECIPE / "patches").is_dir():
        pytest.skip("this recipe ships no patches")
    script, dist, calls = kit_tree(tmp_path)
    (dist / "cwtrain-0.1.0-py3-none-any.whl").write_bytes(b"")
    patches = script.parent.parent / "patches"
    patches.mkdir()
    result = run_kit(script, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "*.patch" not in calls.read_text()
    (patches / "0001-example.patch").write_text("diff\n")
    assert run_kit(script, tmp_path).returncode == 0
    assert "--from-file=0001-example.patch=" in calls.read_text()


def test_make_kit_names_a_cwtrain_wheel_that_does_not_exist(tmp_path: Path) -> None:
    script, dist, _ = kit_tree(tmp_path)
    (dist / "cwtrain-0.1.0-py3-none-any.whl").write_bytes(b"")
    result = run_kit(script, tmp_path, CWTRAIN_WHEEL=str(tmp_path / "typo.whl"))
    assert result.returncode == 1
    assert "CWTRAIN_WHEEL" in result.stderr and "uv build" not in result.stderr


def test_the_cleanup_block_stops_when_it_cannot_reach_the_recipe(tmp_path: Path) -> None:
    # outside a checkout the cd fails; values the shell still holds from an earlier run.env must not
    # be used instead
    shell = {"DEPLOYMENT_ID": "dep-1", "HOTLOAD_PREFIX": "stale/prefix", "NAMESPACE": "ns", "BUCKET": "bkt",
             "BUCKET_ZONE": "US-EAST-04A", "MODEL_PREFIX": "models", "MODEL_SLUG": "m", "JOB_NAME": "job",
             "KIT_NAME": "kit", "STORAGE_SECRET": "s", "INFERENCE_KEY_SECRET": "i", "API_TOKEN_SECRET": "t"}
    result, calls = run_cleanup(tmp_path, git_repo=False, shell=shell)
    assert result.returncode != 0
    assert calls == ""


def test_the_cleanup_block_prefers_the_deployment_file_to_a_prefix_left_in_the_shell(tmp_path: Path) -> None:
    result, calls = run_cleanup(tmp_path, shell={"HOTLOAD_PREFIX": "old/prefix"})
    assert result.returncode == 0, result.stderr
    removed = [line.split()[4] for line in calls.splitlines() if line.startswith("aws s3 rm")]
    assert removed[0] == "s3://bkt/hotload/" and "old/prefix" not in calls


def test_the_cleanup_block_refuses_an_empty_model_prefix(tmp_path: Path) -> None:
    result, calls = run_cleanup(tmp_path, MODEL_PREFIX="")
    assert result.returncode != 0 and "MODEL_PREFIX" in result.stderr
    assert calls == ""


def test_the_parameters_block_keeps_a_filled_in_deployment_file(tmp_path: Path) -> None:
    k8s = tmp_path / "recipe" / "k8s"
    k8s.mkdir(parents=True)
    shutil.copy(RECIPE / "deployment.example.json", k8s.parent / "deployment.example.json")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "cwtrain").write_text("#!/bin/sh\nprintf '{}'\n")
    (bin_dir / "cwtrain").chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin"}

    def run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(["bash", "-c", readme_block("### Creating the deployment")], cwd=k8s, env=env,
                              capture_output=True, text=True, check=False)  # fmt: skip

    assert run().returncode == 0
    assert (k8s / "deployment.run.json").read_text() == (RECIPE / "deployment.example.json").read_text()
    (k8s / "deployment.run.json").write_text('{"name": "mine"}')
    run()
    assert (k8s / "deployment.run.json").read_text() == '{"name": "mine"}'


@needs_envsubst
def test_api_headers_may_carry_tags(tmp_path: Path) -> None:
    value = """'{"X-Trace":"<trace-id>"}'"""
    result = render(filled_run_env(tmp_path, CWTRAIN_API_HEADERS=value), "trainer-job.yaml")
    assert result.returncode == 0, result.stderr
    assert "<trace-id>" in result.stdout


@pytest.mark.parametrize("content", ['{"hotLoad": {"otherKey": "x"}}', "{", "{}"], ids=["no pathPrefix", "truncated", "empty"])
def test_the_cleanup_block_stops_on_a_broken_deployment_file(tmp_path: Path, content: str) -> None:
    # a prefix left in the shell must not stand in for a file that exists but cannot be read
    result, calls = run_cleanup(tmp_path, deployment_json=content, shell={"HOTLOAD_PREFIX": "stale/prefix"})
    assert result.returncode != 0
    assert calls == ""


def test_the_cleanup_block_uses_the_run_env_prefix_without_a_file(tmp_path: Path) -> None:
    shell = {"HOTLOAD_PREFIX": "stale/prefix"}
    result, calls = run_cleanup(tmp_path, hotload_prefix_file=False,
                                HOTLOAD_PREFIX="made/elsewhere", shell=shell)
    assert result.returncode == 0, result.stderr
    assert "s3://bkt/made/elsewhere/" in calls and "stale/prefix" not in calls


@pytest.mark.parametrize("file", [True, False], ids=["deployment file", "run.env"])
def test_the_cleanup_block_ignores_slashes_around_the_prefix(tmp_path: Path, file) -> None:
    # the API stores snapshots under "hotload/" for a pathPrefix of "/hotload/"
    if file:
        result, calls = run_cleanup(tmp_path, deployment_json='{"hotLoad": {"pathPrefix": "/hotload/"}}')
    else:
        result, calls = run_cleanup(tmp_path, hotload_prefix_file=False, HOTLOAD_PREFIX="/hotload/")
    assert result.returncode == 0, result.stderr
    removed = [line.split()[4] for line in calls.splitlines() if line.startswith("aws s3 rm")]
    assert removed[0] == "s3://bkt/hotload/"


@pytest.mark.parametrize("file", [True, False], ids=["deployment file", "run.env"])
def test_the_cleanup_block_refuses_a_bare_slash_prefix(tmp_path: Path, file) -> None:
    # "/" is the top of the bucket; the block must not delete it all
    if file:
        result, calls = run_cleanup(tmp_path, deployment_json='{"hotLoad": {"pathPrefix": "/"}}')
    else:
        result, calls = run_cleanup(tmp_path, hotload_prefix_file=False, HOTLOAD_PREFIX="/")
    assert result.returncode != 0
    assert calls == ""


def test_the_cleanup_block_prefers_the_deployment_file_to_run_env(tmp_path: Path) -> None:
    result, calls = run_cleanup(tmp_path, HOTLOAD_PREFIX="old/prefix")
    assert result.returncode == 0, result.stderr
    assert "s3://bkt/hotload/" in calls and "old/prefix" not in calls


def test_run_env_grants_the_trainer_the_example_deployments_hot_load_prefix() -> None:
    # make-bucket-job.yaml lets the trainer write only under HOTLOAD_PREFIX, so it must be the
    # pathPrefix of the deployment that the README creates
    example = re.search(r"cp \.\./(deployment[\w.]*\.example\.json) deployment\.run\.json",
                        (RECIPE / "README.md").read_text()).group(1)  # fmt: skip
    path_prefix = json.loads((RECIPE / example).read_text())["hotLoad"]["pathPrefix"]
    result = subprocess.run(
        ["bash", "-c", 'set -euo pipefail; set -a; . "$1"; printf "%s" "$HOTLOAD_PREFIX"', "-",
         str(K8S / "run.env.example")],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert result.stdout == path_prefix


def test_the_cleanup_block_explains_a_deployment_without_a_prefix(tmp_path: Path) -> None:
    result, calls = run_cleanup(tmp_path, deployment_json='{"hotLoad": {"bucket": "bkt"}}')
    assert result.returncode != 0 and "no hotLoad.pathPrefix" in result.stderr
    assert "Traceback" not in result.stderr and calls == ""


def test_the_setup_block_keeps_a_filled_in_run_env(tmp_path: Path) -> None:
    k8s = tmp_path / "recipes" / "training" / RECIPE.name / "k8s"
    k8s.mkdir(parents=True)
    shutil.copy(K8S / "run.env.example", k8s / "run.env.example")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "kubectl").write_text("#!/bin/sh\necho ctx\n")
    (bin_dir / "kubectl").chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin"}

    def run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(["bash", "-c", readme_block("## Running it on Kubernetes")], cwd=tmp_path, env=env,
                              capture_output=True, text=True, check=False)  # fmt: skip

    assert run().returncode == 0
    (k8s / "run.env").write_text((k8s / "run.env").read_text() + "DEPLOYMENT_ID=mine\n")
    assert run().returncode == 0
    assert (k8s / "run.env").read_text().endswith("DEPLOYMENT_ID=mine\n")


def test_the_cleanup_block_accepts_a_lone_angle_bracket(tmp_path: Path) -> None:
    result, calls = run_cleanup(tmp_path, MODEL_SLUG="'run<v2'")
    assert result.returncode == 0, result.stderr
    assert "s3://bkt/models/run<v2/" in calls


@needs_envsubst
def test_render_refuses_an_uppercase_placeholder(tmp_path: Path) -> None:
    result = render(filled_run_env(tmp_path, ROLLOUT_MODEL="'<Served Model>'"), "trainer-job.yaml")
    assert result.returncode == 1 and "ROLLOUT_MODEL" in result.stderr
