# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""The Kubernetes steps an external user follows, for every training recipe: run.env renders every
manifest, nothing is left unset or half filled in, the Job cannot start without its Secrets, and the
kit script explains itself.

The recipes share the files in k8s/ and keep their own run.env.example, trainer-job.yaml,
run-trainer.sh and make-kit.sh. A test that reads a recipe's files or README runs once per recipe.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

SHARED = Path(__file__).resolve().parents[1]
ROOT = SHARED.parent
PLACEHOLDER = re.compile(r"<[A-Za-z][A-Za-z0-9 _-]*>")
needs_envsubst = pytest.mark.skipif(
    shutil.which("envsubst") is None, reason="render.sh needs envsubst"
)


@dataclass(frozen=True)
class Recipe:
    name: str  # its directory under training/recipes
    extra: str  # the run.env variable for free-form trainer arguments
    run_id: str  # how RUN_ID in run.env.example starts
    deployment: str  # the example deployment that the README creates
    model_prefix: str  # MODEL_PREFIX and MODEL_SLUG in a filled-in run.env.example
    model_slug: str
    # a variable that run.env.example leaves unfilled and trainer-job.yaml uses
    placeholder: str = "ROLLOUT_BASE_URL"
    inference_key: bool = True  # whether the trainer reads an inference API key Secret
    diagram: str = "docs/hotload-flow.drawio.svg"  # the hot-load flow that the README embeds

    @property
    def path(self) -> Path:
        return ROOT / "training" / "recipes" / self.name

    @property
    def k8s(self) -> Path:
        return self.path / "k8s"


RECIPES = [
    Recipe(
        name="miles",
        extra="EXTRA_TRAIN_ARGS",
        run_id="qwen3-4b-grpo",
        deployment="deployment.example.json",
        model_prefix="models",
        model_slug="Qwen3-4B",
    ),
    Recipe(
        name="slime",
        extra="EXTRA_TRAIN_ARGS",
        run_id="qwen3-4b-grpo",
        deployment="deployment.example.json",
        model_prefix="models",
        model_slug="Qwen3-4B",
    ),
    Recipe(
        name="nemo-rl",
        extra="EXTRA_OVERRIDES",
        run_id="nemotron-lightning-grpo",
        deployment="deployment.nemotron.example.json",
        model_prefix="cookbook/filled/models",  # cookbook/${DEPLOYMENT_RUN}/models
        model_slug="NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    ),
    Recipe(
        name="axolotl-sft",
        extra="EXTRA_TRAIN_ARGS",
        run_id="qwen3-4b-sft",
        deployment="deployment.example.json",
        model_prefix="models",
        model_slug="Qwen3-4B",
        placeholder="DEPLOYMENT_ID",  # SFT samples no rollouts, so it has no rollout endpoint
        inference_key=False,
        diagram="docs/hotload-flow-sft.drawio.svg",  # one publish after training, no RL loop
    ),
]
per_recipe = pytest.mark.parametrize("recipe", RECIPES, ids=[recipe.name for recipe in RECIPES])


def manifests(recipe: Recipe) -> list[Path]:
    """The recipe's own manifests and the shared ones."""
    return sorted([*SHARED.glob("*.yaml"), *recipe.k8s.glob("*.yaml")], key=lambda path: path.name)


def filled_run_env(recipe: Recipe, tmp_path: Path, **overrides: str) -> Path:
    """run.env.example with every <placeholder> filled in, as a user's run.env would be."""
    text = PLACEHOLDER.sub("filled", (recipe.k8s / "run.env.example").read_text())
    path = tmp_path / "run.env"
    path.write_text(text + "".join(f"{key}={value}\n" for key, value in overrides.items()))
    return path


def render(env_file: Path, manifest: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SHARED / "render.sh"), str(env_file), str(manifest)],
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=False,
    )


def git(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    """git with only the repository's own ignore rules, not a global one. Outside a git checkout,
    such as a ZIP download, the test skips."""
    if not (ROOT / ".git").exists():
        pytest.skip("needs a git checkout")
    result = subprocess.run(
        ["git", "-c", f"core.excludesFile={os.devnull}", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr  # check-ignore exits 1 when it ignores none
    return result


@per_recipe
def test_run_env_example_can_be_sourced_and_names_a_new_run_each_time(recipe: Recipe) -> None:
    # quoted placeholders are values to bash, not redirections
    result = subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail; set -a; . "$1"; printf "%s" "$RUN_ID"',
            "-",
            str(recipe.k8s / "run.env.example"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert re.fullmatch(rf"{re.escape(recipe.run_id)}-\d{{8}}T\d{{6}}Z-\d+", result.stdout)


@needs_envsubst
@pytest.mark.parametrize(
    ("recipe", "manifest"),
    [(recipe, manifest) for recipe in RECIPES for manifest in manifests(recipe)],
    ids=[f"{recipe.name}-{manifest.name}" for recipe in RECIPES for manifest in manifests(recipe)],
)
def test_every_manifest_renders_from_a_filled_in_run_env(
    tmp_path: Path, recipe: Recipe, manifest: Path
) -> None:
    result = render(filled_run_env(recipe, tmp_path), manifest)
    assert result.returncode == 0, result.stderr
    assert "${" not in result.stdout


@needs_envsubst
@per_recipe
def test_render_refuses_a_placeholder_left_in_place(tmp_path: Path, recipe: Recipe) -> None:
    env = tmp_path / "run.env"
    shutil.copy(recipe.k8s / "run.env.example", env)
    result = render(env, recipe.k8s / "trainer-job.yaml")
    assert result.returncode == 1 and result.stdout == ""
    assert "fill in" in result.stderr and recipe.placeholder in result.stderr
    # step 1 needs only NAMESPACE, so it works before the rest is filled in
    assert render(env, SHARED / "namespace.yaml").returncode == 0


@needs_envsubst
@per_recipe
def test_render_refuses_a_variable_the_env_file_does_not_set(
    tmp_path: Path, recipe: Recipe
) -> None:
    env = tmp_path / "run.env"
    env.write_text("NAMESPACE=ns\n")
    result = render(env, recipe.k8s / "trainer-job.yaml")
    assert result.returncode == 1 and result.stdout == ""
    assert "does not set" in result.stderr and "DEPLOYMENT_ID" in result.stderr


@needs_envsubst
@per_recipe
def test_free_form_arguments_may_carry_tags(tmp_path: Path, recipe: Recipe) -> None:
    # "<qwen size>" reads as a run.env placeholder anywhere but in EXTRA_* arguments
    value = "'--tag <qwen size> --stop </answer>'"
    env = filled_run_env(recipe, tmp_path, **{recipe.extra: value})
    result = render(env, recipe.k8s / "trainer-job.yaml")
    assert result.returncode == 0, result.stderr
    assert "--tag <qwen size> --stop </answer>" in result.stdout


@needs_envsubst
@per_recipe
def test_the_hot_load_settings_come_from_run_env(tmp_path: Path, recipe: Recipe) -> None:
    settings = {"WAIT": "0", "TIMEOUT": "3600", "VERIFY": "1", "PROMPT_CACHE_POLICY": "preserve"}
    overrides = {f"HOTLOAD_{key}": value for key, value in settings.items()}
    env = filled_run_env(recipe, tmp_path, **overrides)
    rendered = render(env, recipe.k8s / "trainer-job.yaml").stdout
    for key, value in settings.items():
        assert f'{{name: CWTRAIN_HOTLOAD_{key}, value: "{value}"}}' in rendered


@per_recipe
def test_the_trainer_does_not_start_without_the_api_token(recipe: Recipe) -> None:
    job = (recipe.k8s / "trainer-job.yaml").read_text()
    token = job[job.index("- name: COREWEAVE_API_TOKEN") :].splitlines()[1]
    assert "secretKeyRef" in token and "optional" not in token


@per_recipe
def test_the_example_deployment_sets_no_engine_env_or_internal_options(recipe: Recipe) -> None:
    runtime = json.loads((recipe.path / recipe.deployment).read_text())["runtime"]
    assert "engineEnv" not in runtime
    assert not {"enable-rl", "load-format"} & set(runtime["engineConfig"])


def kit_tree(recipe: Recipe, tmp_path: Path) -> tuple[Path, Path, Path]:
    """A copy of make-kit.sh and the shared script it sources in a repository layout, an empty
    wheel dir, and a recording kubectl."""
    script = tmp_path / "training" / "recipes" / recipe.name / "k8s" / "make-kit.sh"
    script.parent.mkdir(parents=True)
    shutil.copy(recipe.k8s / "make-kit.sh", script)
    (tmp_path / "k8s").mkdir()
    shutil.copy(SHARED / "find-wheel.sh", tmp_path / "k8s" / "find-wheel.sh")
    dist = tmp_path / "libs" / "cwtrain" / "dist"
    dist.mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    kubectl = bin_dir / "kubectl"
    kubectl.write_text(
        f'#!/bin/sh\nprintf "%s\\n" "$*" >> {calls}\ncase "$*" in *"-f -"*) cat >/dev/null;; esac\n'
    )
    kubectl.chmod(0o755)
    return script, dist, calls


def run_kit(script: Path, tmp_path: Path, **extra_env: str) -> subprocess.CompletedProcess[str]:
    env = {"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin", **extra_env}
    return subprocess.run(
        ["bash", str(script), "ctx", "ns", "kit"],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )


@per_recipe
def test_make_kit_says_how_to_build_the_missing_wheel(tmp_path: Path, recipe: Recipe) -> None:
    script, _, _ = kit_tree(recipe, tmp_path)
    result = run_kit(script, tmp_path)
    assert result.returncode == 1
    assert "uv build --wheel" in result.stderr


@per_recipe
def test_make_kit_bundles_the_newest_wheel(tmp_path: Path, recipe: Recipe) -> None:
    script, dist, calls = kit_tree(recipe, tmp_path)
    older, newer = dist / "cwtrain-0.10.0-py3-none-any.whl", dist / "cwtrain-0.2.0-py3-none-any.whl"
    for wheel, mtime in ((older, 1_700_000_000), (newer, 1_800_000_000)):
        wheel.write_bytes(b"")
        os.utime(wheel, (mtime, mtime))
    result = run_kit(script, tmp_path)
    assert result.returncode == 0, result.stderr
    assert f"--from-file={newer.name}=" in calls.read_text() and older.name not in calls.read_text()


@per_recipe
def test_make_kit_bundles_files_that_exist(tmp_path: Path, recipe: Recipe) -> None:
    # make-kit.sh finds the shared files by a path relative to itself; kubectl would fail on a
    # wrong one only against a cluster
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    args = tmp_path / "args"
    kubectl = bin_dir / "kubectl"
    kubectl.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" >> {args}\ncat >/dev/null\n')
    kubectl.chmod(0o755)
    wheel = tmp_path / "cwtrain-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"")
    result = run_kit(recipe.k8s / "make-kit.sh", tmp_path, CWTRAIN_WHEEL=str(wheel))
    assert result.returncode == 0, result.stderr
    prefix = "--from-file="
    lines = args.read_text().splitlines()
    bundled = dict(line[len(prefix) :].split("=", 1) for line in lines if line.startswith(prefix))
    needed = {wheel.name, "stage-model.sh", "run-trainer.sh", "bucket-policy.example.json"}
    assert needed <= bundled.keys()
    for key, path in bundled.items():
        assert Path(path).is_file(), f"make-kit.sh bundles {key} from {path}, which is not a file"


SECRET = "never-printed-4f2c"


def test_storage_secret_refuses_a_placeholder_region(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    kubectl = bin_dir / "kubectl"
    kubectl.write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> {calls}\ncat >/dev/null\n')
    kubectl.chmod(0o755)
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "AWS_ACCESS_KEY_ID": "id",
        "AWS_SECRET_ACCESS_KEY": SECRET,
    }
    script = [str(SHARED / "make-storage-secret.sh"), "ctx", "ns", "secret"]

    def run(region: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", *script],
            env={**env, "AWS_REGION": region},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
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


def readme_block(recipe: Recipe, heading: str, index: int = 0) -> str:
    """The index-th bash block under a heading of the recipe's README."""
    text = (recipe.path / "README.md").read_text()
    section = text[text.index(heading) :]
    start = -1
    for _ in range(index + 1):
        start = section.index("```bash\n", start + 1)
    start += len("```bash\n")
    return section[start : section.index("\n```", start)]


@per_recipe
def test_the_step_table_names_files_that_exist(recipe: Recipe) -> None:
    # the tests run the README's bash blocks, but not the commands in its step table: check that
    # each file they name exists, a $K/ one in the setup block's K, any other in the recipe's k8s/
    k = re.search(r"^K=(\S+)", readme_block(recipe, "## Running it on Kubernetes"), re.M)
    assert k and (recipe.k8s / k.group(1)).resolve() == SHARED
    text = (recipe.path / "README.md").read_text()
    table = text[text.index("| Step | Command |") :].split("\n\n")[0]
    named = re.findall(r"(\$K/)?\b([\w-]+\.(?:sh|yaml))\b", table)
    assert {name for shared, name in named if shared} >= {"render.sh", "namespace.yaml"}
    for shared, name in named:
        assert (SHARED if shared else recipe.k8s).joinpath(name).is_file(), f"{shared}{name}"


@per_recipe
def test_the_readme_links_shared_files_that_exist(recipe: Recipe) -> None:
    # the README links the files in k8s/ and embeds the diagram in docs/ by their paths from the
    # recipe; no other test reads k8s/README.md, the trainer access policy or the diagram
    text = (recipe.path / "README.md").read_text()
    links = set(re.findall(r"\]\((\.\./\.\./\.\./(?:k8s|docs)/[^)#\s]+)", text))
    expected = [
        "k8s/README.md",
        "k8s/trainer-access-policy.example.json",
        recipe.diagram,
    ]
    assert {f"../../../{path}" for path in expected} <= links
    for link in links:
        assert (recipe.path / link).is_file(), f"{link} is not a file"


# two runs of this deployment, as `cwtrain hotload list` prints them (newest first)
HOT_LOADS = ["run-b-v000000", "run-a-v000001", "run-a-v000000"]


def run_cleanup(
    recipe: Recipe,
    tmp_path: Path,
    *,
    cwtrain_exit: int = 0,
    hotload_prefix_file: bool = True,
    from_k8s: bool = False,
    git_repo: bool = True,
    shell: dict[str, str] | None = None,
    deployment_json: str | None = None,
    hot_loads: list[str] = HOT_LOADS,
    interpreter: str = "bash",
    **run_env: str,
) -> tuple[subprocess.CompletedProcess[str], str]:
    """Run the README's cleanup block in a repository with recording cwtrain, kubectl and aws.

    shell holds variables already exported in the user's shell, run_env the run.env overrides,
    hot_loads the identities that the deployment hot-loaded, and interpreter the shell that runs the
    block."""
    k8s = tmp_path / "training" / "recipes" / recipe.name / "k8s"
    k8s.mkdir(parents=True)
    if git_repo:
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    defaults = {
        "BUCKET": "bkt",
        "BUCKET_ZONE": "US-EAST-04A",
        "NAMESPACE": "ns",
        "DEPLOYMENT_ID": "dep-1",
        "JOB_NAME": "job",
        "STORAGE_SECRET": "store",
        "ADMIN_STORAGE_SECRET": "admin",
        "INFERENCE_KEY_SECRET": "ikey",
        "API_TOKEN_SECRET": "token",
        "KIT_NAME": "kit",
    }
    filled_run_env(recipe, k8s, **{**defaults, **run_env})  # writes k8s/run.env
    if hotload_prefix_file:
        good = json.dumps({"hotLoad": {"pathPrefix": "hotload"}})
        (k8s / "deployment.run.json").write_text(
            good if deployment_json is None else deployment_json
        )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    # a failing cwtrain says why on stderr, as it does for a refused delete or a missing token
    refuse = (
        f'echo "error: permission_denied: not allowed" >&2; exit {cwtrain_exit}'
        if cwtrain_exit
        else ":"
    )
    listed = tmp_path / "hot_loads.json"
    listed.write_text(json.dumps([{"spec": {"identity": identity}} for identity in hot_loads]))
    cwtrain = f'case "$1 $2" in "hotload list") cat {listed};; "deployment delete") {refuse};; esac'
    for tool, reply in (("cwtrain", cwtrain), ("kubectl", ":"), ("aws", ":")):
        (bin_dir / tool).write_text(f'#!/bin/sh\nprintf "{tool} %s\\n" "$*" >> {calls}\n{reply}\n')
        (bin_dir / tool).chmod(0o755)
    (bin_dir / "python3").symlink_to(sys.executable)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), **(shell or {})}
    result = subprocess.run(
        [interpreter, "-c", readme_block(recipe, "### Cleaning up")],
        cwd=k8s if from_k8s else tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, calls.read_text() if calls.exists() else ""


# users paste the block into their shell: bash, or zsh, the default on macOS
shells = pytest.mark.parametrize(
    "interpreter",
    [
        "bash",
        pytest.param(
            "zsh", marks=pytest.mark.skipif(shutil.which("zsh") is None, reason="needs zsh")
        ),
    ],
)


@shells
@pytest.mark.parametrize("from_k8s", [False, True], ids=["from the repository root", "from k8s"])
@per_recipe
def test_the_cleanup_block_deletes_the_deployment_recipe_objects_and_prefixes(
    tmp_path: Path, recipe: Recipe, from_k8s, interpreter
) -> None:
    result, calls = run_cleanup(recipe, tmp_path, from_k8s=from_k8s, interpreter=interpreter)
    assert result.returncode == 0, result.stderr
    lines = calls.splitlines()
    # the runs are listed while the deployment exists; a second cleanup still passes
    assert lines[:2] == [
        "cwtrain hotload list --deployment dep-1 --limit 100000",
        "cwtrain deployment delete dep-1 --ignore-not-found",
    ]
    delete = "kubectl -n ns delete"
    secrets = "store admin ikey token" if recipe.inference_key else "store admin token"
    assert [line for line in lines if line.startswith("kubectl")] == [
        f"{delete} job cwtrain-job cwtrain-stage-job cwtrain-make-bucket --ignore-not-found",
        f"{delete} secret {secrets} --ignore-not-found",
        f"{delete} configmap kit --ignore-not-found",
        f"{delete} serviceaccount cwtrain-trainer --ignore-not-found",
    ]  # only the recipe's objects; the namespace may hold other work
    removed = [line.split(" --endpoint-url")[0] for line in lines if line.startswith("aws s3 rm")]
    rm = "aws s3 rm --recursive s3://bkt"
    assert removed == [  # only this deployment's runs: every deployment shares the hot-load prefix
        f"{rm}/hotload/ --exclude * --include run-a-v* --include _manifests/run-a-v*",
        f"{rm}/hotload/ --exclude * --include run-b-v* --include _manifests/run-b-v*",
        f"{rm}/{recipe.model_prefix}/{recipe.model_slug}/",
        f"{rm}/{recipe.model_prefix}/_probe/",
    ]


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ({"BUCKET_ZONE": "'<availability zone>'"}, "fill in run.env"),
        ({"hotload_prefix_file": False, "HOTLOAD_PREFIX": ""}, "set HOTLOAD_PREFIX"),
        ({"BUCKET": "'<bucket>'"}, "fill in run.env"),
        ({"DEPLOYMENT_ID": "'<deployment id>'"}, "fill in run.env"),
    ],
    ids=["zone placeholder", "no hot-load prefix", "bucket placeholder", "deployment placeholder"],
)
@per_recipe
def test_the_cleanup_block_touches_nothing_until_everything_is_set(
    tmp_path: Path, recipe: Recipe, case, error
) -> None:
    result, calls = run_cleanup(recipe, tmp_path, **case)
    assert result.returncode != 0 and error in result.stderr, result.stderr
    assert calls == ""


@per_recipe
def test_the_cleanup_block_is_valid_bash(recipe: Recipe) -> None:
    block = readme_block(recipe, "### Cleaning up")
    assert subprocess.run(["bash", "-n", "-c", block], check=False).returncode == 0


@per_recipe
def test_the_cleanup_block_stops_when_the_deployment_is_not_deleted(
    tmp_path: Path, recipe: Recipe
) -> None:
    result, calls = run_cleanup(recipe, tmp_path, cwtrain_exit=1)
    assert result.returncode != 0 and "permission_denied" in result.stderr
    assert calls.splitlines() == [  # nothing else
        "cwtrain hotload list --deployment dep-1 --limit 100000",
        "cwtrain deployment delete dep-1 --ignore-not-found",
    ]


@per_recipe
def test_the_cleanup_block_leaves_the_shared_prefix_alone_without_hot_loads(
    tmp_path: Path, recipe: Recipe
) -> None:
    result, calls = run_cleanup(recipe, tmp_path, hot_loads=[])
    assert result.returncode == 0, result.stderr
    assert "found no hot-loads of dep-1" in result.stderr
    assert "s3://bkt/hotload/" not in calls  # never the whole prefix that other deployments share


@pytest.mark.parametrize("prefix", ["hotload//", "//hotload", "/hotload/"])
@per_recipe
def test_the_cleanup_block_strips_every_outer_slash(
    tmp_path: Path, recipe: Recipe, prefix: str
) -> None:
    # the bucket Job and the API strip them all, so cleanup must target the same objects
    result, calls = run_cleanup(recipe, tmp_path, hotload_prefix_file=False, HOTLOAD_PREFIX=prefix)
    assert result.returncode == 0, result.stderr
    removed = [line.split()[4] for line in calls.splitlines() if line.startswith("aws s3 rm")]
    assert removed[0] == "s3://bkt/hotload/"


@pytest.mark.parametrize("prefix", ["/", "///"])
@per_recipe
def test_the_cleanup_block_refuses_a_prefix_of_slashes(
    tmp_path: Path, recipe: Recipe, prefix: str
) -> None:
    result, calls = run_cleanup(recipe, tmp_path, hotload_prefix_file=False, HOTLOAD_PREFIX=prefix)
    assert result.returncode != 0 and "HOTLOAD_PREFIX" in result.stderr
    assert calls == ""


@per_recipe
def test_run_trainer_checks_snapshot_access_before_training(recipe: Recipe) -> None:
    script = (recipe.k8s / "run-trainer.sh").read_text()
    check = re.search(r"^(?!\s*#).*-m cwtrain\.integrations\.preflight\s*$", script, re.M)
    launch = re.search(r"^\s*exec\b", script, re.M)
    assert check and launch and check.start() < launch.start(), (
        "the access check must run before training"
    )


@needs_envsubst
@per_recipe
def test_the_trainer_job_can_run_as_the_trainer_service_account(
    tmp_path: Path, recipe: Recipe
) -> None:
    env = filled_run_env(recipe, tmp_path, TRAINER_SERVICE_ACCOUNT="trainer-sa", ORG_ID="org1")
    result = render(env, recipe.k8s / "trainer-job.yaml")
    assert result.returncode == 0, result.stderr
    assert 'serviceAccountName: "trainer-sa"' in result.stdout
    assert "https://api.coreweave.com/v1/cwobject/temporary-credentials/oidc/org1" in result.stdout
    assert 'audience: "https://coreweave.com/iam"' in result.stdout


@per_recipe
def test_the_create_block_creates_the_deployment_and_waits(tmp_path: Path, recipe: Recipe) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    (bin_dir / "cwtrain").write_text(f'#!/bin/sh\nprintf "cwtrain %s\\n" "$*" >> {calls}\n')
    (bin_dir / "cwtrain").chmod(0o755)
    result = subprocess.run(
        ["bash", "-c", readme_block(recipe, "### Creating the deployment", 1)],
        cwd=tmp_path,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert calls.read_text() == "cwtrain deployment create --file deployment.run.json --wait\n"


@pytest.mark.parametrize(("value", "seen"), [("", "unset"), (" ", "unset"), ("900", "900")])
@per_recipe
def test_run_trainer_leaves_an_empty_hot_load_timeout_to_cwtrain(
    recipe: Recipe, value: str, seen: str
) -> None:
    script = (recipe.k8s / "run-trainer.sh").read_text()
    guard = re.search(r"^hotload_timeout=.*\n.*unset CWTRAIN_HOTLOAD_TIMEOUT$", script, re.M)
    assert guard, "run-trainer.sh must drop a blank CWTRAIN_HOTLOAD_TIMEOUT"
    out = subprocess.run(
        ["bash", "-c", guard.group(0) + '\nprintf %s "${CWTRAIN_HOTLOAD_TIMEOUT-unset}"'],
        env={"PATH": "/usr/bin:/bin", "CWTRAIN_HOTLOAD_TIMEOUT": value},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert out == seen
    # the first non-comment launch line
    launch = re.search(r"^(?!\s*#).*(?:\bexec\b|run_hotload\.py)", script, re.M)
    assert launch and guard.start() < launch.start(), "the guard must run before the launcher"


@per_recipe
def test_the_bucket_policy_command_applies_the_git_ignored_copy(recipe: Recipe) -> None:
    text = (recipe.path / "README.md").read_text()
    assert "file://bucket-policy.json" not in text
    assert "file://k8s/bucket-policy.run.json" in text
    # git check-ignore exits 0 when git ignores the path
    copy = "k8s/bucket-policy.run.json"
    ignored = git("check-ignore", "-q", "--no-index", copy, cwd=recipe.path)
    assert ignored.returncode == 0, f"git does not ignore {copy}"


@pytest.mark.parametrize(
    "directory",
    [*(recipe.k8s for recipe in RECIPES), SHARED],
    ids=[*(recipe.name for recipe in RECIPES), "k8s"],
)
def test_the_filled_in_copies_are_git_ignored(directory: Path) -> None:
    # they hold ids, endpoints and headers: run.env, and the *.run.json copies of the templates,
    # in a recipe's k8s/ or next to the templates in k8s/
    copies = [
        "run.env",
        "deployment.run.json",
        "bucket-policy.run.json",
        "trainer-access-policy.run.json",
    ]
    ignored = git("check-ignore", "--no-index", *copies, cwd=directory).stdout.splitlines()
    assert ignored == copies


@per_recipe
def test_make_kit_skips_an_empty_patches_directory(tmp_path: Path, recipe: Recipe) -> None:
    if not (recipe.path / "patches").is_dir():
        pytest.skip("this recipe ships no patches")
    script, dist, calls = kit_tree(recipe, tmp_path)
    (dist / "cwtrain-0.1.0-py3-none-any.whl").write_bytes(b"")
    patches = script.parent.parent / "patches"
    patches.mkdir()
    result = run_kit(script, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "*.patch" not in calls.read_text()
    (patches / "0001-example.patch").write_text("diff\n")
    assert run_kit(script, tmp_path).returncode == 0
    assert "--from-file=0001-example.patch=" in calls.read_text()


@per_recipe
def test_make_kit_names_a_cwtrain_wheel_that_does_not_exist(tmp_path: Path, recipe: Recipe) -> None:
    script, dist, _ = kit_tree(recipe, tmp_path)
    (dist / "cwtrain-0.1.0-py3-none-any.whl").write_bytes(b"")
    result = run_kit(script, tmp_path, CWTRAIN_WHEEL=str(tmp_path / "typo.whl"))
    assert result.returncode == 1
    assert "CWTRAIN_WHEEL" in result.stderr and "uv build" not in result.stderr


@per_recipe
def test_the_cleanup_block_stops_when_it_cannot_reach_the_recipe(
    tmp_path: Path, recipe: Recipe
) -> None:
    # outside a checkout the cd fails; values the shell still holds from an earlier run.env must not
    # be used instead
    shell = {
        "DEPLOYMENT_ID": "dep-1",
        "HOTLOAD_PREFIX": "stale/prefix",
        "NAMESPACE": "ns",
        "BUCKET": "bkt",
        "BUCKET_ZONE": "US-EAST-04A",
        "MODEL_PREFIX": "models",
        "MODEL_SLUG": "m",
        "JOB_NAME": "job",
        "KIT_NAME": "kit",
        "STORAGE_SECRET": "s",
        "INFERENCE_KEY_SECRET": "i",
        "API_TOKEN_SECRET": "t",
    }
    result, calls = run_cleanup(recipe, tmp_path, git_repo=False, shell=shell)
    assert result.returncode != 0
    assert calls == ""


@per_recipe
def test_the_cleanup_block_prefers_the_deployment_file_to_a_prefix_left_in_the_shell(
    tmp_path: Path, recipe: Recipe
) -> None:
    result, calls = run_cleanup(recipe, tmp_path, shell={"HOTLOAD_PREFIX": "old/prefix"})
    assert result.returncode == 0, result.stderr
    removed = [line.split()[4] for line in calls.splitlines() if line.startswith("aws s3 rm")]
    assert removed[0] == "s3://bkt/hotload/" and "old/prefix" not in calls


@per_recipe
def test_the_cleanup_block_refuses_an_empty_model_prefix(tmp_path: Path, recipe: Recipe) -> None:
    result, calls = run_cleanup(recipe, tmp_path, MODEL_PREFIX="")
    assert result.returncode != 0 and "MODEL_PREFIX" in result.stderr
    assert calls == ""


@per_recipe
def test_the_parameters_block_keeps_a_filled_in_deployment_file(
    tmp_path: Path, recipe: Recipe
) -> None:
    k8s = tmp_path / "recipe" / "k8s"
    k8s.mkdir(parents=True)
    example = recipe.path / recipe.deployment
    shutil.copy(example, k8s.parent / recipe.deployment)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "cwtrain").write_text("#!/bin/sh\nprintf '{}'\n")
    (bin_dir / "cwtrain").chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin"}

    def run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "-c", readme_block(recipe, "### Creating the deployment")],
            cwd=k8s,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    assert run().returncode == 0
    assert (k8s / "deployment.run.json").read_text() == example.read_text()
    (k8s / "deployment.run.json").write_text('{"name": "mine"}')
    run()
    assert (k8s / "deployment.run.json").read_text() == '{"name": "mine"}'


@needs_envsubst
@per_recipe
def test_api_headers_may_carry_tags(tmp_path: Path, recipe: Recipe) -> None:
    value = """'{"X-Trace":"<trace-id>"}'"""
    env = filled_run_env(recipe, tmp_path, CWTRAIN_API_HEADERS=value)
    result = render(env, recipe.k8s / "trainer-job.yaml")
    assert result.returncode == 0, result.stderr
    assert "<trace-id>" in result.stdout


@pytest.mark.parametrize(
    "content",
    ['{"hotLoad": {"otherKey": "x"}}', "{", "{}"],
    ids=["no pathPrefix", "truncated", "empty"],
)
@per_recipe
def test_the_cleanup_block_stops_on_a_broken_deployment_file(
    tmp_path: Path, recipe: Recipe, content: str
) -> None:
    # a prefix left in the shell must not stand in for a file that exists but cannot be read
    shell = {"HOTLOAD_PREFIX": "stale/prefix"}
    result, calls = run_cleanup(recipe, tmp_path, deployment_json=content, shell=shell)
    assert result.returncode != 0
    assert calls == ""


@per_recipe
def test_the_cleanup_block_uses_the_run_env_prefix_without_a_file(
    tmp_path: Path, recipe: Recipe
) -> None:
    shell = {"HOTLOAD_PREFIX": "stale/prefix"}
    result, calls = run_cleanup(
        recipe, tmp_path, hotload_prefix_file=False, HOTLOAD_PREFIX="made/elsewhere", shell=shell
    )
    assert result.returncode == 0, result.stderr
    assert "s3://bkt/made/elsewhere/" in calls and "stale/prefix" not in calls


@pytest.mark.parametrize("file", [True, False], ids=["deployment file", "run.env"])
@per_recipe
def test_the_cleanup_block_ignores_slashes_around_the_prefix(
    tmp_path: Path, recipe: Recipe, file
) -> None:
    # the API stores snapshots under "hotload/" for a pathPrefix of "/hotload/"
    if file:
        result, calls = run_cleanup(
            recipe, tmp_path, deployment_json='{"hotLoad": {"pathPrefix": "/hotload/"}}'
        )
    else:
        result, calls = run_cleanup(
            recipe, tmp_path, hotload_prefix_file=False, HOTLOAD_PREFIX="/hotload/"
        )
    assert result.returncode == 0, result.stderr
    removed = [line.split()[4] for line in calls.splitlines() if line.startswith("aws s3 rm")]
    assert removed[0] == "s3://bkt/hotload/"


@pytest.mark.parametrize("file", [True, False], ids=["deployment file", "run.env"])
@per_recipe
def test_the_cleanup_block_refuses_a_bare_slash_prefix(
    tmp_path: Path, recipe: Recipe, file
) -> None:
    # "/" is the top of the bucket; the block must not delete it all
    if file:
        result, calls = run_cleanup(
            recipe, tmp_path, deployment_json='{"hotLoad": {"pathPrefix": "/"}}'
        )
    else:
        result, calls = run_cleanup(recipe, tmp_path, hotload_prefix_file=False, HOTLOAD_PREFIX="/")
    assert result.returncode != 0
    assert calls == ""


@per_recipe
def test_the_cleanup_block_prefers_the_deployment_file_to_run_env(
    tmp_path: Path, recipe: Recipe
) -> None:
    result, calls = run_cleanup(recipe, tmp_path, HOTLOAD_PREFIX="old/prefix")
    assert result.returncode == 0, result.stderr
    assert "s3://bkt/hotload/" in calls and "old/prefix" not in calls


@per_recipe
def test_run_env_grants_the_trainer_the_example_deployments_hot_load_prefix(recipe: Recipe) -> None:
    # make-bucket-job.yaml lets the trainer write only under HOTLOAD_PREFIX, so it must be the
    # pathPrefix of the deployment that the README creates
    example = re.search(
        r"cp \.\./(deployment[\w.]*\.example\.json) deployment\.run\.json",
        (recipe.path / "README.md").read_text(),
    ).group(1)
    path_prefix = json.loads((recipe.path / example).read_text())["hotLoad"]["pathPrefix"]
    result = subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail; set -a; . "$1"; printf "%s" "$HOTLOAD_PREFIX"',
            "-",
            str(recipe.k8s / "run.env.example"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == path_prefix


@per_recipe
def test_the_cleanup_block_explains_a_deployment_without_a_prefix(
    tmp_path: Path, recipe: Recipe
) -> None:
    result, calls = run_cleanup(recipe, tmp_path, deployment_json='{"hotLoad": {"bucket": "bkt"}}')
    assert result.returncode != 0 and "no hotLoad.pathPrefix" in result.stderr
    assert "Traceback" not in result.stderr and calls == ""


@per_recipe
def test_the_setup_block_keeps_a_filled_in_run_env(tmp_path: Path, recipe: Recipe) -> None:
    k8s = tmp_path / "training" / "recipes" / recipe.name / "k8s"
    k8s.mkdir(parents=True)
    shutil.copy(recipe.k8s / "run.env.example", k8s / "run.env.example")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "kubectl").write_text("#!/bin/sh\necho ctx\n")
    (bin_dir / "kubectl").chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin"}

    def run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "-c", readme_block(recipe, "## Running it on Kubernetes")],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    assert run().returncode == 0
    (k8s / "run.env").write_text((k8s / "run.env").read_text() + "DEPLOYMENT_ID=mine\n")
    assert run().returncode == 0
    assert (k8s / "run.env").read_text().endswith("DEPLOYMENT_ID=mine\n")


@per_recipe
def test_the_cleanup_block_accepts_a_lone_angle_bracket(tmp_path: Path, recipe: Recipe) -> None:
    result, calls = run_cleanup(recipe, tmp_path, MODEL_SLUG="'run<v2'")
    assert result.returncode == 0, result.stderr
    assert f"s3://bkt/{recipe.model_prefix}/run<v2/" in calls


@needs_envsubst
@per_recipe
def test_render_refuses_an_uppercase_placeholder(tmp_path: Path, recipe: Recipe) -> None:
    env = filled_run_env(recipe, tmp_path, **{recipe.placeholder: "'<Served Model>'"})
    result = render(env, recipe.k8s / "trainer-job.yaml")
    assert result.returncode == 1 and recipe.placeholder in result.stderr


def test_no_other_directory_holds_a_copy_of_a_shared_file() -> None:
    # every recipe uses the files in k8s/ by their path, so a change reaches all of them. The shared
    # files are the tracked ones, so a .DS_Store that Finder leaves in k8s/ is not one of them.
    tracked = [Path(path) for path in git("ls-files", "k8s").stdout.splitlines()]
    shared = {path.name for path in tracked if path.parent == Path("k8s")} - {"README.md"}
    listed = git("ls-files", "--cached", "--others", "--exclude-standard").stdout.splitlines()
    copies = [path for path in listed if not path.startswith("k8s/") and Path(path).name in shared]
    assert copies == [], "use the files in k8s/ by path, not copies"


def test_every_training_recipe_is_in_recipes() -> None:
    # a recipe that is not in RECIPES gets none of the tests above
    found = {path.parent.name for path in (ROOT / "training" / "recipes").glob("*/k8s")}
    assert found == {recipe.name for recipe in RECIPES}, "add the recipe to RECIPES"
