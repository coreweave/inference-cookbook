# Contributing to the CoreWeave Inference Cookbook

Issues and pull requests are welcome: new recipes, fixes to existing ones, and improvements to
`cwtrain`. This page covers the contributor agreement, how to run the checks locally, the
conventions reviewers look for, and how `cwtrain` is released.

## Contributor License Agreement

Contributors must agree to the [CoreWeave CLA](./CLA.md) when pushing code to this project.

Agreement with the CoreWeave CLA must be signified by including a `Signed-off-by` trailer in every
submitted Git commit to this repository. By signing off, you certify that you have the right to
submit the contribution and that you agree to and are bound by the CoreWeave Contributor License
Agreement in effect at the date of your submission, found in [CLA.md](./CLA.md) in this
repository, which governs your submission. If you are contributing on behalf of an entity, you
further certify that you are authorized to bind that entity to the CLA.

To sign off one commit, use the `--signoff` (`-s`) option of
[`git commit`](https://git-scm.com/docs/git-commit#Documentation/git-commit.txt---signoff). To sign
off every commit in your clone, add a `prepare-commit-msg` hook:

```bash
cat > .git/hooks/prepare-commit-msg <<'EOF'
#!/bin/sh
git interpret-trailers --in-place --if-exists doNothing \
  --trailer "Signed-off-by: $(git config user.name) <$(git config user.email)>" "$1"
EOF
chmod +x .git/hooks/prepare-commit-msg
```

To sign off the commits already on your branch, run `git rebase --signoff main`.

## Development

### `libs/cwtrain`

```bash
cd libs/cwtrain
uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e '.[dev]' --group nemo-rl-tests
ruff check .
ruff format --check .
pytest
uv build   # builds the wheel and source distribution
```

The tests use an in-memory fake of the API. They need no credentials and no GPUs. The NeMo-RL
integration tests need torch (a CPU build is enough) and safetensors, from the `nemo-rl-tests`
dependency group. Without torch, pytest skips them. CI (`.github/workflows/cwtrain.yaml`) installs
both and runs the same checks on Python 3.12 to 3.14 for every pull request that touches
`libs/cwtrain`. It also builds a wheel and source distribution to catch packaging errors.

### Recipes

From the repository root, with the environment above active:

```bash
pytest recipes/*/*/tests
python .github/scripts/check_shared_recipe_files.py
for f in recipes/*/*/k8s/*.sh; do bash -n "$f"; done
```

Recipes are grouped by use case, then named after the framework they integrate. For example,
`recipes/training/slime/` is RL training with slime. Each training recipe is self-contained, so you
can copy it and use it on its own. As a result, some setup files exist in each recipe and must stay
byte-identical: the example bucket policy and deployment, the shared `k8s/` manifests and scripts,
and the flow diagram. When you change one, copy it to every other training recipe.
`check_shared_recipe_files.py` lists these files and fails when the copies differ.

Put a new training recipe under `recipes/training/<framework>/`, with its own README and the
shared files. Add a row for it to the table in the top-level [README](README.md).

### Live hot-load test

The unit tests never call the CoreWeave API. An opt-in workflow does. To start it, a user with
write access comments `/hotload-e2e` on a pull request from a branch of this repository. The
workflow creates a gateway and a deployment, hot-loads a checkpoint, verifies every replica, and
deletes what it created. [libs/cwtrain/e2e/README.md](libs/cwtrain/e2e/README.md) describes its
configuration, trigger rules, cleanup, and how to run it from a workstation.

## Pull requests

- Keep one logical change per pull request. Include tests with code changes. CI runs the checks
  above on pull requests that touch `libs/cwtrain` or `recipes/`.
- Commit messages and pull request titles follow
  [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix(launch):`, `ci:`,
  `docs:`, `chore:`.
- Never commit deployment IDs, endpoints, tokens, or keys. For this reason, git ignores filled-in
  copies of the recipe templates (`k8s/run.env`, `k8s/*.run.json`).

## Releasing `cwtrain`

To start the release pipeline, push a stable semantic version tag in the form
`vMAJOR.MINOR.PATCH` (for example, `v0.1.0`). The version must exactly match `project.version` in
`libs/cwtrain/pyproject.toml`. The workflow does not accept leading zeros, prerelease suffixes, or
build metadata.

The workflow runs lint, tests, and builds on every Python version in the CI matrix. When they pass
and the `pypi` environment's reviewers approve, it publishes the Python 3.12 wheel and source
distribution to public PyPI. It then creates a GitHub release with the same files. The release
notes are a link to `libs/cwtrain` at the tag. Pull requests and branch pushes run the checks but
do not publish or create releases.

One-time setup for a repository and PyPI maintainer:

1. Create the GitHub environment `pypi` in `coreweave/inference-cookbook` with required
   reviewers and a deployment rule that allows only `v*` tags. The reviewers are the only approval
   between a pushed tag and the upload. Without them, the next tag publishes immediately.
2. An owner of the [`cwtrain` project on PyPI](https://pypi.org/project/cwtrain/) adds a
   [Trusted Publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/) with owner
   `coreweave`, repository `inference-cookbook`, workflow filename `cwtrain.yaml`, and environment
   `pypi`. You do not need a PyPI API token or a GitHub secret.
3. Make sure the repository and organization policy lets the release job's `GITHUB_TOKEN` use
   `contents: write` to create GitHub releases.

For each release:

1. Set `project.version` in `libs/cwtrain/pyproject.toml` to the release version, and merge that
   commit into `main`.
2. Tag that commit and push the tag (replace `0.1.0` with the new version):

   ```bash
   git tag -a v0.1.0 <release-commit> -m "Release v0.1.0"
   git push origin v0.1.0
   ```

3. Approve the `pypi` environment deployment, if it requires review.
4. Verify the GitHub release and the package on PyPI. You do not create the GitHub release by hand.

The build and test jobs have read-only repository access. Only the release job can create GitHub
releases. Only the publish job has OIDC permission for PyPI Trusted Publishing. See the
[PyPA publishing guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
for additional setup details.

If publishing fails, fix the publishing configuration and rerun only the failed job. The rerun
uploads the distributions that this workflow run already built. PyPI accepts a byte-identical
re-upload, so files that were already published do not block the retry. Do not rebuild (for
example, with "Re-run all jobs" after the build tools change): a rebuild can produce different
bytes, and PyPI never lets you replace a published file. Changed artifacts need a new version.

If the GitHub release step fails after the upload, rerun that job. It uploads the files again and
completes the existing release.

## License headers
<!--- REUSE-IgnoreStart -->

The cookbook uses the [Apache-2.0 license](LICENSE.md) and follows the
[REUSE specification](https://reuse.software/spec-3.3/). Source files have an SPDX-style header
with:
- Year the file was created & copyright owner: `SPDX-FileCopyrightText: 2026 CoreWeave, Inc.`
- SPDX license identifier: `SPDX-License-Identifier: Apache-2.0`
- Package name: `SPDX-PackageName: inference-cookbook`

Every Python, shell and YAML file starts with that header. Files that can't carry a comment, such
as JSON, SVG and patches, are covered by [REUSE.toml](REUSE.toml) instead.

Install the [REUSE tool](https://reuse.software/dev/#tool) with
`uv tool install 'reuse[charset-normalizer]'`, then add the header to a new file with:

```shell
reuse annotate --license Apache-2.0 --copyright 'CoreWeave, Inc.' --year "$(date +%Y)" --template default_template --skip-existing $FILE
```

Review each file before you add a header, so that you do not assign the wrong copyright owner. If
a file contains code you did not write, such as code copied from an upstream project, keep its
original copyright and license.

To validate the licensing state and generate an SPDX bill of materials, run:

```shell
reuse lint
reuse spdx
```

<!--- REUSE-IgnoreEnd -->
