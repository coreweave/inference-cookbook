# Pending NeMo Gym patch: SWE-Gym patch boundary

This is a temporary vendored patch while [NVIDIA-NeMo/Gym#6251](https://github.com/NVIDIA-NeMo/Gym/pull/6251)
is reviewed. It hardens SWE-Gym's patch transfer into its existing fresh verifier
sandbox. It is opt-in for coding environments; no recipe installs it automatically.
The ASearcher/BrowseComp search recipe does not execute agent workspace code and
must not consume this patch as a claimed search-workflow defense.

| Pin | Value |
| --- | --- |
| Repository | `https://github.com/NVIDIA-NeMo/Gym.git` |
| Required base | `59d0b04f17cc5860e66bc51a35a7dcd8db0b1ebb` |
| Patched commit | `2d354c0ce4b0a3be38ad3c2d3c1cd87cd676806e` |
| Fork branch | `anson627/Gym:codex/verifier-isolation` |
| Upstream review | [Draft PR #6251](https://github.com/NVIDIA-NeMo/Gym/pull/6251) |
| Source license | Apache-2.0; NVIDIA SPDX notices retained in the patch |

The patch was exported with `git format-patch -1 --stdout --no-signature` from the
patched commit. It is independent of the cookbook's answer/judge changes in PR #54
and the companion agent-image/network work.

## Apply manually in a separate Gym checkout

Use a clean checkout pinned to the required base. Do not apply this to an arbitrary
installed Gym release: the patch depends on the current upstream SWE session
lifecycle, and compatibility with the search recipe's Gym version is not established.

Set `COOKBOOK_ROOT` to this cookbook checkout, then run from your separate Gym
checkout:

```bash
test "$(git rev-parse HEAD)" = 59d0b04f17cc5860e66bc51a35a7dcd8db0b1ebb
test -z "$(git status --porcelain)"
git apply --check "$COOKBOOK_ROOT/training/patches/nemo-gym/0001-swe-gym-patch-boundary.patch"
git am "$COOKBOOK_ROOT/training/patches/nemo-gym/0001-swe-gym-patch-boundary.patch"
```

The resulting commit SHA can differ because `git am` records a new committer date;
the applied file tree must match the pinned patched commit. Run Gym's focused
regressions from the patched checkout:

```bash
pytest resources_servers/swe_gym/tests \
  resources_servers/swebench/tests/test_patch_capture.py \
  tests/unit_tests/verification/test_sandbox_verification.py
pre-commit run --all-files
```

## Guarantees and outstanding work

The patch caps selected diffs at 1 MiB, rejects malformed/unsafe paths and recognized
test/grader/environment edits, checks the trusted verifier base and symlinks, removes
fuzzy application, and separates scored policy failures from inconclusive masked
verification. Existing provider/session infrastructure owns isolation, retries,
resource enforcement and teardown. See the patched SWE-Gym README for the full
contract and compatibility changes.

**This does not make grading tamper-proof.** Candidate Python can still influence
pytest from inside the test process and forge passing stdout. The suite includes a
strict expected-failure test for that remaining boundary. Do not treat this patch
as completion of defense #2 or enable it for production reward generation on the
strength of unit tests alone. Trusted grading outside candidate-controlled
processes, real model/provider smoke rollouts, and golden-patch coverage validation
remain outstanding. Image hygiene and outbound access belong to separate controls.

At the pinned commit, the focused suite reported **213 passed, 1 expected failure**.
All-file pre-commit checks passed; the private guardword catalog was unavailable and
its hook skipped that scan. Focused mypy could not start because the locked Gym
development environment lacks `pathspec`. No paid GPU jobs were run.

## Update and retirement

When revising the upstream draft, regenerate this patch from the reviewed commit,
update both commit pins, and check application against a fresh exact-base checkout.
Keep unrelated upstream fixes out of this patch. A changed base requires rerunning
the focused suite and reviewing compatibility; do not rely on fuzzy patch offsets.

Once a compatible pinned Gym release includes the accepted changes, remove this
patch and its manual application instructions in the same cookbook change that
adopts that release. If upstream accepts a smaller or different solution, reconcile
this temporary patch with that implementation before using it.
