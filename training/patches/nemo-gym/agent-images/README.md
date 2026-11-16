# Pending NeMo Gym patch: offline SWE agent images

This is a temporary downstream copy of [NVIDIA-NeMo/Gym draft PR #6252](https://github.com/NVIDIA-NeMo/Gym/pull/6252)
while upstream review proceeds. It adds opt-in image reconstruction and admission
for SWE-Gym coding sandboxes. No cookbook recipe installs or enables it automatically.
The shipped SWE-Gym OpenCode harness needs model egress and cannot use this offline
mode directly; a host-side driver using Gym's seed/connection contract is required.
Keep this experimental until live Docker and compatible-harness rollouts are validated.

The cookbook and upstream PR are public. This is a downstream patch, not confidential
storage. It contains no task datasets, hidden tests, reference solutions or credentials.

| Pin | Value |
| --- | --- |
| Source repository | `https://github.com/NVIDIA-NeMo/Gym.git` |
| Required base | `59d0b04f17cc5860e66bc51a35a7dcd8db0b1ebb` |
| Patched commit | `80bfa5057a6466620b783b461eac5d8537af34c7` |
| Patched file tree | `225d5d66d3a7bfcfcdd4f64468aed794ba012c3d` |
| Fork branch | `anson627/Gym:codex/clean-agent-images` |
| Upstream review | [Draft PR #6252](https://github.com/NVIDIA-NeMo/Gym/pull/6252) |
| Patch SHA-256 | `4d709643ab3303dd4870bd223912f159b2aff2f67c890c7ea8618b2bcbf27fe3` |
| Source license | Apache-2.0; NVIDIA notices retained; attribution recorded in `REUSE.toml` |

The patch is exported with `git format-patch -1 --stdout --no-signature`. It is
independent of cookbook PR #54 and verifier-boundary Gym PR #6251. This directory
carries only the agent-image/network change. At these pinned revisions, the verifier
patch from Gym #6251 does not apply unchanged on top of this patch because both touch
`app.py`. Combining them requires an explicit rebase/merge and fresh integration tests;
this carrier does not claim a validated combined stack.

## Apply manually

Use a clean, separate Gym checkout pinned to the required base. Compatibility with
the web-search recipe's pinned Gym release has not been established. ASearcher
training and BrowseComp evaluation have server-side search/browse tools and no agent
shell; this patch must not be presented as an improvement to that search workflow.

Set `COOKBOOK_ROOT` to this cookbook checkout, then run in the separate Gym checkout:

```bash
test "$(git rev-parse HEAD)" = 59d0b04f17cc5860e66bc51a35a7dcd8db0b1ebb
test -z "$(git status --porcelain)"
PATCH_FILE="$COOKBOOK_ROOT/training/patches/nemo-gym/agent-images/0001-swe-gym-clean-agent-images.patch"
git apply --check "$PATCH_FILE"
git am "$PATCH_FILE"
test "$(git rev-parse 'HEAD^{tree}')" = 225d5d66d3a7bfcfcdd4f64468aed794ba012c3d
```

`git am` may produce a different commit SHA because the committer timestamp changes;
the file tree must match. Read the patched [image/admission guide](https://github.com/anson627/Gym/blob/80bfa5057a6466620b783b461eac5d8537af34c7/fern/versions/latest/pages/infrastructure/sandbox/clean-agent-images.mdx)
for building an approved runtime, supplying all trusted references, enabling
`agent_image_manifests`, and clearing inherited provider options. Admission requires
an immutable local Docker image ID and explicit offline controls; unsupported
providers or unsafe mounts/flags fail closed.

Run the focused checks from the patched checkout:

```bash
python -m pytest resources_servers/swe_gym/tests \
  tests/unit_tests/sandbox/test_docker_provider.py -q -o addopts=''
pre-commit run --all-files
```

## Scope and validation limits

The patch reconstructs the base files into fresh one-commit Git history and combines
them with a separately approved, task-free runtime in one flattened Docker layer.
It rejects known artifact/credential paths and reference fragments, scans bounded
common compression, normalizes timestamps, and records task/base/runtime/rootfs
metadata in a trusted-host manifest. Offline Docker admission blocks arbitrary
network destinations and checks direct IPv4/IPv6 plus actual/external UDP/TCP DNS
before agent handoff. Patch capture uses the reconstructed commit. Verifier images
and trusted test execution are outside this change.

At the pinned commit, the focused suite reported **229 passed**. A clean-base
`git apply --check` and `git am` reproduced the exact pinned file tree. Gym's
all-file pre-commit checks passed; the private guardword catalog was unavailable,
so that scan was skipped. Fern validation reported zero errors; its authenticated
redirect check was skipped.

No actual imported image, Docker dataplane or representative model rollout was
validated: no Docker CLI/daemon was available. Network probe tests exercise the
logic with controlled responses and do not certify a deployed network. No paid GPU
jobs were launched. Full locked-environment/data-validation and maintainer CI checks
remain pending in the upstream draft.

Runtime/dependency cutoff review and complete trusted reference inputs remain
operator responsibilities. Fragment scanning cannot detect every encoding,
credential format or semantic answer leak. Source symlinks/submodules and runtime
hardlinks/special files are rejected. Manifests are not cryptographically signed by
this code, and registry admission is not implemented. CoreWeave and other providers
are rejected; `deny_egress` alone is insufficient for CoreWeave DNS/platform isolation.
Docker remains a shared-kernel runtime. This is not production reward-generation
approval or completion of the full reward-hacking defense design.

## Update and remove

When the upstream draft changes, review the new diff, regenerate this patch, update
all pins and the checksum, then repeat clean-base application, exact-tree comparison
and the focused checks. Keep this artifact limited to its upstream change.

Remove this directory when a compatible pinned Gym release includes the accepted
implementation, in the same cookbook change that adopts that release. If upstream
accepts a different design, reconcile this patch before using it. Do not silently
apply it to a moving release or enable it for unrelated recipes.
