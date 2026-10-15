# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Policy-worker extensions that add ``export_hf_checkpoint``.

Register a class with ``policy.worker_extension_cls_fqn``. (The recipe's NeMo-RL patch defaults
to these classes.) NeMo-RL instantiates it instead of the stock worker. So each class derives from
the worker implementation the policy would otherwise use (DTensor v2 or Megatron), with the same
``ray.remote`` decoration.

``export_hf_checkpoint`` is a collective. Every rank iterates the HF-layout weights, with the same
generator the NCCL refit uses (it gathers TP/PP shards). Rank 0 writes the safetensors shards, the
index and the checkpoint's non-weight files (``config.json``, tokenizer) to ``path``.

Dtypes:

* The DTensor worker writes each tensor in the dtype that the deployment's checkpoint uses. It
  casts once from the policy's dtype, so tensors that the checkpoint keeps in FP32 stay in FP32.
  It casts only between F16, BF16, F32 and F64. A quantized checkpoint (FP8 or integer weights)
  also needs its scales, so the export stops on such a tensor unless the policy already holds
  that dtype.
* The Megatron worker writes the dtypes Megatron-Bridge exports (see :func:`_megatron_export`).

This module imports only torch and safetensors, so it loads in the worker's venv without the
other ``cwtrain`` dependencies.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
from pathlib import Path
from typing import Any

from cwtrain.integrations.nemo_rl import _missing_dependency

try:
    import torch
except ModuleNotFoundError as exc:
    raise _missing_dependency(exc) from exc

SHARD_BYTES = 4 * 2**30
_SKIP_SUFFIXES = (".safetensors", ".safetensors.index.json")
#: safetensors dtype names, as checkpoint headers spell them.
_TORCH_DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}
#: The dtypes the export casts between. Other dtypes are written only if the policy holds them.
_FLOAT_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32, torch.float64})


class _ShardWriter:
    """Accumulate CPU tensors and write them as ``model-XXXXX-of-YYYYY.safetensors`` shards."""

    def __init__(self, path: Path, shard_bytes: int) -> None:
        self.path = path
        self.tmp = path.with_name(path.name + ".tmp")
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.tmp.mkdir(parents=True)
        self.shard_bytes = shard_bytes
        self._pending: dict[str, torch.Tensor] = {}
        self._pending_bytes = 0
        self._shards: list[tuple[str, list[str]]] = []
        self.total_bytes = 0

    def add(self, name: str, tensor: torch.Tensor) -> None:
        if self._pending and self._pending_bytes + tensor.nbytes > self.shard_bytes:
            self._flush()
        self._pending[name] = tensor
        self._pending_bytes += tensor.nbytes
        self.total_bytes += tensor.nbytes

    def _flush(self) -> None:
        if not self._pending:
            return
        try:
            from safetensors.torch import save_file
        except ModuleNotFoundError as exc:
            raise _missing_dependency(exc) from exc

        fname = f"model-{len(self._shards):05d}.safetensors"
        save_file(self._pending, str(self.tmp / fname), metadata={"format": "pt"})
        self._shards.append((fname, list(self._pending)))
        self._pending, self._pending_bytes = {}, 0

    def finish(self, config_dir: str | None) -> dict[str, Any]:
        self._flush()
        total = len(self._shards)
        weight_map: dict[str, str] = {}
        for i, (fname, names) in enumerate(self._shards):
            final = f"model-{i + 1:05d}-of-{total:05d}.safetensors"
            os.replace(self.tmp / fname, self.tmp / final)
            weight_map.update({n: final for n in names})
        (self.tmp / "model.safetensors.index.json").write_text(
            json.dumps(
                {"metadata": {"total_size": self.total_bytes}, "weight_map": weight_map}, indent=1
            )
        )
        if config_dir:
            for entry in Path(config_dir).iterdir():
                if entry.is_file() and not entry.name.endswith(_SKIP_SUFFIXES):
                    shutil.copy2(entry, self.tmp / entry.name)
        shutil.rmtree(self.path, ignore_errors=True)
        self.tmp.replace(self.path)
        return {
            "path": str(self.path),
            "tensors": len(weight_map),
            "bytes": self.total_bytes,
            "shards": total,
        }


def _any_rank_failed(failed: bool) -> bool:
    """Whether any rank failed, agreed by every rank (a MAX all-reduce of a 0/1 flag)."""
    if not torch.distributed.is_initialized():
        return failed
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    flag = torch.tensor([1 if failed else 0], dtype=torch.int32, device=device)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    return bool(flag.item())


def _target_dtype(name: str, tensor: torch.Tensor, dtypes: dict[str, str] | None) -> torch.dtype:
    """The dtype to write ``name`` in: the checkpoint's, when the policy's tensor converts to it.

    A cast to FP8 or an integer type ignores a quantized checkpoint's scales. It writes wrong
    values that still match the checkpoint's layout. So the only casts are between the
    floating-point dtypes in ``_FLOAT_DTYPES``.
    """
    if not dtypes or name not in dtypes:
        return tensor.dtype
    target = _TORCH_DTYPES.get(dtypes[name])
    if target is None:
        raise ValueError(
            f"{name} is {dtypes[name]} in the deployment's checkpoint, a safetensors dtype the "
            "export does not know"
        )
    if target == tensor.dtype or {target, tensor.dtype} <= _FLOAT_DTYPES:
        return target
    raise ValueError(
        f"{name} is {dtypes[name]} in the deployment's checkpoint and {tensor.dtype} in the "
        "policy; the export casts only between F16, BF16, F32 and F64 (a quantized checkpoint "
        "needs its scales too), so it cannot write that dtype"
    )


def _export(
    params: Any,
    path: str,
    allowed_names: list[str] | None,
    config_dir: str | None,
    shard_bytes: int,
    dtypes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Iterate ``params`` on every rank; rank 0 writes the checkpoint.

    ``dtypes`` maps tensor names to their safetensors dtype in the deployment's checkpoint. Rank 0
    writes each of those tensors in that dtype. Other tensors keep the dtype ``params`` yields.

    The generator runs a collective per tensor (it gathers TP/PP shards). A rank that stops
    iterating early leaves its peers blocked in the next collective until the distributed
    timeout. So a write failure on rank 0 (for example, a full disk) stops the writing but not the
    iteration. After the last tensor, the ranks agree on whether any rank failed, and all raise
    together.

    Raises:
        RuntimeError: The export failed on this rank or on another rank.
    """
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    allowed = set(allowed_names) if allowed_names else None
    error: BaseException | None = None
    writer: _ShardWriter | None = None
    if rank == 0:
        try:
            writer = _ShardWriter(Path(path), shard_bytes)
        except Exception as exc:  # noqa: BLE001 - reported once every rank has finished
            error = exc
    seen: set[str] = set()
    for name, tensor in params:
        if allowed is not None and name not in allowed:
            continue
        seen.add(name)
        if writer is not None and error is None:
            try:
                target = _target_dtype(name, tensor, dtypes)
                writer.add(name, tensor.detach().to("cpu", target, copy=True).contiguous())
            except Exception as exc:  # noqa: BLE001 - keep iterating so peers do not block
                error = exc
    result: dict[str, Any] = {"rank": rank, "tensors": len(seen)}
    if rank == 0 and error is None:
        if allowed is not None and seen != allowed:
            missing = sorted(allowed - seen)[:5]
            error = RuntimeError(
                f"export is missing {len(allowed - seen)} tensors the deployment holds, "
                f"e.g. {missing}"
            )
        else:
            try:
                result = {**writer.finish(config_dir), "rank": rank}  # type: ignore[union-attr]
            except Exception as exc:  # noqa: BLE001 - reported below, on every rank
                error = exc
    if _any_rank_failed(error is not None):
        if writer is not None:
            shutil.rmtree(writer.tmp, ignore_errors=True)
        if error is not None:
            raise RuntimeError(f"checkpoint export failed on rank {rank}: {error}") from error
        raise RuntimeError(f"checkpoint export failed on another rank; rank {rank} stops too")
    return result


# Ray forbids subclassing a decorated actor class. So the extensions derive from NeMo-RL's
# undecorated *Impl classes and apply the same ``ray.remote`` runtime env as the stock workers.
# NeMo-RL's own quantized workers do the same.

try:  # DTensor v2 (Automodel) policy
    import ray
    from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
    from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import (
        DTensorPolicyWorkerV2Impl,
        dtensor_params_generator,
    )
except Exception:  # pragma: no cover - depends on the NeMo-RL install
    DTensorPolicyWorkerV2Impl = None  # type: ignore[assignment,misc]
    dtensor_params_generator = None  # type: ignore[assignment]


def _dtensor_hf_params(model: Any, policy_dtype: Any, dtypes: dict[str, str] | None) -> Any:
    """NeMo-RL's HF-layout weights, uncast when ``dtypes`` names the checkpoint's dtypes.

    ``dtensor_params_generator`` converts every tensor to the one dtype it is given. That turns a
    checkpoint's FP32 tensors (for example, MoE router biases) into bf16 and rounds their values.
    ``tensor.to(None)`` changes nothing. So with ``dtypes``, the tensors keep the policy's dtype,
    and :func:`_export` casts each tensor once, to the dtype of the deployment's checkpoint.

    This relies on how NeMo-RL applies the cast: ``tensor.to(target_dtype, non_blocking=True)`` in
    v0.7.0, the version the recipe pins. :func:`_check_keeps_tensors` checks it first.
    """
    if not dtypes:
        return dtensor_params_generator(model, policy_dtype)
    _check_keeps_tensors(dtensor_params_generator)
    return dtensor_params_generator(model, None)


@functools.cache
def _check_keeps_tensors(generator: Any) -> None:
    """Check that ``generator(module, None)`` yields a plain module's tensors as they are.

    The probe holds one tensor in each dtype the export casts between. Its values need more
    mantissa bits than the narrower dtypes have, or more range than F16 has. A generator that
    still casts some dtypes for ``None``, or rounds a tensor and casts it back (even in place),
    then stops the export. It does not round the checkpoint's FP32 tensors.

    This is a sanity check of the installed NeMo-RL, not a proof. A generator that treats the real
    model's tensors (DTensor shards, adapter output, GPU memory) differently from a plain CPU
    module's tensors passes. A passing check is cached per generator. A failing check raises again
    on the next export.

    Raises:
        RuntimeError: The generator changes a tensor's dtype or values.
    """
    values = torch.tensor(
        [1 + 2**-40, 1 + 2**-20, 1 + 2**-10, 1 + 2**-7, -1.0e5, 1.0e-6], dtype=torch.float64
    )
    probe = torch.nn.Module()
    for name, dtype in _TORCH_DTYPES.items():
        if dtype in _FLOAT_DTYPES:
            probe.register_parameter(
                name, torch.nn.Parameter(values.to(dtype), requires_grad=False)
            )
    # Copies, so a generator that changes the probe's own tensors cannot pass by yielding them.
    expected = {name: tensor.clone() for name, tensor in probe.state_dict().items()}
    yielded = dict(generator(probe, None))
    if set(yielded) != set(expected):
        raise RuntimeError(
            f"NeMo-RL's dtensor_params_generator returned {sorted(yielded)} for a plain module "
            f"holding {sorted(expected)}, so cwtrain cannot check that it keeps each tensor's "
            "dtype; check cwtrain against this NeMo-RL version"
        )
    changed = [
        f"{name} came back {yielded[name].dtype}"
        if yielded[name].dtype != tensor.dtype
        else f"{name} changed value"
        for name, tensor in expected.items()
        if yielded[name].dtype != tensor.dtype or not torch.equal(yielded[name], tensor)
    ]
    if changed:
        raise RuntimeError(
            "NeMo-RL's dtensor_params_generator no longer keeps each tensor's dtype and values "
            f"when given None ({', '.join(changed)}), so the export would round the checkpoint's "
            "FP32 tensors; check cwtrain against this NeMo-RL version"
        )


def _megatron_export(
    params: Any,
    path: str,
    allowed_names: list[str] | None,
    config_dir: str | None,
    shard_bytes: int,
    dtypes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Write Megatron-Bridge's HF-layout tensors in the dtypes it yields them in.

    Megatron-Bridge converts the weights on export. This function does not check whether it
    rounds FP32 tensors. A cast back to the checkpoint's dtype would hide that rounding from the
    synchronizer's layout check. So ``dtypes`` (the synchronizer sends it to every worker) is not
    applied. An export whose dtypes differ from the checkpoint's stops before anything is
    published.
    """
    del dtypes
    return _export(params, path, allowed_names, config_dir, shard_bytes)


if DTensorPolicyWorkerV2Impl is not None:

    @ray.remote(
        runtime_env=get_runtime_env_for_policy_worker("dtensor_policy_worker_v2")
    )  # pragma: no cover
    class CoreWeaveDTensorPolicyWorker(DTensorPolicyWorkerV2Impl):  # type: ignore[misc,valid-type]
        """DTensor v2 policy worker with an HF-layout checkpoint export."""

        @torch.no_grad()
        def export_hf_checkpoint(
            self,
            path: str,
            allowed_names: list[str] | None = None,
            config_dir: str | None = None,
            shard_bytes: int = SHARD_BYTES,
            dtypes: dict[str, str] | None = None,
        ) -> dict[str, Any]:
            """Export the policy's weights as an HF-layout checkpoint at ``path``.

            Args:
                path: The output directory. Rank 0 replaces it.
                allowed_names: Tensors to export. ``None`` exports all of them. When set, the
                    export fails unless every named tensor is found.
                config_dir: Directory whose non-weight files (``config.json``, tokenizer) are
                    copied into the checkpoint.
                shard_bytes: Approximate size of each safetensors shard.
                dtypes: Safetensors dtype of each tensor in the deployment's checkpoint.

            Returns:
                On rank 0: ``path``, ``tensors``, ``bytes``, ``shards`` and ``rank``. On other
                ranks: ``rank`` and ``tensors``.
            """
            offloaded = bool(getattr(self, "cpu_offload", False))
            if offloaded:
                self.model = self.move_to_cuda(self.model)
            try:
                return _export(
                    _dtensor_hf_params(self.model, self.dtype, dtypes),
                    path,
                    allowed_names,
                    config_dir,
                    shard_bytes,
                    dtypes,
                )
            finally:
                if offloaded:
                    self.model = self.move_to_cpu(self.model)


try:  # Megatron policy
    import ray
    from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )
except Exception:  # pragma: no cover
    MegatronPolicyWorkerImpl = None  # type: ignore[assignment,misc]

if MegatronPolicyWorkerImpl is not None:

    @ray.remote(
        runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker")
    )  # pragma: no cover
    class CoreWeaveMegatronPolicyWorker(MegatronPolicyWorkerImpl):  # type: ignore[misc,valid-type]
        """Megatron policy worker with an HF-layout checkpoint export."""

        @torch.no_grad()
        def export_hf_checkpoint(
            self,
            path: str,
            allowed_names: list[str] | None = None,
            config_dir: str | None = None,
            shard_bytes: int = SHARD_BYTES,
            dtypes: dict[str, str] | None = None,
        ) -> dict[str, Any]:
            """Export Megatron-Bridge's HF-layout tensors in the dtypes the bridge yields.

            Takes the same arguments as the DTensor worker. ``dtypes`` is accepted, because the
            synchronizer sends every worker the same call, but not applied. A cast back to the
            checkpoint's dtypes would hide any rounding from the synchronizer's layout check (see
            :func:`_megatron_export`).
            """
            return _megatron_export(
                self._iter_params_with_optional_kv_scales(),
                path,
                allowed_names,
                config_dir,
                shard_bytes,
                dtypes,
            )


__all__ = ["CoreWeaveDTensorPolicyWorker", "CoreWeaveMegatronPolicyWorker", "SHARD_BYTES"]
