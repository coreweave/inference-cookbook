# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""``policy.generation.backend: coreweave`` for NeMo-RL.

NeMo-RL is token-in, token-out. :meth:`CoreWeaveGeneration.generate` receives right-padded
``input_ids`` with ``input_lengths``. It returns ``output_ids`` (prompt + response, right padded),
per-token ``logprobs``, ``generation_lengths``, ``unpadded_sequence_lengths`` and ``truncated``,
with the same shapes as the vLLM backend's output.

Prompts go to the deployment as token ids. Responses come back as token ids when the endpoint
honors vLLM's ``return_tokens_as_token_ids``. Otherwise the text is re-tokenized and the stop
token is appended.

Configuration lives under ``policy.generation.coreweave_cfg``. The rollout keys (all but
``max_model_len``, ``require_logprobs`` and ``importance_sampling``) can also come from the
``CWTRAIN_ROLLOUT_*`` environment (see :mod:`cwtrain.integrations.rollout`):

``base_url``          OpenAI-compatible base URL of the deployment, ending in ``/v1``.
``model``             Served model name.
``api_key_file``      File holding the gateway key (not needed for the in-cluster frontend).
``concurrency``       Requests in flight (default 16; the NeMo-RL recipe's config sets 32).
``max_model_len``     Cap on prompt + response tokens; set it to
                      ``policy.max_total_sequence_length``.
``token_id_prompts``  Send prompts as token ids (default true). If the endpoint rejects them and
                      a text request succeeds, the run sends text from then on.
``seeds``             Seed every Gym request, each of which samples one choice (default true).
                      The backend's own requests sample all of a prompt's choices on one engine
                      and go unseeded.
``require_logprobs``  Fail the step when a response has no aligned log-probs (default true).
``importance_sampling`` NeMo-RL's importance-sampling correction compares the log-probs with its
                      own (default false; the recipe's config ties it to
                      ``loss_fn.use_importance_sampling_correction``). Needs ``temperature`` 1,
                      ``top_p`` 1 and no ``top_k``: the endpoint's log-probs come before them,
                      NeMo-RL's after.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from cwtrain.integrations.nemo_rl import _missing_dependency
from cwtrain.integrations.nvext import as_cache_salt, warn_without_wait
from cwtrain.integrations.rollout import (
    ENV_PREFIX,
    Completion,
    CompletionSampler,
    RolloutConfig,
    RolloutError,
    align_logprobs,
    check_stop_limits,
    omitted_stop_token,
    sample_prompt,
)

try:
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.generation.interfaces import GenerationInterface
except ModuleNotFoundError as exc:
    raise _missing_dependency(exc) from exc

log = logging.getLogger(__name__)

_ENV_KEYS = {
    "base_url": "BASE_URL",
    "model": "MODEL",
    "api_key": "API_KEY",
    "api_key_file": "API_KEY_FILE",
    "concurrency": "CONCURRENCY",
    "timeout": "TIMEOUT",
    "max_attempts": "MAX_ATTEMPTS",
    "logprobs": "LOGPROBS",
    "token_ids": "TOKEN_IDS",
    "token_id_prompts": "TOKEN_ID_PROMPTS",
    "seeds": "SEEDS",
}


def rollout_config_from(
    cfg: dict[str, Any], environ: dict[str, str] | None = None
) -> RolloutConfig:
    """Build the rollout config. ``coreweave_cfg`` keys override ``CWTRAIN_ROLLOUT_*``.

    An error about a value from ``coreweave_cfg`` names its key, not the environment variable.
    """
    env = dict(os.environ if environ is None else environ)
    labels: dict[str, str] = {}
    for key, suffix in _ENV_KEYS.items():
        if cfg.get(key) is not None:
            env[f"{ENV_PREFIX}{suffix}"] = str(cfg[key])
            labels[f"{ENV_PREFIX}{suffix}"] = f"coreweave_cfg.{key}"
    return RolloutConfig.from_env(env, labels=labels).without_routed_experts("NeMo-RL")


class CoreWeaveGeneration(GenerationInterface):
    """Rollouts from a Dedicated Inference deployment; weights arrive by hot-load."""

    def __init__(self, config: dict[str, Any], tokenizer: Any) -> None:
        self.cfg = config
        self.tokenizer = tokenizer
        cw = dict(config.get("coreweave_cfg") or {})
        self.rollout_config = rollout_config_from(cw)
        self.token_id_prompts = self.rollout_config.token_id_prompts
        self.max_model_len: int | None = cw.get("max_model_len")
        # NeMo-RL's importance-sampling correction reads the generation log-probs. A response
        # without aligned log-probs must stop the step, not pass zeros as real values.
        self.require_logprobs = bool(cw.get("require_logprobs", True))
        if self.require_logprobs and not self.rollout_config.logprobs:
            raise RolloutError(
                "coreweave_cfg.require_logprobs needs sampled-token log-probs: keep "
                "CWTRAIN_ROLLOUT_LOGPROBS enabled or set require_logprobs: false"
            )
        # NeMo-RL scores with policy.generation's temperature, top_p and top_k. The endpoint
        # reports log-probs before them (vLLM's default), so the two compare only without them.
        top_k = config.get("top_k")
        shaped = (
            float(config.get("temperature", 1.0)) != 1.0
            or float(config.get("top_p", 1.0)) != 1.0
            or (top_k is not None and int(top_k) > 0)
        )
        if bool(cw.get("importance_sampling", False)) and shaped:
            raise RolloutError(
                "the importance-sampling correction compares NeMo-RL's log-probs, which apply "
                "policy.generation's temperature, top_p and top_k, with the endpoint's, which "
                "it reports before them. Keep temperature 1.0, top_p 1.0 and top_k null, or turn "
                "the correction off (loss_fn.use_importance_sampling_correction: false)"
            )
        # As in NeMo-RL's vLLM backend, configured stops apply to every request, merged with
        # per-row stop strings.
        self.stop_strings: list[str] = list(config.get("stop_strings") or [])
        self.stop_token_ids: list[int] = [int(t) for t in (config.get("stop_token_ids") or [])]
        # Fail at setup, not at the first rollout, when the configured stops alone are too many.
        check_stop_limits(self.stop_strings, self.stop_token_ids)
        self._warned_missing_logprobs = False
        pad = config.get("_pad_token_id")
        self.pad_token_id = int(
            pad if pad is not None else tokenizer.pad_token_id or tokenizer.eos_token_id
        )
        self.eos_token_id = int(tokenizer.eos_token_id)
        self.weight_synchronizer: Any = None  # attached by the training setup
        self._refit_info: dict[str, Any] | None = None
        self.requests = 0
        self.misaligned_logprobs = 0
        print(
            f"[cwtrain] rollouts from {self.rollout_config.base_url} "
            f"model={self.rollout_config.model} token_id_prompts={self.token_id_prompts}",
            flush=True,
        )

    # -------------------------------------------------------------- lifecycle (no engines)

    def init_collective(self, ip: str, port: int, world_size: int, **kwargs: Any) -> list[Any]:
        return []

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def shutdown(self) -> bool:
        return True

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        self._refit_info = state_dict_info

    # ---------------------------------------------------------------------- generation

    def generate(self, data: BatchedDataDict, greedy: bool = False) -> BatchedDataDict:
        import torch

        input_ids = data["input_ids"]
        input_lengths = data["input_lengths"]
        batch = int(input_lengths.shape[0])
        stop_strings = data.get("stop_strings") if hasattr(data, "get") else None
        if stop_strings is None:
            stop_strings = [None] * batch
        prompts = [input_ids[i, : int(input_lengths[i])].tolist() for i in range(batch)]

        completions = asyncio.run(self._sample_rows(prompts, list(stop_strings), greedy))
        rows = [self._decode(c) for c in completions]
        exact = sum(1 for c in completions if c.token_ids is not None)
        print(
            f"[cwtrain] generated {len(rows)} responses in {self.requests} requests so far; "
            f"exact token ids for {exact}/{len(rows)}, misaligned log-probs total "
            f"{self.misaligned_logprobs}",
            flush=True,
        )
        missing = sum(1 for gen, lps, _t in rows if gen and not lps)
        if missing and self.require_logprobs:
            raise RolloutError(
                f"{missing}/{len(rows)} responses came back without aligned sampled-token "
                "log-probs, which NeMo-RL's importance-sampling correction would read as zeros. "
                "Use an endpoint that returns them (vLLM with logprobs), or set "
                "coreweave_cfg.require_logprobs: false while "
                "loss_fn.use_importance_sampling_correction is off"
            )
        if missing and not self._warned_missing_logprobs:
            self._warned_missing_logprobs = True
            log.warning(
                "%d/%d responses without aligned log-probs; their positions stay zero "
                "(require_logprobs is off)",
                missing,
                len(rows),
            )

        padded_input_length = int(input_ids.shape[1])
        max_gen = max((len(gen) for gen, _lp, _t in rows), default=0)
        total = padded_input_length + max_gen
        output_ids = torch.full((batch, total), self.pad_token_id, dtype=input_ids.dtype)
        logprobs = torch.zeros((batch, total), dtype=torch.float32)
        gen_lengths, seq_lengths, truncated = [], [], []
        for i, (gen, lps, trunc) in enumerate(rows):
            length = int(input_lengths[i])
            output_ids[i, :length] = input_ids[i, :length]
            if gen:
                output_ids[i, length : length + len(gen)] = torch.tensor(gen, dtype=input_ids.dtype)
            if lps:
                logprobs[i, length : length + len(lps)] = torch.tensor(lps, dtype=torch.float32)
            gen_lengths.append(len(gen))
            seq_lengths.append(length + len(gen))
            truncated.append(trunc)
        return BatchedDataDict(
            {
                "output_ids": output_ids,
                "logprobs": logprobs,
                "generation_lengths": torch.tensor(gen_lengths, dtype=torch.long),
                "unpadded_sequence_lengths": torch.tensor(seq_lengths, dtype=torch.long),
                "truncated": torch.tensor(truncated, dtype=torch.bool),
            }
        )

    async def _sample_rows(
        self, prompts: list[list[int]], stop_strings: list[Any], greedy: bool
    ) -> list[Completion]:
        """One request per distinct (prompt, stop) with ``n`` = its multiplicity."""
        groups: dict[tuple[tuple[int, ...], Any], list[int]] = {}
        for i, (prompt, stop) in enumerate(zip(prompts, stop_strings, strict=True)):
            key = (tuple(prompt), tuple(stop) if isinstance(stop, list) else stop)
            groups.setdefault(key, []).append(i)
        results: list[Completion | None] = [None] * len(prompts)
        sampler = CompletionSampler(self.rollout_config)
        try:

            async def one(key: tuple[tuple[int, ...], Any], rows: list[int]) -> None:
                prompt_ids, stop = key
                completions = await self._sample_group(
                    sampler, list(prompt_ids), len(rows), stop, greedy
                )
                for row, completion in zip(rows, completions, strict=True):
                    results[row] = completion

            await asyncio.gather(*(one(k, rows) for k, rows in groups.items()))
        finally:
            await sampler.aclose()
        self.requests += len(groups)
        return [c for c in results if c is not None]

    async def _sample_group(
        self, sampler: CompletionSampler, prompt_ids: list[int], n: int, stop: Any, greedy: bool
    ) -> list[Completion]:
        max_tokens = int(self.cfg["max_new_tokens"])
        if self.max_model_len:
            max_tokens = max(1, min(max_tokens, int(self.max_model_len) - len(prompt_ids)))
        row_stop = [stop] if isinstance(stop, str) else (list(stop) if stop else [])
        stops = list(dict.fromkeys([*self.stop_strings, *row_stop]))
        kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": 0.0 if greedy else float(self.cfg.get("temperature", 1.0)),
            "top_p": float(self.cfg.get("top_p", 1.0) or 1.0),
            "top_k": self.cfg.get("top_k"),
            "stop": stops or None,
            "stop_token_ids": self.stop_token_ids or None,
            # NeMo-RL's own backends keep the stop string in the output, so do the same here.
            "include_stop_str_in_output": True if stops else None,
            "cache_salt": self._cache_salt(),
        }
        completions, still = await sample_prompt(
            sampler,
            prompt_ids,
            n,
            text=lambda: self.tokenizer.decode(prompt_ids, skip_special_tokens=False),
            token_id_prompts=self.token_id_prompts,
            **kwargs,
        )
        # Rows finish in any order. After one row falls back to text, none switches back.
        self.token_id_prompts = self.token_id_prompts and still
        return completions

    def _cache_salt(self) -> str | None:
        """The served identity, so that no version reuses prompts another one cached."""
        sync = self.weight_synchronizer
        identity = getattr(sync, "served_identity", None)
        if not identity:
            return None
        warn_without_wait(getattr(getattr(sync, "config", None), "wait", True))
        return as_cache_salt(identity)

    def _decode(self, completion: Completion) -> tuple[list[int], list[float], bool]:
        """Response token ids, aligned log-probs (or none) and the truncation flag."""
        if completion.token_ids is not None:
            gen = list(completion.token_ids)
        else:
            gen = list(self.tokenizer.encode(completion.text, add_special_tokens=False))
            stop_id = omitted_stop_token(completion, gen, self.eos_token_id)
            if stop_id is not None:
                gen.append(stop_id)
        # Same rule as the slime/miles path (see align_logprobs).
        lps: list[float] = []
        if completion.token_logprobs is not None:
            aligned = align_logprobs(completion, gen, self.tokenizer)
            if aligned is None:
                self.misaligned_logprobs += 1
            else:
                lps = aligned
        return gen, lps, completion.finish_reason == "length"


__all__ = ["CoreWeaveGeneration", "rollout_config_from"]
