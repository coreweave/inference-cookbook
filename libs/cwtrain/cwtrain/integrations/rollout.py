# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Sample GRPO groups from an OpenAI-compatible completions endpoint.

The trainer owns no inference engines. Rollouts come from a CoreWeave Dedicated Inference
deployment, through its gateway or, for a trainer in the same cluster, its frontend Service. This
module is the framework-agnostic part of the miles and slime rollout functions in
``training/recipes/``. It handles configuration, one ``/v1/completions`` request per GRPO group
with ``n`` choices, retries, and filling the framework's ``Sample`` objects.

Configuration (environment of the process that runs the rollout function):

==============================  =====================================================
``CWTRAIN_ROLLOUT_BASE_URL``     OpenAI-compatible base URL, ending in ``/v1`` (required).
``CWTRAIN_ROLLOUT_MODEL``        Served model name, the deployment's ``model.name`` (required).
``CWTRAIN_ROLLOUT_API_KEY``      Bearer token, or ``CWTRAIN_ROLLOUT_API_KEY_FILE`` with a path.
                                 Not needed for a deployment's in-cluster frontend Service.
``CWTRAIN_ROLLOUT_CONCURRENCY``  Group requests in flight (default 16), each with ``n``
                                 sequences. A replica decodes at most its deployment's
                                 ``autoscaling.concurrency`` sequences at once.
``CWTRAIN_ROLLOUT_TIMEOUT``      Per-request timeout in seconds (default 600).
``CWTRAIN_ROLLOUT_MAX_ATTEMPTS`` Attempts per request on 429/5xx/transport errors (default 6).
``CWTRAIN_ROLLOUT_LOGPROBS``     ``1`` (default) asks for per-token log-probs and stores them as
                                 ``rollout_log_probs`` when they align with the response tokens;
                                 ``0`` never sets them.
``CWTRAIN_ROLLOUT_TOKEN_IDS``    ``1`` (default) asks vLLM to report the sampled token ids
                                 (``return_tokens_as_token_ids``); ``0`` re-tokenizes the text.
``CWTRAIN_ROLLOUT_TOKEN_ID_PROMPTS`` ``1`` (default) sends each prompt as the token ids the
                                 trainer encodes and trains on; ``0`` sends the rendered text.
``CWTRAIN_ROLLOUT_ROUTED_EXPERTS`` ``1`` asks a Mixture-of-Experts deployment for each sequence's
                                 routed experts; ``0`` (default) does not. See "Routed experts".
``CWTRAIN_ROLLOUT_SEEDS``        ``1`` (default) seeds every request that samples one choice of a
                                 group on its own; ``0`` never sends a seed. See "Seeds".
==============================  =====================================================

Responses
---------

When the endpoint reports them, responses are the exact sampled token ids, with one log-prob each.
They include the token that ended the response.

An endpoint that drops the token-id flag returns text. The text is re-tokenized with the trainer's
tokenizer. vLLM leaves the end-of-sequence or stop token out of the text, so it is appended. The
stopping action then stays in the trained sequence. A matched stop string adds nothing.

Re-tokenized log-probs are kept only when, for *every* sample of the batch, the endpoint's token
strings name exactly the re-tokenized tokens, one per position. Equal counts are not enough: the
same text can re-tokenize into tokens other than those sampled. Otherwise ``rollout_log_probs``
stays unset on the whole batch, and the trainer recomputes log-probs. (The trainers tensorize it
batch-wide.)

slime's advantage step needs a per-token tensor to shape its zero KL. A run without aligned
log-probs therefore also needs the ``0003`` patch in the slime recipe.

slime and miles read every sample's log-probs with ``--use-tis``, ``--use-rollout-logprobs`` or
``--get-mismatch-metrics``, and cannot train a batch without them. With any of these flags,
:meth:`RolloutConfig.for_trainer` makes them required. A group whose log-probs do not align is then
sampled again, up to :data:`LOGPROB_RESAMPLES` times, and the rollout fails only if it still does
not align. The trainer must also sample at temperature 1. The endpoint reports log-probs before
temperature (vLLM's default), but slime and miles score at ``--rollout-temperature``, so at any
other temperature the two are not comparable. With ``--get-mismatch-metrics`` alone, which only
reports, that is a warning.

Prompts
-------

Prompts go out as token ids, so the policy samples from exactly the sequence the trainer scores.
The endpoint does not re-tokenize them. (CoreWeave's frontend ignores ``add_special_tokens``.)

If an endpoint rejects a token-id prompt with a 4xx, the prompt is retried as text. After a text
request succeeds, the rest of the run sends text, with ``add_special_tokens: false``, which vLLM
honors. Groups already in flight still try token ids first, so the first batch against such an
endpoint costs one extra request per group. If the text request also gets a 4xx, nothing
changes.

A request carries at most four stop strings and four stop token ids. CoreWeave's frontend rejects
more, so such a request fails before it is sent.

Prefix cache
------------

The deployment caches the KV of prompt blocks, and with routed experts their routes. A hot-load
with the ``PRESERVE`` prompt-cache policy keeps them, so a later version could serve a prompt from
blocks that an earlier one computed. A cache salt (Dynamo's ``nvext.cache_salt``) goes into the
cache key, so requests reuse blocks only when their salts are equal. slime and miles salt each
rollout with :func:`rollout_cache_salt`. NeMo-RL salts with the identity that the deployment
serves, on its own path and on Gym's (see :mod:`cwtrain.integrations.nvext`).

The salt must change only after every replica serves the new version, as it does when the
hot-load hook waits for the hot-load (the default). A request with the new salt could otherwise
reach a replica that has not switched yet, which would cache old KV under the new salt. Without
the wait, the rollout function logs a warning.

Seeds
-----

A request that samples one choice of a group on its own carries a random ``seed``, below 2**62. That
is each request of a group with routed experts, and each of the Gym model server's. Without a seed,
a replica takes a request's sampling noise from its engine's random state, and every replica starts
that state from the same default seed. So replicas that start together sample a prompt identically
when it arrives as the same request on each, and a group's choices on different replicas could
come back equal. A request with ``n`` > 1 samples every choice on one engine, which draws distinct
noise for each, so it goes unseeded: a seed costs 1-3% of decode throughput. Seeds make sampling
independent of the replica, not reproducible: decoding on a deployment is not bit-exact.

Routed experts
--------------

A Mixture-of-Experts deployment started with the ``enable-return-routed-experts`` engine option
reports the experts that each token was routed to in every layer. A trainer that replays them
(rollout routing replay) then runs the same experts as the engine. Each request asks for them in
Dynamo's ``nvext.extra_fields``.

A response carries the routes of one sequence, so a group of ``n`` becomes ``n`` requests of one
choice each, sent together. Each completion's ``routed_experts`` has shape
``[prompt + response - 1, layers, top_k]``: one row for every token that the engine ran through
the model, which is every token but the last one sampled. Rows of dense layers are zeros.

Routes line up only with exact tokens. They need token-id prompts and the sampled token ids, so a
rejected token-id prompt is not retried as text. A response without routes, or with a row count
that does not match its tokens, is an error: a trainer that replays routes cannot train on it.

slime and miles replay routes with ``--use-rollout-routing-replay``. With that flag,
:meth:`RolloutConfig.for_trainer` turns routes on, and :func:`fill_group` stores them as
``rollout_routed_experts``: int32, in the shape that both expect.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import logging
import math
import os
import random
import secrets
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from cwtrain.errors import CWTrainError
from cwtrain.integrations._env import flag, positive_float, positive_int
from cwtrain.integrations.nvext import (
    MAX_CACHE_SALT,
    as_cache_salt,
    rollout_cache_salt,
    with_nvext,
)

log = logging.getLogger(__name__)

ENV_PREFIX = "CWTRAIN_ROLLOUT_"
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class RolloutError(CWTrainError):
    """The completions endpoint returned something the trainer cannot train on."""


#: slime and miles flags that read ``rollout_log_probs`` from every sample of a batch.
LOGPROB_READERS = ("use_tis", "use_rollout_logprobs", "get_mismatch_metrics")
#: With log-probs required, how many more times a group whose log-probs do not align is sampled.
LOGPROB_RESAMPLES = 2
#: The slime and miles flag that hands sampling to a per-sample generate function, such as a
#: multi-turn or agent loop. The rollout functions here never call one: they sample each prompt in
#: one turn.
GENERATE_FUNCTION_FLAG = "--custom-generate-function-path"
_NO_GENERATE_FUNCTION = (
    f"{GENERATE_FUNCTION_FLAG} is not supported: the rollout function samples each prompt in one "
    "turn and never calls a generate function, so a multi-turn or agent loop would not run"
)


def refuse_generate_function(trainer_args: Iterable[str]) -> None:
    """Raise if a trainer command line names a generate function, so that a launcher stops before
    it stages anything. :meth:`RolloutConfig.for_trainer` refuses one set by any means, but only
    once the trainer runs the rollout function.

    Raises:
        CWTrainError: ``trainer_args`` holds :data:`GENERATE_FUNCTION_FLAG`.
    """
    if any(arg.partition("=")[0] == GENERATE_FUNCTION_FLAG for arg in trainer_args):
        raise CWTrainError(_NO_GENERATE_FUNCTION)


#: CoreWeave's OpenAI frontend rejects a request with more stop strings or stop token ids.
MAX_STOPS = 4


def check_stop_limits(
    stop: Sequence[str] | None = None, stop_token_ids: Sequence[int] | None = None
) -> None:
    """Raise before sending when a request would carry more stops than the endpoint accepts."""
    for what, items in (("stop strings", stop), ("stop token ids", stop_token_ids)):
        if items and len(items) > MAX_STOPS:
            raise RolloutError(
                f"a completions request takes at most {MAX_STOPS} {what}, got {len(items)}"
            )


#: Request seeds stay below this, so that vLLM's ``seed + n - 1`` for a request's last choice still
#: fits the engine's int64.
SEED_LIMIT = 2**62


def request_seed() -> int:
    """A random seed for one request (see "Seeds" above)."""
    return secrets.randbelow(SEED_LIMIT)


@dataclass(frozen=True)
class RolloutConfig:
    base_url: str
    model: str
    api_key: str | None = None
    concurrency: int = 16
    timeout: float = 600.0
    max_attempts: int = 6
    logprobs: bool = True
    #: Ask vLLM-based endpoints to report sampled tokens as ``token_id:N`` strings, so responses
    #: are exact token ids. Endpoints that drop the flag ignore it.
    token_ids: bool = True
    #: Send prompts as token ids, so the endpoint cannot tokenize them differently from the
    #: trainer.
    token_id_prompts: bool = True
    #: Ask a Mixture-of-Experts deployment for each sequence's routed experts (see "Routed
    #: experts" above).
    routed_experts: bool = False
    #: Fail a batch in which a sample's log-probs do not align, instead of dropping them from the
    #: whole batch. For a trainer that reads them from every sample (see :meth:`for_trainer`).
    require_logprobs: bool = False
    #: Seed every request that samples one choice of a group on its own (see "Seeds" above).
    seeds: bool = True

    def __post_init__(self) -> None:
        for name, value in (("concurrency", self.concurrency), ("max_attempts", self.max_attempts)):
            if value < 1:
                raise CWTrainError(f"RolloutConfig.{name} must be at least 1, got {value}")
        if not (math.isfinite(self.timeout) and self.timeout > 0):
            raise CWTrainError(
                f"RolloutConfig.timeout must be a finite number above 0, got {self.timeout}"
            )
        if self.routed_experts and not (self.logprobs and self.token_ids and self.token_id_prompts):
            raise CWTrainError(
                "RolloutConfig.routed_experts needs exact tokens: keep logprobs, token_ids and "
                "token_id_prompts on"
            )
        if self.require_logprobs and not self.logprobs:
            raise CWTrainError("RolloutConfig.require_logprobs needs logprobs on")

    def without_routed_experts(self, trainer: str) -> RolloutConfig:
        """This config with routed experts off, for a trainer that cannot replay them.

        Routes split each group into one request per choice and add hundreds of bytes per token,
        so a trainer that never reads them should not ask for them.
        """
        if not self.routed_experts:
            return self
        log.warning(
            "%s does not replay routed experts; ignoring %sROUTED_EXPERTS", trainer, ENV_PREFIX
        )
        return dataclasses.replace(self, routed_experts=False)

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None, *, labels: Mapping[str, str] | None = None
    ) -> RolloutConfig:
        """Read the ``CWTRAIN_ROLLOUT_*`` settings from ``environ`` (default: ``os.environ``).

        Args:
            environ: The environment to read. Defaults to ``os.environ``.
            labels: Maps a variable to the name that its errors show instead. Use it when the
                caller copied the value from another source, such as NeMo-RL's ``coreweave_cfg``.

        Raises:
            CWTrainError: A required setting is missing, or a value is not valid.
        """
        env = os.environ if environ is None else environ
        names = labels or {}
        base_url = env.get(f"{ENV_PREFIX}BASE_URL", "").strip().rstrip("/")
        model = env.get(f"{ENV_PREFIX}MODEL", "").strip()
        for variable, value in ((f"{ENV_PREFIX}BASE_URL", base_url), (f"{ENV_PREFIX}MODEL", model)):
            if not value:
                raise CWTrainError(f"{names.get(variable, variable)} is required")
        key = env.get(f"{ENV_PREFIX}API_KEY", "").strip()
        key_file = env.get(f"{ENV_PREFIX}API_KEY_FILE", "").strip()
        if not key and key_file and Path(key_file).exists():
            key = Path(key_file).read_text().strip()
        return cls(
            base_url=base_url,
            model=model,
            api_key=key or None,
            concurrency=positive_int(env, f"{ENV_PREFIX}CONCURRENCY", 16, labels=names),
            timeout=positive_float(env, f"{ENV_PREFIX}TIMEOUT", 600.0, labels=names),
            max_attempts=positive_int(env, f"{ENV_PREFIX}MAX_ATTEMPTS", 6, labels=names),
            logprobs=flag(env, f"{ENV_PREFIX}LOGPROBS", True, labels=names),
            token_ids=flag(env, f"{ENV_PREFIX}TOKEN_IDS", True, labels=names),
            token_id_prompts=flag(env, f"{ENV_PREFIX}TOKEN_ID_PROMPTS", True, labels=names),
            routed_experts=flag(env, f"{ENV_PREFIX}ROUTED_EXPERTS", False, labels=names),
            seeds=flag(env, f"{ENV_PREFIX}SEEDS", True, labels=names),
        )

    @classmethod
    def for_trainer(cls, args: Any, environ: Mapping[str, str] | None = None) -> RolloutConfig:
        """:meth:`from_env`, with routed experts on exactly when the trainer replays them, and
        log-probs required when it reads them.

        slime and miles replay the rollout's routes with ``--use-rollout-routing-replay``
        (``args.use_rollout_routing_replay``), and then fail on a sample without them. Without
        it, routes would cost a request per choice and go unread. The flag decides, and an
        explicit ``CWTRAIN_ROLLOUT_ROUTED_EXPERTS`` that disagrees is ignored with a warning. For
        a trainer without the flag, the environment decides.

        With one of :data:`LOGPROB_READERS` on, such as ``--use-tis``, log-probs are required
        (see "Responses" above).

        The rollout functions replace the trainer's per-sample generate step, so a
        ``--custom-generate-function-path`` would go unused. It is refused, so that a multi-turn
        run does not quietly train single-turn.

        Raises:
            CWTrainError: The trainer names a generate function. Or it reads log-probs, but they
                are off, or it trains on them at a temperature other than 1.
        """
        if getattr(args, "custom_generate_function_path", None):
            raise CWTrainError(_NO_GENERATE_FUNCTION)
        env = os.environ if environ is None else environ
        config = cls.from_env(env)
        flag = getattr(args, "use_rollout_routing_replay", None)
        if flag is not None:
            replay = bool(flag)
            explicit = env.get(f"{ENV_PREFIX}ROUTED_EXPERTS", "").strip()
            if explicit and config.routed_experts != replay:
                log.warning(
                    "ignoring %sROUTED_EXPERTS=%s: --use-rollout-routing-replay is %s",
                    ENV_PREFIX,
                    explicit,
                    "on" if replay else "off",
                )
            config = dataclasses.replace(config, routed_experts=replay)
        readers = [
            "--" + name.replace("_", "-") for name in LOGPROB_READERS if getattr(args, name, False)
        ]
        if not readers:
            return config
        if not config.logprobs:
            raise CWTrainError(
                f"{readers[0]} reads the endpoint's log-probs: keep {ENV_PREFIX}LOGPROBS on"
            )
        temperature = float(getattr(args, "rollout_temperature", 1.0))
        if temperature != 1.0:
            mismatch = (
                f"{readers[0]} compares log-probs that the trainer computes at "
                f"--rollout-temperature {temperature:g} with the endpoint's, which it reports "
                "before temperature"
            )
            if readers != ["--get-mismatch-metrics"]:
                raise CWTrainError(f"{mismatch}. Sample at temperature 1, or turn {readers[0]} off")
            # the metrics only report; training does not use them
            log.warning("%s, so the mismatch metrics are off", mismatch)
        return dataclasses.replace(config, require_logprobs=True)


@dataclass(frozen=True)
class Completion:
    index: int
    text: str
    finish_reason: str | None
    #: vLLM's reason for a ``stop`` finish: ``None`` for end-of-sequence, else the stop string or
    #: stop token id that matched.
    stop_reason: Any = None
    token_logprobs: tuple[float, ...] | None = None
    #: The endpoint's string for each sampled token (vLLM reports the vocabulary symbol or its
    #: decoded text), used to check that log-probs belong to the re-tokenized tokens.
    token_strings: tuple[str, ...] | None = None
    #: Exact sampled token ids when the endpoint reported them (see ``RolloutConfig.token_ids``).
    token_ids: tuple[int, ...] | None = None
    #: The experts that each token was routed to, ``[prompt + response - 1, layers, top_k]``, when
    #: requested (see ``RolloutConfig.routed_experts``).
    routed_experts: np.ndarray | None = field(default=None, compare=False, repr=False)
    #: Prompt tokens the deployment served from its prefix cache, when it reported them.
    cached_tokens: int | None = None


class CompletionSampler:
    """One ``/v1/completions`` request per GRPO group, ``n`` choices per request.

    With routed experts, each choice is its own request (see "Routed experts" above).
    """

    def __init__(self, config: RolloutConfig, *, transport: httpx.AsyncBaseTransport | None = None):
        self.config = config
        headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
        self._client = httpx.AsyncClient(
            base_url=config.base_url, headers=headers, timeout=config.timeout, transport=transport
        )
        self._semaphore = asyncio.Semaphore(config.concurrency)
        #: Set to False for the rest of the run when the endpoint refuses token-id prompts.
        self.token_id_prompts = config.token_id_prompts
        self._loop: asyncio.AbstractEventLoop | None = None  # set by shared_sampler
        log.info(
            "rollouts from %s model=%s concurrency=%d",
            config.base_url,
            config.model,
            config.concurrency,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def sample(
        self,
        prompt: str | Sequence[int],
        n: int,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        top_k: int | None = None,
        stop: Sequence[str] | None = None,
        stop_token_ids: Sequence[int] | None = None,
        include_stop_str_in_output: bool | None = None,
        cache_salt: str | None = None,
    ) -> list[Completion]:
        """Return exactly ``n`` completions for ``prompt``, ordered by choice index.

        ``prompt`` is text or a list of token ids. The OpenAI completions API accepts both. Token
        ids avoid re-tokenizing the trainer's prompt. ``stop_token_ids`` and
        ``include_stop_str_in_output`` are vLLM extensions of the API. ``cache_salt`` keys the
        deployment's prefix cache (see "Prefix cache" above); empty or ``None`` sends none.

        Raises:
            RolloutError: The request has too many stops or a cache salt that vLLM rejects, fails
                after all retries, gets a status that is not retried, or returns a choice count
                other than ``n``. With routed experts: the prompt is text, or a response lacks
                routes or token ids, or its routes do not line up with its tokens.
        """
        check_stop_limits(stop, stop_token_ids)
        if cache_salt and as_cache_salt(cache_salt) != cache_salt:
            raise RolloutError(
                f"a cache salt takes at most {MAX_CACHE_SALT} characters and none of @ / \\ or "
                f"NUL; as_cache_salt() makes one from any name, got {cache_salt!r}"
            )
        payload: dict[str, Any] = {
            "model": self.config.model,
            "prompt": prompt if isinstance(prompt, str) else list(prompt),
            "n": int(n),
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": False,
        }
        if stop:
            payload["stop"] = list(stop)
        if stop_token_ids:
            payload["stop_token_ids"] = [int(t) for t in stop_token_ids]
        if include_stop_str_in_output is not None:
            payload["include_stop_str_in_output"] = bool(include_stop_str_in_output)
        if top_k is not None and top_k > 0:
            payload["top_k"] = int(top_k)
        if cache_salt:
            payload = with_nvext(payload, cache_salt=cache_salt)
        if isinstance(prompt, str):
            # The trainer encodes prompts without special tokens. vLLM adds them (often a BOS)
            # unless told not to, and then the served prompt differs from the scored one.
            # CoreWeave's frontend ignores this flag, so prompts normally go out as token ids.
            payload["add_special_tokens"] = False
        if self.config.logprobs:
            payload["logprobs"] = 1
            if self.config.token_ids:
                payload["return_tokens_as_token_ids"] = True
        if self.config.routed_experts:
            return await self._sample_with_routes(prompt, n, payload)
        async with self._semaphore:
            body = await self._post_with_retry(payload)
        choices = sorted(body.get("choices") or [], key=lambda c: c.get("index", 0))
        if len(choices) != n:
            raise RolloutError(f"endpoint returned {len(choices)} choices for n={n}")
        return [self._completion(i, c, body) for i, c in enumerate(choices)]

    def _completion(self, index: int, choice: dict[str, Any], body: dict[str, Any]) -> Completion:
        logprobs = self.config.logprobs
        return Completion(
            index=index,
            cached_tokens=_cached_tokens(body),
            text=choice.get("text") or "",
            finish_reason=choice.get("finish_reason"),
            stop_reason=choice.get("stop_reason"),
            token_logprobs=_token_logprobs(choice) if logprobs else None,
            token_strings=_token_strings(choice) if logprobs else None,
            token_ids=_token_ids(choice) if logprobs and self.config.token_ids else None,
        )

    async def _sample_with_routes(
        self, prompt: str | Sequence[int], n: int, payload: dict[str, Any]
    ) -> list[Completion]:
        """``n`` requests of one choice each, because a response carries one sequence's routes."""
        if isinstance(prompt, str):
            raise RolloutError("routed experts need a token-id prompt")
        payload = with_nvext({**payload, "n": 1}, extra_fields=["routed_experts"])
        # Each choice is its own request and may reach another replica, so each gets its own
        # seed (see "Seeds" above). Equal seeds would sample equal choices.
        payloads = [
            {**payload, "seed": request_seed()} if self.config.seeds else payload for _ in range(n)
        ]
        async with self._semaphore:
            bodies = await _gather_all([self._post_with_retry(body) for body in payloads])
        completions = []
        for index, body in enumerate(bodies):
            choices = body.get("choices") or []
            if len(choices) != 1:
                raise RolloutError(f"endpoint returned {len(choices)} choices for n=1")
            completion = self._completion(index, choices[0], body)
            if completion.token_ids is None:
                raise RolloutError(
                    "routed experts need the sampled token ids, and the endpoint did not report "
                    "them"
                )
            nvext = body.get("nvext")
            value = nvext.get("routed_experts") if isinstance(nvext, dict) else None
            if value is None:
                raise RolloutError(
                    "the endpoint returned no routed experts; start the deployment with the "
                    "enable-return-routed-experts engine option"
                )
            routes = decode_routed_experts(value)
            rows = len(prompt) + len(completion.token_ids) - 1
            if routes.shape[0] != rows:
                raise RolloutError(
                    f"routed experts have {routes.shape[0]} rows for {len(prompt)} prompt and "
                    f"{len(completion.token_ids)} response tokens, not {rows}"
                )
            completions.append(dataclasses.replace(completion, routed_experts=routes))
        return completions

    async def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        delay = 1.0
        last = ""
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                response = await self._client.post("/completions", json=payload)
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    try:
                        return response.json()
                    except ValueError:
                        # A gateway or portal page served with 200: retry like a 503.
                        last = f"HTTP 200 with a non-JSON body: {response.text[:200]!r}"
                else:
                    last = f"HTTP {response.status_code}: {response.text[:200]}"
                    if response.status_code not in RETRY_STATUS:
                        raise RolloutError(f"completions request failed: {last}")
            if attempt == self.config.max_attempts:
                break
            sleep = min(30.0, delay) * (0.5 + random.random())
            log.warning("completions request failed (%s); retrying in %.1fs", last, sleep)
            await asyncio.sleep(sleep)
            delay *= 2
        raise RolloutError(
            f"completions request failed after {self.config.max_attempts} attempts: {last}"
        )


async def sample_prompt(
    sampler: CompletionSampler,
    prompt_ids: Sequence[int],
    n: int,
    *,
    text: Callable[[], str],
    token_id_prompts: bool,
    **kwargs: Any,
) -> tuple[list[Completion], bool]:
    """Sample ``n`` completions for a prompt; also return whether token-id prompts still apply.

    With ``token_id_prompts``, the prompt goes out as ``prompt_ids``. On a 4xx, it is retried once
    as ``text()``. Token-id prompts are dropped only if the text request succeeds. A 4xx with
    another cause (a prompt that is too long, a bad sampling parameter) then still fails, and the
    run does not switch to the less exact text path. Routed experts need exact tokens, so with
    them the 4xx is raised instead.

    Raises:
        RolloutError: The request fails. If the text retry also fails, the token-id error is
            raised.
    """
    if not token_id_prompts:
        return await sampler.sample(text(), n, **kwargs), False
    try:
        return await sampler.sample(prompt_ids, n, **kwargs), True
    except RolloutError as exc:
        if "HTTP 4" not in str(exc) or sampler.config.routed_experts:
            raise
        rejected = exc
    try:
        completions = await sampler.sample(text(), n, **kwargs)
    except RolloutError as text_error:
        # Report the token-id error. The text error stays attached as its context.
        log.warning(
            "endpoint rejected a token-id prompt (%s) and the text retry failed too (%s); "
            "keeping token-id prompts",
            rejected,
            text_error,
        )
        raise rejected  # noqa: B904 - chaining the text error as the cause would mislead
    log.warning("endpoint rejected token-id prompts (%s); sending text prompts", rejected)
    return completions, False


_shared_samplers: dict[tuple[int, RolloutConfig], CompletionSampler] = {}


def shared_sampler(config: RolloutConfig) -> CompletionSampler:
    """A sampler reused by every rollout on the running event loop, so connections stay pooled.

    slime and miles run each rollout coroutine on one long-lived background loop. An httpx client
    belongs to the loop it first ran on, so there is one sampler per loop and configuration. A
    cached sampler keeps what it learned about the endpoint, such as a switch to text prompts.
    """
    loop = asyncio.get_running_loop()
    key = (id(loop), config)
    sampler = _shared_samplers.get(key)
    if sampler is None or sampler._loop is not loop:
        sampler = CompletionSampler(config)
        sampler._loop = loop
        _shared_samplers[key] = sampler
    return sampler


def _token_logprobs(choice: dict[str, Any]) -> tuple[float, ...] | None:
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return None
    values = logprobs.get("token_logprobs")
    if not isinstance(values, list) or any(v is None for v in values):
        return None
    return tuple(float(v) for v in values)


def _token_strings(choice: dict[str, Any]) -> tuple[str, ...] | None:
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return None
    tokens = logprobs.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
        return None
    return tuple(tokens)


def same_token(tokenizer: Any, token_id: int, piece: str) -> bool:
    """Whether the endpoint's token string ``piece`` names ``token_id``.

    Depending on its version, vLLM reports each sampled token as its vocabulary symbol (``Ġthe``)
    or as its decoded text (`` the``). Both are accepted.
    """
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if convert is not None:
        try:
            if convert(token_id) == piece:
                return True
        except Exception:  # noqa: BLE001 - fall through to the decoded form
            pass
    return tokenizer.decode([token_id]) == piece


def same_tokens(tokenizer: Any, token_ids: Sequence[int], pieces: Sequence[str]) -> bool:
    """:func:`same_token` for a whole response, with one vocabulary lookup for all positions.

    Only positions whose vocabulary symbol does not match fall back to decoding that token. A
    tokenizer that cannot convert a list of ids is checked one token at a time.
    """
    if len(token_ids) != len(pieces):
        return False
    symbols: Sequence[Any] | None = None
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if convert is not None:
        try:
            converted = convert(list(token_ids))
            if isinstance(converted, list | tuple) and len(converted) == len(token_ids):
                symbols = converted
        except Exception:  # noqa: BLE001 - checked token by token below
            pass
    if symbols is None:
        return all(
            same_token(tokenizer, tid, piece) for tid, piece in zip(token_ids, pieces, strict=True)
        )
    for token_id, piece, symbol in zip(token_ids, pieces, symbols, strict=True):
        if symbol != piece and tokenizer.decode([token_id]) != piece:
            return False
    return True


def _token_ids(choice: dict[str, Any]) -> tuple[int, ...] | None:
    """Sampled token ids from vLLM's ``token_id:N`` token strings, if the endpoint kept the flag."""
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return None
    tokens = logprobs.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        return None
    ids = []
    for token in tokens:
        if not isinstance(token, str) or not token.startswith("token_id:"):
            return None
        try:
            ids.append(int(token[len("token_id:") :]))
        except ValueError:
            return None
    return tuple(ids)


def decode_routed_experts(value: Any) -> np.ndarray:
    """Decode a response's ``nvext.routed_experts``.

    Dynamo sends ``{"data", "shape", "start", "dtype"}``:

    - ``data`` is base64 of the array's raw C-order bytes;
    - ``shape`` is ``[rows, layers, top_k]``;
    - ``dtype`` is the element type, ``uint8`` for models with at most 256 experts, else
      ``uint16``;
    - ``start`` is the sequence position of the first row.

    Returns:
        A read-only array of shape ``[rows, layers, top_k]``.

    Raises:
        RolloutError: The value is malformed, or its rows do not start at the first token.
    """
    if not isinstance(value, dict):
        raise RolloutError(f"routed experts must be an object, got {type(value).__name__}")
    shape = value.get("shape")
    if not (
        isinstance(shape, list)
        and len(shape) == 3
        and all(isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape)
    ):
        raise RolloutError(f"routed experts shape must be [rows, layers, top_k], got {shape!r}")
    start = value.get("start")
    if start != 0:
        # Replay needs a row for every token, prompt included.
        raise RolloutError(f"routed experts start at token {start!r}, not at the first token")
    try:
        dtype = np.dtype(value.get("dtype") or "")
    except TypeError:
        dtype = None
    if dtype is None or dtype.kind not in "iu":
        raise RolloutError(
            f"routed experts dtype must be an integer type, got {value.get('dtype')!r}"
        )
    try:
        raw = base64.b64decode(value.get("data"), validate=True)
    except (TypeError, ValueError):
        raise RolloutError("routed experts data is not base64") from None
    expected = math.prod(shape) * dtype.itemsize
    if len(raw) != expected:
        raise RolloutError(
            f"routed experts data has {len(raw)} bytes, not the {expected} that shape {shape} of "
            f"{dtype} needs"
        )
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


async def _gather_all(awaitables: Sequence[Awaitable[Any]]) -> list[Any]:
    """Await ``awaitables`` concurrently. On the first error, cancel the rest and wait for them."""
    tasks = [asyncio.ensure_future(a) for a in awaitables]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _cached_tokens(body: dict[str, Any]) -> int | None:
    """``usage.prompt_tokens_details.cached_tokens``, when the response has it."""
    usage = body.get("usage")
    details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
    value = details.get("cached_tokens") if isinstance(details, dict) else None
    return int(value) if isinstance(value, int) else None


def align_logprobs(
    completion: Completion, response_ids: Sequence[int], tokenizer: Any
) -> list[float] | None:
    """The endpoint's sampled-token log-probs if they belong to exactly the response's tokens.

    With exact token ids from the endpoint, the response *is* those tokens, so only the count is
    checked. For re-tokenized text, matching counts are not enough. ``eq`` + ``uation`` and
    ``equ`` + ``ation`` are both two tokens of the same text, but their log-probs differ. So each
    position's token string must name the re-tokenized token there.

    A response that stopped on end-of-sequence has one more entry than the text has tokens (the
    stop token). That entry is dropped.

    Returns:
        The log-probs, or ``None`` when they do not align.
    """
    lp, pieces = completion.token_logprobs, completion.token_strings
    if lp is None or pieces is None or len(pieces) != len(lp):
        return None
    n = len(response_ids)
    if completion.token_ids is not None:
        return list(lp) if len(lp) == n else None
    if len(lp) == n + 1 and completion.finish_reason == "stop":
        lp, pieces = lp[:-1], pieces[:-1]
    elif len(lp) != n:
        return None
    if not same_tokens(tokenizer, response_ids, pieces):
        return None
    return list(lp)


def omitted_stop_token(
    completion: Completion, response_ids: Sequence[int], eos_token_id: int | None
) -> int | None:
    """The token that ended a ``stop`` finish, when the endpoint left it out of the text.

    vLLM drops two kinds of stop token from the response text: the end-of-sequence token
    (``stop_reason`` null) and a matched ``stop_token_ids`` entry (an integer ``stop_reason``). It
    still samples them like any other token and reports their log-probs. A matched stop *string*
    (a string ``stop_reason``) is part of the text.

    Returns:
        The token id, or ``None`` when nothing is missing or the token is unknown.
    """
    if completion.finish_reason != "stop":
        return None
    reason = completion.stop_reason
    if reason is None:
        stop_id = eos_token_id
    elif isinstance(reason, int) and not isinstance(reason, bool):
        stop_id = reason
    else:
        return None
    if stop_id is None or (response_ids and response_ids[-1] == stop_id):
        return None
    return int(stop_id)


def fill_group(
    group: Sequence[Any],
    completions: Sequence[Completion],
    *,
    prompt_ids: Sequence[int],
    tokenizer: Any,
) -> None:
    """Write completions into a framework's ``Sample`` objects (slime or miles; same fields).

    Sets ``tokens`` (prompt + response tokens), ``response``, ``response_length`` and
    ``status`` (``TRUNCATED`` when the endpoint stopped on length, else ``COMPLETED``). The
    response tokens are the exact sampled ids when the endpoint reports them. Otherwise the text
    is re-tokenized, and the token that ended the response (end-of-sequence or a stop token id) is
    appended. The stopping action then stays in the trained sequence, with its log-prob. Also sets
    ``rollout_log_probs`` (see :func:`align_logprobs`), and ``rollout_routed_experts`` when the
    completions carry routes: int32, ``[len(tokens) - 1, layers, top_k]``, as slime and miles
    replay them.

    Raises:
        RolloutError: The completion count differs from the group's, or a completion's routes do
            not have a row for every token but the last.
    """
    if len(group) != len(completions):
        raise RolloutError(f"{len(completions)} completions for a group of {len(group)}")
    for sample, completion in zip(group, completions, strict=True):
        if completion.token_ids is not None:
            # Exact ids from the endpoint. A trailing stop token is part of the response.
            response_ids = list(completion.token_ids)
        else:
            response_ids = list(tokenizer.encode(completion.text, add_special_tokens=False))
            stop_id = omitted_stop_token(
                completion, response_ids, getattr(tokenizer, "eos_token_id", None)
            )
            if stop_id is not None:
                response_ids.append(stop_id)
        sample.tokens = list(prompt_ids) + response_ids
        sample.response = completion.text
        sample.response_length = len(response_ids)
        status = type(sample).Status
        sample.status = (
            status.TRUNCATED if completion.finish_reason == "length" else status.COMPLETED
        )
        sample.rollout_log_probs = align_logprobs(completion, response_ids, tokenizer)
        if completion.routed_experts is not None:
            if completion.routed_experts.shape[0] != len(sample.tokens) - 1:
                raise RolloutError(
                    f"routed experts have {completion.routed_experts.shape[0]} rows for "
                    f"{len(sample.tokens)} tokens"
                )
            sample.rollout_routed_experts = completion.routed_experts.astype(np.int32)


def check_routes_fit(groups: Sequence[Sequence[Any]], num_layers: int, top_k: int) -> None:
    """Fail unless every sample's routes have the trainer's layer count and top-k.

    slime checks this itself; miles' replay would otherwise fail deep inside its router.

    Raises:
        RolloutError: a sample's ``rollout_routed_experts`` is not ``[rows, num_layers, top_k]``.
    """
    for group in groups:
        for sample in group:
            routes = getattr(sample, "rollout_routed_experts", None)
            if routes is not None and tuple(routes.shape[1:]) != (num_layers, top_k):
                raise RolloutError(
                    f"routed experts have {routes.shape[1]} layers and top-{routes.shape[2]}, but "
                    f"the trainer has --num-layers {num_layers} and --moe-router-topk {top_k}"
                )


async def sample_groups(
    sampler: CompletionSampler,
    groups: Sequence[Sequence[Any]],
    *,
    tokenizer: Any,
    max_tokens: int,
    temperature: float,
    top_p: float = 1.0,
    stop: Sequence[str] | None = None,
    cache_salt: str | None = None,
) -> dict[str, float]:
    """Sample every group concurrently and fill its samples; returns timing/shape metrics.

    Every request carries ``cache_salt``, such as :func:`rollout_cache_salt` of the rollout.

    With log-probs required, a group whose log-probs do not align is sampled again, up to
    :data:`LOGPROB_RESAMPLES` times.

    Raises:
        RolloutError: The config requires log-probs, and a group's still do not align.
    """
    started = time.monotonic()
    resampled = 0

    async def one(group: Sequence[Any]) -> None:
        nonlocal resampled
        prompt = group[0].prompt
        if not isinstance(prompt, str):
            raise RolloutError(
                "prompts must arrive rendered as strings; enable --apply-chat-template"
            )
        prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
        for attempt in range(LOGPROB_RESAMPLES + 1):
            completions, still = await sample_prompt(
                sampler,
                prompt_ids,
                len(group),
                text=lambda: prompt,
                token_id_prompts=sampler.token_id_prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                cache_salt=cache_salt,
            )
            # Groups finish in any order. After one group falls back to text, none switches back.
            sampler.token_id_prompts = sampler.token_id_prompts and still
            fill_group(group, completions, prompt_ids=prompt_ids, tokenizer=tokenizer)
            aligned = all(getattr(s, "rollout_log_probs", None) is not None for s in group)
            if aligned or not sampler.config.require_logprobs or attempt == LOGPROB_RESAMPLES:
                return
            resampled += 1
            log.warning(
                "sampling a group again: its log-probs do not line up with its tokens (try %d/%d)",
                attempt + 1,
                LOGPROB_RESAMPLES,
            )

    await asyncio.gather(*(one(g) for g in groups))
    flat = [s for g in groups for s in g]
    truncated = sum(1 for s in flat if s.status is type(s).Status.TRUNCATED)
    aligned = sum(1 for s in flat if getattr(s, "rollout_log_probs", None) is not None)
    # The trainers tensorize rollout_log_probs for the whole batch, so keep all or none. If any
    # sample's log-probs do not align, drop them everywhere and let the trainer recompute them.
    kept = aligned == len(flat)
    if not kept and sampler.config.require_logprobs:
        raise RolloutError(
            f"{len(flat) - aligned}/{len(flat)} samples came back without log-probs that line up "
            "with their tokens, and the trainer reads them from every sample (as with --use-tis). "
            f"With {ENV_PREFIX}TOKEN_IDS on, an endpoint that reports the sampled token ids lines "
            "them up exactly"
        )
    if not kept:
        for sample in flat:
            sample.rollout_log_probs = None
        if aligned:
            log.warning(
                "dropping rollout log-probs for this batch: %d/%d samples did not align with the "
                "re-tokenized response",
                len(flat) - aligned,
                len(flat),
            )
    return {
        "coreweave/generation_seconds": round(time.monotonic() - started, 3),
        "coreweave/samples": float(len(flat)),
        "coreweave/mean_response_length": sum(s.response_length for s in flat) / max(1, len(flat)),
        "coreweave/truncated_fraction": truncated / max(1, len(flat)),
        "coreweave/logprobs_aligned_fraction": aligned / max(1, len(flat)),
        "coreweave/logprobs_kept": 1.0 if kept else 0.0,
        "coreweave/resampled_groups": float(resampled),
    }


__all__ = [
    "Completion",
    "CompletionSampler",
    "GENERATE_FUNCTION_FLAG",
    "LOGPROB_READERS",
    "LOGPROB_RESAMPLES",
    "RolloutConfig",
    "RolloutError",
    "align_logprobs",
    "as_cache_salt",
    "check_stop_limits",
    "decode_routed_experts",
    "fill_group",
    "omitted_stop_token",
    "refuse_generate_function",
    "request_seed",
    "rollout_cache_salt",
    "same_token",
    "same_tokens",
    "sample_groups",
    "sample_prompt",
    "shared_sampler",
]
