# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""You.com search and page contents for Gym's BrowseComp harness.

Gym's ``browsecomp_advanced_harness`` resources server calls a Tavily client's ``search`` and
``extract``. ``YouSearchClient`` has the same two methods and returns Tavily-shaped results from
the You.com Search and Contents APIs. So the harness's tools, page workspace and judge prompt run
unchanged, in Gym v0.4.0 (bundled with NeMo-RL v0.7.0) and v0.7.0.

Gym v0.4.0 has no contamination filter; v0.7.0 filters BrowseComp leaks. This shared adapter also
drops ASearcher and legacy SimpleQA answer keys before shortening search or page text
(``contaminated``), and backports BrowseComp filtering to training. The judge sees only the final ``Exact Answer:`` line
(``judged_answer``), not the explanation or earlier turns.

``install`` copies this file into the harness directory. A server whose ``entrypoint`` is
``you_search.py`` swaps its Tavily clients for You.com clients, one per key in its
``tavily_api_key`` setting. ``launch.py start`` installs it for the evaluation. For training, the case study ships it in the
NeMo-RL kit, and grpo_search_agent.yaml's ``coreweave_files`` copies it the same way. The copy runs
in the harness's own venv, which has neither cwtrain nor the case study's files, so this module imports only
the standard library and httpx, which the harness already uses.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

BASE_URL = "https://ydc-index.io/v1"
HARNESS = Path("resources_servers") / "browsecomp_advanced_harness"
ENTRYPOINT = "you_search.py"
# Characters kept of each result's page text and snippets; unset or 0 keeps all of it. A smaller
# cap fits more tool calls into a trajectory's tokens.
MAX_CHARS_ENV = "YOU_SEARCH_MAX_CHARS"
# Retry rate limits and server errors; a 4xx other than 429 is the request's fault.
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
# A request You.com rejects for its content, such as a query it finds too long (422). The agent
# wrote it, so the harness shows the agent "Search failed" instead of failing the rollout. Any other
# error, such as a bad key (401), still fails the rollout: it is not the agent's fault.
REJECTED_STATUS = frozenset({400, 422})
# Characters of You.com's error that the agent sees.
REJECTION_CHARS = 300

try:
    # The error the harness catches from its Tavily client and shows the agent.
    from tavily.errors import BadRequestError
except ImportError:  # outside the harness venv, such as in the case study's tests

    class BadRequestError(Exception):
        """Stands in for ``tavily.errors.BadRequestError``."""

# A result that quotes a benchmark (a dataset mirror, a leaderboard, the simple-evals repo or its
# CSVs) hands the agent the answer key. These are Gym v0.7.0's CONTAMINATION_PATTERNS, plus
# ASearcher's names for the training data and SimpleQA's for legacy answer keys.
# Case-insensitive, matched anywhere in the result, its URL included.
CONTAMINATION_PATTERNS = (
    "browsecomp",
    "browse_comp",
    "browse-comp",
    "simple-eval",
    "bcplus",
    "bc-plus",
    "bc_plus",
    "asearcher",
    "a_searcher",
    "a-searcher",
    "simpleqa",
    "simple_qa",
    "simple-qa",
)
# Matched against the result's URL only. Dataset viewers serve mirrors whose names match no
# pattern; a page that only links to one is kept.
CONTAMINATED_URL_SUBSTRINGS = ("huggingface.co/datasets", "datasets-server.huggingface.co")
# Characters of the final answer that the judge sees.
JUDGE_ANSWER_CHARS = 200
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_EXACT_ANSWER = re.compile(r"exact answer\W*:\s*(.*)", re.IGNORECASE)
_CONFIDENCE = re.compile(r"\bconfidence\s*:", re.IGNORECASE)
# The judge prompt's own fields. In Gym v0.4.0 and v0.6.0 [correct_answer] follows the
# response, so a response could otherwise supply one.
_JUDGE_FIELDS = re.compile(
    r"\[(?:question|response|correct_answer)\]\s*:?"
    r"|\b(?:correct_answer|extracted_final_answer|answer_type)\s*:"
    r"|\bcorrect\s*:\s*(?:yes|no)\b",
    re.IGNORECASE,
)


def excluded(url: str, domains: Iterable[str]) -> bool:
    """Whether ``url``'s host is one of ``domains`` or a subdomain of one."""
    host = (urlparse(url).hostname or "").lower()
    return any(host == d.lower() or host.endswith("." + d.lower()) for d in domains)


def contaminated(item: Mapping[str, Any]) -> bool:
    """Whether a provider result quotes a benchmark or comes from a dataset viewer.

    The whole result is checked, not chosen fields, so page text, snippets and title all count.
    """
    text = json.dumps(item, default=str).lower()
    url = str(item.get("url") or "").lower()
    return any(p in text for p in CONTAMINATION_PATTERNS) or any(
        s in url for s in CONTAMINATED_URL_SUBSTRINGS
    )


def _report_drops(function: str, dropped: int, kept: int) -> None:
    # The same line as Gym v0.7.0's filter, so one grep counts drops in either version.
    if dropped:
        print(
            f"[browsecomp][contamination] fn={function} provider=you dropped={dropped} kept={kept}",
            flush=True,
        )


def judged_answer(response: str) -> str:
    """The part of a final response that the judge grades: its last ``Exact Answer:``, capped.

    The judge never sees the explanation or earlier turns (Gym v0.4.0 passes the text of every
    turn), so text that argues for an answer, lists other candidates or imitates the judge's
    fields does not reach it. A response with no ``Exact Answer:`` is graded as no answer.
    """
    matches = _EXACT_ANSWER.findall(_THINK.sub("", response or ""))
    if not matches:
        return ""
    answer = _CONFIDENCE.split(matches[-1])[0]
    answer = " ".join(_JUDGE_FIELDS.sub(" ", answer).split()).strip("*_` ")
    return f"Exact Answer: {answer[:JUDGE_ANSWER_CHARS]}" if answer else ""


def max_chars() -> int | None:
    """The per-result character cap from ``YOU_SEARCH_MAX_CHARS``, or ``None`` for no cap."""
    value = os.environ.get(MAX_CHARS_ENV, "").strip()
    return int(value) if value and int(value) > 0 else None


def _cap(text: str | None) -> str | None:
    limit = max_chars()
    return text[:limit] if text and limit else text


def search_results(payload: dict, exclude_domains: Sequence[str] = ()) -> list[dict[str, Any]]:
    """Convert a You.com search response into Tavily's ``results`` list.

    ``content`` holds the snippets, ``raw_content`` the live-crawled page (``None`` when You.com
    did not crawl it). Results on excluded domains and ``contaminated`` results are dropped, the
    latter before the character cap, so a quote past the cap still drops its result.
    """
    results, dropped = [], 0
    for item in (payload.get("results") or {}).get("web") or []:
        url = item.get("url") or ""
        if not url or excluded(url, exclude_domains):
            continue
        if contaminated(item):
            dropped += 1
            continue
        snippets = item.get("snippets") or []
        results.append(
            {
                "title": item.get("title") or "",
                "url": url,
                "content": _cap("\n".join(snippets) or item.get("description") or ""),
                "raw_content": _cap((item.get("contents") or {}).get("markdown") or None),
            }
        )
    _report_drops("search", dropped, len(results))
    return results


def contents_results(payload: Any) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Convert a You.com contents response into Tavily's ``results`` and ``failed_results``.

    ``contaminated`` pages are in neither, so when every page is dropped the harness answers as it
    does for pages it could not fetch, and the agent is not told that the benchmark is nearby.
    """
    items = payload if isinstance(payload, list) else (payload or {}).get("results") or []
    results, failed, dropped = [], [], 0
    for item in items:
        if contaminated(item):
            dropped += 1
            continue
        text = item.get("markdown") or ""
        target = results if text else failed
        target.append({"url": item.get("url") or "", "raw_content": _cap(text)})
    _report_drops("browse", dropped, len(results))
    return results, failed


class YouSearchClient:
    """An async stand-in for ``tavily.AsyncTavilyClient`` backed by You.com.

    Args:
        api_key: The You.com API key, sent as ``X-API-Key``.
        base_url: The You.com API root.
        timeout: Seconds per HTTP request.
        attempts: Tries per call for rate limits, server errors and connection errors.
        client: An ``httpx.AsyncClient`` to use instead of creating one (for tests).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = BASE_URL,
        timeout: float = 60.0,
        attempts: int = 4,
        client: Any = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("You.com search requires a nonempty API key")
        if client is None:
            import httpx

            client = httpx.AsyncClient(
                base_url=base_url.rstrip("/"),
                headers={"X-API-Key": api_key.strip()},
                timeout=timeout,
            )
        self._client = client
        self._attempts = max(1, attempts)

    async def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        import httpx

        for attempt in range(self._attempts):
            last = attempt == self._attempts - 1
            try:
                response = await self._client.request(method, path, **kwargs)
            except httpx.TransportError:
                if last:
                    raise
            else:
                if response.status_code in REJECTED_STATUS:
                    raise BadRequestError(
                        f"You.com rejected the request ({response.status_code}): "
                        f"{response.text[:REJECTION_CHARS]}"
                    )
                if response.status_code not in RETRY_STATUS or last:
                    response.raise_for_status()
                    return response.json()
            await asyncio.sleep(0.5 * 2**attempt)
        raise AssertionError("unreachable")

    async def search(
        self,
        query: str,
        max_results: int = 5,
        exclude_domains: Sequence[str] | None = None,
        include_raw_content: bool = True,
        **_: Any,
    ) -> dict[str, Any]:
        """Search the web. Tavily-only options such as ``search_depth`` are ignored."""
        params: dict[str, Any] = {"query": query, "count": max_results}
        if include_raw_content:
            params.update(livecrawl="web", livecrawl_formats="markdown")
        payload = await self._call("GET", "/search", params=params)
        return {"query": query, "results": search_results(payload, exclude_domains or ())}

    async def extract(self, urls: Sequence[str], **_: Any) -> dict[str, Any]:
        """Fetch the pages' markdown. Tavily's ``query`` hint is ignored."""
        payload = await self._call(
            "POST", "/contents", json={"urls": list(urls), "formats": ["markdown"]}
        )
        results, failed = contents_results(payload)
        return {"results": results, "failed_results": failed}


def install(gym_root: Path) -> Path:
    """Copy this module into Gym's BrowseComp harness as its You.com entrypoint.

    Raises:
        FileNotFoundError: ``gym_root`` has no BrowseComp harness.
    """
    harness = Path(gym_root) / HARNESS
    if not (harness / "app.py").is_file():
        raise FileNotFoundError(f"no BrowseComp harness at {harness}")
    target = harness / ENTRYPOINT
    target.write_text(Path(__file__).read_text())
    return target


def server_class(harness: Any) -> type:
    """The harness's resources server with You.com clients and a judge that sees only the answer.

    Args:
        harness: The harness's ``app`` module.
    """

    class YouSearchResourcesServer(harness.TavilySearchResourcesServer):
        def model_post_init(self, context: Any) -> None:
            super().model_post_init(context)
            keys = self.config.tavily_api_key
            keys = [keys] if isinstance(keys, str) else list(keys or [])
            self._async_tavily_clients = [YouSearchClient(key) for key in keys]
            print(f"Search provider: you.com ({len(keys)} key(s))", flush=True)

        async def _verify_answer_with_judge(
            self, question: str, ground_truth: str, response: str
        ) -> Any:
            # Same signature in Gym v0.4.0 to v0.7.0; the return value passes through unchanged.
            return await super()._verify_answer_with_judge(
                question, ground_truth, judged_answer(response)
            )

    return YouSearchResourcesServer


def run_server() -> None:
    """Start the BrowseComp harness as ``server_class`` changes it."""
    import nemo_gym  # noqa: F401 - adds the Gym install root to sys.path
    from resources_servers.browsecomp_advanced_harness import app as harness

    server_class(harness).run_webserver()


if __name__ == "__main__":
    run_server()
