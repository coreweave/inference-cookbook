# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""You.com search and page contents for Gym's BrowseComp harness.

Gym's ``browsecomp_advanced_harness`` resources server calls a Tavily client's ``search`` and
``extract``. ``YouSearchClient`` has the same two methods and returns Tavily-shaped results from
the You.com Search and Contents APIs. So the harness's tools, page workspace and judge run
unchanged, in Gym v0.4.0 (bundled with NeMo-RL v0.7.0) and v0.6.0.

``install`` copies this file into the harness directory. A server whose ``entrypoint`` is
``you_search.py`` swaps its Tavily clients for You.com clients, one per key in its
``tavily_api_key`` setting. ``launch.py start`` installs it for the evaluation; the NeMo-RL recipe
ships it in its kit, and cwtrain's BrowseComp training preset copies it the same way. The copy runs
in the harness's own venv, which has neither cwtrain nor this recipe, so this module imports only
the standard library and httpx, which the harness already uses.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterable, Sequence
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


def excluded(url: str, domains: Iterable[str]) -> bool:
    """Whether ``url``'s host is one of ``domains`` or a subdomain of one."""
    host = (urlparse(url).hostname or "").lower()
    return any(host == d.lower() or host.endswith("." + d.lower()) for d in domains)


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
    did not crawl it). Results on excluded domains are dropped.
    """
    results = []
    for item in (payload.get("results") or {}).get("web") or []:
        url = item.get("url") or ""
        if not url or excluded(url, exclude_domains):
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
    return results


def contents_results(payload: Any) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Convert a You.com contents response into Tavily's ``results`` and ``failed_results``."""
    items = payload if isinstance(payload, list) else (payload or {}).get("results") or []
    results, failed = [], []
    for item in items:
        text = item.get("markdown") or ""
        target = results if text else failed
        target.append({"url": item.get("url") or "", "raw_content": _cap(text)})
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


def run_server() -> None:
    """Start the BrowseComp harness with You.com clients in place of its Tavily clients."""
    import nemo_gym  # noqa: F401 - adds the Gym install root to sys.path
    from resources_servers.browsecomp_advanced_harness import app as harness

    class YouSearchResourcesServer(harness.TavilySearchResourcesServer):
        def model_post_init(self, context: Any) -> None:
            super().model_post_init(context)
            keys = self.config.tavily_api_key
            keys = [keys] if isinstance(keys, str) else list(keys or [])
            self._async_tavily_clients = [YouSearchClient(key) for key in keys]
            print(f"Search provider: you.com ({len(keys)} key(s))", flush=True)

    YouSearchResourcesServer.run_webserver()


if __name__ == "__main__":
    run_server()
