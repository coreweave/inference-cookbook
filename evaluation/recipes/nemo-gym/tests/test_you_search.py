# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

import asyncio
import importlib.util
import sys
from pathlib import Path

import httpx
import pytest



def load():
    spec = importlib.util.spec_from_file_location(
        "you_search", Path(__file__).resolve().parents[1] / "you_search.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


you_search = load()


def client_for(handler, **kwargs):
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(base_url="https://you.invalid/v1", transport=transport)
    return you_search.YouSearchClient("key", client=http, **kwargs)


SEARCH = {
    "results": {
        "web": [
            {
                "url": "https://example.com/a",
                "title": "A",
                "description": "about a",
                "snippets": ["first", "second"],
                "contents": {"markdown": "# A page"},
            },
            {"url": "https://datasets.huggingface.co/leak", "title": "leak", "snippets": ["x"]},
            {"url": "https://example.org/b", "title": "B", "description": "about b"},
        ]
    }
}


def test_search_returns_tavily_shaped_results_without_excluded_domains():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=SEARCH)

    result = asyncio.run(
        client_for(handler).search(
            "q", max_results=3, exclude_domains=["huggingface.co"], search_depth="advanced"
        )
    )
    assert [r["url"] for r in result["results"]] == [
        "https://example.com/a",
        "https://example.org/b",
    ]
    assert result["results"][0] == {
        "title": "A",
        "url": "https://example.com/a",
        "content": "first\nsecond",
        "raw_content": "# A page",
    }
    assert result["results"][1]["content"] == "about b"
    assert result["results"][1]["raw_content"] is None
    params = seen[0].url.params
    assert params["query"] == "q" and params["count"] == "3" and params["livecrawl"] == "web"


def test_extract_splits_fetched_and_failed_pages():
    def handler(request):
        assert request.url.path == "/v1/contents"
        return httpx.Response(
            200,
            json=[
                {"url": "https://example.com/a", "markdown": "text"},
                {"url": "https://example.com/b", "markdown": ""},
            ],
        )

    result = asyncio.run(client_for(handler).extract(urls=["a", "b"], query="hint"))
    assert result["results"] == [{"url": "https://example.com/a", "raw_content": "text"}]
    assert [r["url"] for r in result["failed_results"]] == ["https://example.com/b"]


def test_retries_rate_limits_but_not_bad_requests(monkeypatch):
    async def no_sleep(_):
        pass

    monkeypatch.setattr(you_search.asyncio, "sleep", no_sleep)
    statuses = [429, 503, 200]

    def flaky(request):
        return httpx.Response(statuses.pop(0), json=SEARCH)

    assert asyncio.run(client_for(flaky).search("q"))["results"]
    calls = []

    def bad(request):
        calls.append(request)
        return httpx.Response(400, json={"error": "bad"})

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(client_for(bad).search("q"))
    assert len(calls) == 1


def test_rejects_an_empty_key():
    with pytest.raises(ValueError, match="nonempty"):
        you_search.YouSearchClient(" ")


def test_install_copies_a_self_contained_entrypoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        you_search.install(tmp_path)
    harness = tmp_path / you_search.HARNESS
    harness.mkdir(parents=True)
    (harness / "app.py").write_text("")
    target = you_search.install(tmp_path)
    assert target.name == "you_search.py"
    source = target.read_text()
    # The harness venv has no cwtrain, so the copy must not import it.
    assert "import cwtrain" not in source and "from cwtrain" not in source
    assert "run_server()" in source
