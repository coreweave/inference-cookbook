# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""URL-policy and mocked provider tests; no API keys, downloads or paid calls."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

spec = importlib.util.spec_from_file_location(
    "tool_access_adapter", Path(__file__).resolve().parents[1] / "eval/you_search.py"
)
adapter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = adapter
spec.loader.exec_module(adapter)

DOMAINS = ("huggingface.co", "openaipublic.blob.core.windows.net")
BLOCKED = [
    "file:///secrets/gym/search_api_key",
    "file:///work/hotload/nemo-gym/search.jsonl",
    "/benchmarks/data.jsonl",
    "../data.jsonl",
    "//huggingface.co/datasets/a/b",
    "data:text/plain,answer",
    "javascript:fetch('file:///secrets/gym/search_api_key')",
    "ftp://archive.example.org/key.csv",
    "gopher://example.org/1/",
    "https://huggingface.co./api/datasets/a/b/parquet",
    "https://HUGGINGFACE.CO./datasets/a/b",
    "https://huggingface。co/datasets/a/b",
    "https://%68uggingface.co/datasets/a/b",
    "https://huggingface.co%2e/datasets/a/b",
    "https://huggingface.co@research.example.org/key",
    "https://user:password@research.example.org/paper",
    "https://research.example.org\\@huggingface.co/key",
    "https://research.example.org/%5c%5chuggingface.co/datasets/a/b",
    "https://research.example.org/\nkey",
    "https://research.example.org/%0d%0akey",
    "https://research.example.org/%xy",
    "https://research.example.org/" + "%25" * 8192,
    "https://huggingface.co:notaport/datasets/a/b",
    "https://research.example.org:8080/paper",
    "http://127.0.0.1/key",
    "http://127.1/key",
    "http://2130706433/key",
    "http://0x7f000001/key",
    "http://127.0.0.0x1/key",
    "http://[::1]/key",
    "https://localhost/key",
    "https://node.internal/key",
    "https://hf.co/datasets/a/b",
    "https://redirect.example.org/?url=hf.co/api/datasets/a/b/parquet",
    "https://redirect.example.org//hf.co/datasets/a/b?next=//hf.co/datasets/a/b",
    "https://r.jina.ai/file:///work/hotload/nemo-gym/search.jsonl",
    "https://ｈｕｇｇｉｎｇｆａｃｅ.co./api/datasets/a/b",
    "https://cdn-lfs.hf.co/datasets/a/b/key.parquet",
    "https://datasets-server.huggingface.co/rows?dataset=a/b",
    "https://hf-mirror.com/api/datasets/a/b/parquet",
    "https://hf-mirror.com/a/../%64atasets/a/b/resolve/main/key.parquet",
    "https://r.jina.ai/https://huggingface.co/api/datasets/a/b/parquet",
    "https://web.archive.org/web/20200101000000/https://huggingface.co/datasets/a/b",
    "https://redirect.example.org/?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fa%2Fb",
    "https://redirect.example.org/?next=%252F%252Fhuggingface.co%252Fdatasets%252Fa%252Fb",
    "https://redirect.example.org/?url=file%3A%2F%2F%2Fsecrets%2Fgym%2Fsearch_api_key",
    "https://api.github.com/repos/openai/%2573imple-evals/contents/data.csv",
    "https://openaipublic.blob.core.windows.net./anything.csv",
]
ALLOWED = [
    "https://arxiv.org/pdf/2601.12345",
    "https://pubmed.ncbi.nlm.nih.gov/123456/",
    "https://research.example.org/results.csv",
    "https://research.example.org/data.json",
    "https://api.crossref.org/works?query=protein",
    "https://raw.githubusercontent.com/science/project/main/data.json",
    "https://github.com/science/project/blob/main/results.csv",
    "https://huggingface.co.evil.example.org/article",
    "https://research.example.org/?q=author%3ASmith",
    "https://www.kaggle.com/datasets/science/climate",
    "https://web.archive.org/web/20200101000000/https://arxiv.org/pdf/2601.12345",
    "https://redirect.example.org/?url=https%3A%2F%2Farxiv.org%2Fpdf%2F2601.12345",
]


@pytest.mark.parametrize("url", BLOCKED)
def test_blocked_destinations(url):
    assert adapter.allowed_url(url, DOMAINS) is None


@pytest.mark.parametrize("url", ALLOWED)
def test_research_destinations_stay_available(url):
    assert adapter.allowed_url(url, DOMAINS) is not None


def test_canonicalization_and_policy_do_not_confuse_authority_with_path():
    assert (
        adapter.allowed_url("HTTPS://ExAmPlE.org.:443/paper%20name.pdf#section")
        == "https://example.org/paper%20name.pdf"
    )
    assert (
        adapter.allowed_url("https://bücher.example/paper") == "https://xn--bcher-kva.example/paper"
    )
    assert adapter.excluded("https://archive.HUGGINGFACE.CO./page", ["HuggingFace.Co."])
    assert not adapter.excluded("https://arxiv.org/paper", ["huggingface.co"])


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/api/datasets/a/b/parquet",
        "https://hf.co/api/datasets/a/b/tree/main",
        "https://hf-mirror.com/api/resolve-cache/datasets/a/b/data.json",
        "https://datasets-server.hf.co/rows",
    ],
)
def test_dataset_api_transport_is_blocked_even_without_an_exclude_list(url):
    assert adapter.allowed_url(url) is None


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/blog/research",
        "https://hf-mirror.com/docs/api",
        "https://example.org/datasets/paper.csv",
    ],
)
def test_route_policy_is_not_a_generic_dataset_or_file_extension_ban(url):
    assert adapter.allowed_url(url) is not None


def make_client(handler, **kwargs):
    # Deliberately enable redirects on the test client; the adapter must override them for API keys.
    http = httpx.AsyncClient(
        base_url="https://provider.example.org/v1",
        transport=httpx.MockTransport(handler),
        headers={"X-API-Key": "server-only-test-key"},
        follow_redirects=True,
    )
    return adapter.YouSearchClient(
        "server-only-test-key", client=http, exclude_domains=DOMAINS, **kwargs
    )


@pytest.mark.parametrize("url", BLOCKED)
def test_blocked_browse_never_calls_provider(url):
    def never(request):
        pytest.fail("blocked destination reached the provider")

    assert asyncio.run(make_client(never).extract([url])) == {
        "results": [],
        "failed_results": [],
    }


def test_browse_mixed_batch_validates_requests_and_provider_results_before_cap(
    monkeypatch,
):
    monkeypatch.setenv("YOU_SEARCH_MAX_CHARS", "4")
    calls = []

    def provider(request):
        calls.append(request)
        assert json.loads(request.content) == {
            "urls": ["https://arxiv.org/pdf/2601.12345"],
            "formats": ["markdown"],
        }
        return httpx.Response(
            200,
            json=[
                {
                    "url": "https://arxiv.org/pdf/2601.12345",
                    "markdown": "research content",
                },
                {"url": "https://huggingface.co./api/datasets/a/b", "markdown": "leak"},
                {
                    "url": "https://openaipublic.blob.core.windows.net./answers.csv",
                    "markdown": "",
                },
            ],
        )

    result = asyncio.run(
        make_client(provider).extract(
            [BLOCKED[0], "HTTPS://ARXIV.ORG.:443/pdf/2601.12345", BLOCKED[7]]
        )
    )
    assert result == {
        "results": [{"url": "https://arxiv.org/pdf/2601.12345", "raw_content": "rese"}],
        "failed_results": [],
    }
    assert len(calls) == 1


@pytest.mark.parametrize(
    "field",
    [
        "final_url",
        "source_url",
        "resolved_url",
        "requested_url",
        "original_url",
        "redirect_url",
    ],
)
@pytest.mark.parametrize("operation", ["search", "browse"])
def test_reported_source_destination_cannot_launder_content(field, operation):
    item = {
        "url": "https://research.example.org/paper",
        "markdown": "leak",
        "contents": {"markdown": "leak"},
        field: "https://hf.co/api/datasets/a/b",
    }

    def provider(request):
        return httpx.Response(
            200, json={"results": {"web": [item]}} if operation == "search" else [item]
        )

    client = make_client(provider)
    result = asyncio.run(
        client.search("ordinary research")
        if operation == "search"
        else client.extract([item["url"]])
    )
    assert result["results"] == []
    assert not result.get("failed_results")


@pytest.mark.parametrize(
    "location",
    [
        "https://huggingface.co./api/datasets/a/b",
        "//hf.co/datasets/a/b",
        "file:///secrets/gym/key",
    ],
)
def test_visible_redirect_hops_are_checked(location):
    item = {
        "url": "https://research.example.org/paper",
        "markdown": "leak",
        "redirects": [{"location": location}],
    }
    assert adapter.contents_results([item], DOMAINS) == ([], [])


def test_search_exclusions_are_persistent_across_calls_and_cover_browse():
    def provider(request):
        item = {
            "url": "https://huggingface.co./api/datasets/a/b",
            "snippets": ["answer"],
        }
        return httpx.Response(200, json={"results": {"web": [item]}})

    client = make_client(provider)
    assert asyncio.run(client.search("q", exclude_domains=[]))["results"] == []
    assert asyncio.run(client.search("q", exclude_domains=["example.org"]))["results"] == []
    assert asyncio.run(client.extract(["https://hf.co/blog/paper"])) == {
        "results": [],
        "failed_results": [],
    }


def test_api_redirects_do_not_forward_server_key_or_call_other_host():
    calls = []

    def provider(request):
        calls.append(request)
        return httpx.Response(307, headers={"Location": "https://other.example.org/steal"})

    with pytest.raises(adapter.SearchProviderError, match="Search evidence is unavailable"):
        asyncio.run(make_client(provider).search("q"))
    assert len(calls) == 1
    assert calls[0].url.host == "provider.example.org"
    assert calls[0].headers["X-API-Key"] == "server-only-test-key"


def test_tool_kwargs_cannot_choose_endpoint_method_headers_or_local_files():
    calls = []

    def provider(request):
        calls.append(request)
        if request.url.path == "/v1/search":
            return httpx.Response(
                200,
                json={
                    "results": {
                        "web": [
                            {
                                "url": "https://arxiv.org/pdf/2601.12345",
                                "snippets": ["research"],
                            }
                        ]
                    }
                },
            )
        return httpx.Response(
            200,
            json=[{"url": "https://arxiv.org/pdf/2601.12345", "markdown": "research"}],
        )

    client = make_client(provider)
    first = asyncio.run(
        client.search(
            "q",
            base_url="file:///secrets/gym/key",
            path="/arbitrary",
            method="POST",
            headers={"secret": "policy"},
        )
    )
    second = asyncio.run(
        client.extract(
            [first["results"][0]["url"]],
            base_url="https://other.example.org",
            path="/arbitrary",
            headers={"secret": "policy"},
        )
    )
    assert second["results"][0]["raw_content"] == "research"
    assert [(r.method, r.url.path) for r in calls] == [
        ("GET", "/v1/search"),
        ("POST", "/v1/contents"),
    ]
    assert all(r.url.host == "provider.example.org" and "secret" not in r.headers for r in calls)


def test_clean_page_links_and_scientific_formats_are_not_treated_as_sources():
    item = {
        "url": "https://research.example.org/data.json",
        "markdown": "Methods: see https://huggingface.co/datasets/science/climate",
        "metadata": {"favicon_url": "https://huggingface.co/logo.png"},
    }
    assert adapter.contents_results([item], DOMAINS)[0][0]["raw_content"] == item["markdown"]


def test_server_binds_exclusions_to_every_rotating_client(monkeypatch):
    made = []

    class Client:
        def __init__(self, key, **kwargs):
            made.append((key, kwargs))

    class Harness:
        class TavilySearchResourcesServer:
            JUDGE_PROMPT_TEMPLATE = (
                "Apply these rules IN ORDER. The FIRST rule that applies decides the verdict."
                "\nR1. fixture"
            )

            def model_post_init(self, context):
                self._exclude_domains = list(DOMAINS)

    monkeypatch.setattr(adapter, "YouSearchClient", Client)
    cls = adapter.server_class(Harness)
    server = cls()
    server.config = SimpleNamespace(
        tavily_api_key=["first", "second"], workspace="none", search_provider="tavily"
    )
    server.model_post_init(None)
    assert [key for key, _ in made] == ["first", "second"]
    assert all(kwargs["exclude_domains"] == list(DOMAINS) for _, kwargs in made)
    assert all(callable(kwargs["abort_on_auth"]) for _, kwargs in made)
    server.config.workspace = "per_session"
    with pytest.raises(ValueError, match="without a shell"):
        server.model_post_init(None)
    server.config.workspace = "none"
    server.config.search_provider = "exa"
    with pytest.raises(ValueError, match="adapted Tavily"):
        server.model_post_init(None)


@pytest.fixture
def actual_harness_server():
    from importlib.metadata import version

    pytest.importorskip("nemo_gym")
    pytest.importorskip("tavily")
    pytest.importorskip("pocketshell")
    if version("nemo-gym") != "0.7.0":
        pytest.skip("requires the recipe's pinned Gym v0.7.0")
    from nemo_gym.server_utils import ServerClient
    from omegaconf import OmegaConf
    from resources_servers.browsecomp_advanced_harness import app as harness

    root = Path(__file__).resolve().parents[1]
    config = harness.TavilySearchResourcesServerConfig(
        name="tool_access_test",
        host="127.0.0.1",
        port=12345,
        entrypoint="you_search.py",
        tavily_api_key="server-only-test-key",
        workspace="none",
        exclude_domains_file_path=str(root / "eval/exclude_domains.json"),
        judge_responses_create_params={"input": []},
    )
    server = adapter.server_class(harness)(
        config=config,
        server_client=ServerClient(
            head_server_config={"host": "127.0.0.1", "port": 12346},
            global_config_dict=OmegaConf.create({}),
        ),
    )
    return server, harness, SimpleNamespace(session={"session_id": "tool-access-fixture"})


def test_actual_harness_search_and_browse_enforce_the_shared_boundary(actual_harness_server):
    server, harness, request = actual_harness_server
    calls = []
    allowed = "https://arxiv.org/pdf/2601.12345"

    def provider(request):
        calls.append(request)
        if request.method == "GET":
            items = [
                {
                    "url": allowed,
                    "snippets": ["legitimate research"],
                    "contents": {"markdown": "legitimate research"},
                },
                {"url": "https://huggingface.co./api/datasets/a/b", "snippets": ["key material"]},
                {
                    "url": "https://mirror.example.org/paper",
                    "source_url": "https://hf.co/datasets/a/b",
                    "snippets": ["key material"],
                },
            ]
            return httpx.Response(200, json={"results": {"web": items}})
        assert json.loads(request.content)["urls"] == [allowed]
        return httpx.Response(
            200,
            json=[
                {"url": allowed, "markdown": "legitimate research"},
                {
                    "url": "https://openaipublic.blob.core.windows.net./key.csv",
                    "markdown": "key material",
                },
            ],
        )

    async def run():
        # Retain the actual constructor's exclusion binding; replace only its HTTP transport.
        client = server._async_tavily_clients[0]
        assert client._exclude_domains == DOMAINS
        await client._client.aclose()
        async with httpx.AsyncClient(
            base_url="https://provider.example.org/v1", transport=httpx.MockTransport(provider)
        ) as http:
            client._client = http
            searched = await server.search(
                request, harness.TavilySearchRequest(queries=["research"])
            )
            browsed = await server.browse(
                request,
                harness.BrowseRequest(
                    urls=[
                        "file:///secrets/gym/search_api_key",
                        "https://huggingface.co./api/datasets/a/b",
                        allowed,
                    ]
                ),
            )
            for response in [searched, browsed]:
                assert "legitimate research" in response.results_string
                assert "key material" not in response.results_string
                assert "server-only-test-key" not in response.results_string
            before = len(calls)
            blocked = await server.browse(
                request,
                harness.BrowseRequest(
                    urls=[
                        "https://openaipublic.blob.core.windows.net./key.csv",
                        "file:///work/hotload/nemo-gym/search.jsonl",
                    ]
                ),
            )
            assert len(calls) == before
            assert "key material" not in blocked.results_string

    asyncio.run(run())
    assert [(call.method, call.url.path) for call in calls] == [
        ("GET", "/v1/search"),
        ("POST", "/v1/contents"),
    ]


def test_actual_harness_cannot_enable_shell_to_read_keys_or_benchmark_files(
    actual_harness_server, monkeypatch
):
    server, harness, request = actual_harness_server

    async def no_process(*args, **kwargs):
        pytest.fail("search-only server attempted to start a process")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", no_process)

    async def run():
        response = await server.bash_command(
            request,
            harness.BashCommandRequest(
                keystrokes="cat /secrets/gym/search_api_key /work/hotload/nemo-gym/search.jsonl"
            ),
        )
        assert "not enabled" in response.results_string
        assert server._session_workspaces == {}
        await server._async_tavily_clients[0]._client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("envelope", ["contents", "metadata"])
def test_nested_reported_source_url_is_checked_without_filtering_favicons(envelope):
    item = {
        "url": "https://research.example.org/paper",
        "markdown": "leak",
        envelope: {"url": "https://hf.co/api/datasets/a/b"},
    }
    assert adapter.contents_results([item], DOMAINS) == ([], [])


def test_redirect_record_checks_both_request_url_and_location():
    item = {
        "url": "https://research.example.org/paper",
        "markdown": "leak",
        "redirects": [
            {
                "url": "https://research.example.org/old",
                "location": "https://hf.co/api/datasets/a/b",
            }
        ],
    }
    assert adapter.contents_results([item], DOMAINS) == ([], [])


def test_visible_allowed_redirect_and_literal_percent_research_paths_are_preserved():
    item = {
        "url": "https://research.example.org/paper",
        "markdown": "research",
        "redirects": [{"url": "https://research.example.org/old", "location": "/paper"}],
    }
    assert adapter.contents_results([item], DOMAINS)[0][0]["raw_content"] == "research"
    assert adapter.allowed_url("https://research.example.org/50%25.pdf") is not None


def test_policy_drop_telemetry_contains_counts_not_urls_or_payloads(capsys):
    def never(request):
        pytest.fail("provider contacted for a blocked URL")

    result = asyncio.run(make_client(never).extract(["file:///secrets/gym/search_api_key"]))
    logged = capsys.readouterr().out
    assert "[tool_access] fn=browse_request provider=you dropped=1" in logged
    assert "secrets" not in logged and "search_api_key" not in logged
    assert result == {"results": [], "failed_results": []}


def test_optional_unavailable_source_metadata_is_not_a_claimed_final_origin():
    item = {
        "url": "https://research.example.org/paper.pdf",
        "markdown": "research",
        "final_url": None,
    }
    assert adapter.contents_results([item], DOMAINS)[0][0]["raw_content"] == "research"


@pytest.mark.parametrize(
    "tool_name",
    [
        "verify",
        "seed_session",
        "aggregate_metrics",
        "bash_command",
        "read_file",
        "get_file",
        "search/../verify",
        "../verify",
        "search?operation=verify",
        "search#fragment",
        "%73earch",
        "search%2f..%2fverify",
        "/other.example.org/path",
        "https://other.example.org/path",
        "search",
        "browse",
    ],
)
def test_real_agent_dispatch_rejects_unapproved_tool_paths_and_preserves_host_calls(
    monkeypatch, tool_name
):
    from importlib.metadata import version
    from unittest.mock import AsyncMock, MagicMock

    pytest.importorskip("nemo_gym")
    if version("nemo-gym") != "0.7.0":
        pytest.skip("requires the recipe's pinned Gym v0.7.0")
    from nemo_gym.server_utils import ServerClient
    from omegaconf import OmegaConf
    from responses_api_agents.browsecomp_agent import app as agent_harness

    config = agent_harness.BrowsecompAgentConfig(
        name="policy_agent",
        host="127.0.0.1",
        port=12345,
        entrypoint="you_search.py",
        resources_server={"type": "resources_servers", "name": "tools"},
        model_server={"type": "responses_api_models", "name": "policy"},
        max_steps=3,
    )
    agent = adapter.agent_class(agent_harness)(
        config=config,
        server_client=ServerClient(
            head_server_config={"host": "127.0.0.1", "port": 12346},
            global_config_dict=OmegaConf.create({}),
        ),
    )
    outputs = [
        [
            {
                "type": "function_call",
                "id": "fc1",
                "call_id": "call1",
                "name": tool_name,
                "arguments": "{}",
            }
        ],
        [
            {
                "type": "message",
                "id": "m1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": "Exact Answer: done", "annotations": []}
                ],
            }
        ],
    ]
    calls = []

    async def trusted_transport(self, server_name, url_path, method, **kwargs):
        calls.append((server_name, url_path, method))
        if server_name == "policy":
            payload = agent_harness.NeMoGymResponse(
                id=f"resp{len(outputs)}",
                created_at=0,
                model="policy",
                object="response",
                output=outputs.pop(0),
                parallel_tool_calls=False,
                tool_choice="auto",
                tools=[],
            ).model_dump()
        else:
            payload = {"results_string": "legitimate research"}
        response = MagicMock()
        response.ok = True
        response.status = 200
        response.cookies = {}
        response.read = AsyncMock(return_value=json.dumps(payload).encode())
        response.content.read = response.read
        return response

    monkeypatch.setattr(ServerClient, "request", trusted_transport)

    async def run():
        request = MagicMock(cookies={})
        response = MagicMock()
        body = agent_harness.NeMoGymResponseCreateParamsNonStreaming(input="research question")
        result = await agent.responses(request, response, body)
        assert result.output[-1].content[0].text == "Exact Answer: done"
        tool_outputs = [o.output for o in result.output if o.type == "function_call_output"]
        resource_calls = [path for server, path, method in calls if server == "tools"]
        if tool_name in ("search", "browse"):
            assert resource_calls == ["/" + tool_name]
            assert "legitimate research" in tool_outputs[0]
        else:
            assert resource_calls == []
            assert "Unsupported tool" in tool_outputs[0]
        assert adapter._POLICY_TOOL_SERVER.get() is None
        # The harness's trusted control-plane calls remain usable after the policy turn.
        await agent.server_client.post(server_name="tools", url_path="/verify", json={})
        assert calls[-1] == ("tools", "/verify", "POST")

    asyncio.run(run())


def test_research_parameters_are_not_treated_as_scheme_less_redirects():
    assert adapter.allowed_url("https://research.example.org/api?target=BRCA.1", DOMAINS)
    assert adapter.allowed_url(
        "https://redirect.example.org/?url=https%3A%2F%2Fresearch.example.org%2Fpaper%2520name.pdf",
        DOMAINS,
    )


def test_all_rejected_destinations_are_filtered_on_both_return_paths():
    for url in BLOCKED:
        item = {"url": url, "snippets": ["raw artifact"], "markdown": "raw artifact"}
        assert adapter.search_results({"results": {"web": [item]}}, DOMAINS) == []
        assert adapter.contents_results([item], DOMAINS) == ([], [])


def test_allowed_research_formats_are_delivered_on_both_return_paths():
    for url in ALLOWED:
        item = {"url": url, "snippets": ["research"], "markdown": "research"}
        assert (
            adapter.search_results({"results": {"web": [item]}}, DOMAINS)[0]["content"]
            == "research"
        )
        assert adapter.contents_results([item], DOMAINS)[0][0]["raw_content"] == "research"


def test_install_requires_both_gates_before_writing_either(tmp_path):
    resources = tmp_path / adapter.HARNESS
    resources.mkdir(parents=True)
    (resources / "app.py").write_text("")
    with pytest.raises(FileNotFoundError):
        adapter.install(tmp_path)
    assert not (resources / adapter.ENTRYPOINT).exists()


def test_unicode_hostname_is_not_rewritten_to_a_different_idna2003_domain():
    assert adapter.allowed_url("https://faß.de/paper") == "https://xn--fa-hia.de/paper"


@pytest.mark.parametrize(
    "domain", ["localhost", "*.example.org", "example.org/path", "https://example.org"]
)
def test_invalid_exclusion_fails_client_construction(domain):
    with pytest.raises(ValueError, match="bare DNS hostnames"):
        adapter.YouSearchClient("test-key", exclude_domains=[domain])


def test_exclusions_are_normalized_and_used_on_browse_admission():
    def unexpected(_):
        pytest.fail("excluded destination reached the provider")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected)) as http:
            client = adapter.YouSearchClient(
                "test-key", client=http, exclude_domains=["BÜCHER.Example."]
            )
            return await client.extract(["https://xn--bcher-kva.example/paper"])

    assert asyncio.run(run()) == {"results": [], "failed_results": []}


def test_browse_accepts_valid_final_url_even_when_it_differs_from_request():
    final = "https://www.example.org/paper/"

    def provider(request):
        assert json.loads(request.content)["urls"] == [
            "http://example.org/paper",
            "https://example.org/missing",
        ]
        return httpx.Response(200, json=[{"url": final, "markdown": "research"}])

    result = asyncio.run(
        make_client(provider).extract(
            [
                "http://example.org/paper",
                "https://example.org/missing",
            ]
        )
    )
    assert result == {"results": [{"url": final, "raw_content": "research"}], "failed_results": []}
