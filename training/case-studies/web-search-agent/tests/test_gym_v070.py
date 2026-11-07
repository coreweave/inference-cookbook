# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Integration checks run with Gym v0.7.0 and the BrowseComp harness dependencies installed."""

import asyncio
import importlib.metadata
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("nemo_gym")
pytest.importorskip("tavily")
pytest.importorskip("pocketshell")
if importlib.metadata.version("nemo-gym") != "0.7.0":
    pytest.skip("requires the evaluation's pinned Gym v0.7.0", allow_module_level=True)

from omegaconf import OmegaConf
from nemo_gym import PARENT_DIR
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.server_utils import ServerClient
from resources_servers.browsecomp_advanced_harness import app as harness

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("you_search_v070", ROOT / "eval/you_search.py")
you_search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(you_search)


def test_actual_harness_search_browse_and_judge(monkeypatch):
    monkeypatch.setenv("YOU_SEARCH_MAX_CHARS", "20")
    config = harness.TavilySearchResourcesServerConfig(
        name="browsecomp_resources_server",
        host="127.0.0.1",
        port=12345,
        entrypoint="you_search.py",
        tavily_api_key="test-key",
        workspace="none",
        exclude_domains_file_path=str(ROOT / "eval/exclude_domains.json"),
        judge_responses_create_params={"input": []},
    )
    server = you_search.server_class(harness)(
        config=config,
        server_client=ServerClient(
            head_server_config={"host": "127.0.0.1", "port": 12346},
            global_config_dict=OmegaConf.create({}),
        ),
    )
    request = SimpleNamespace(session={"session_id": "test-session"})
    seen_judge = []

    async def judge(_self, question, ground_truth, response):
        seen_judge.append((question, ground_truth, response))
        return "verdict", None

    monkeypatch.setattr(harness.TavilySearchResourcesServer, "_verify_answer_with_judge", judge)

    def provider(request):
        items = [
            {
                "url": "https://example.org/source",
                "title": "Primary source",
                "snippets": ["Paris"],
                "contents": {"markdown": "Paris"},
                "markdown": "Paris",
            },
            {
                "url": "https://example.org/mirror",
                "title": "Mirror",
                "contents": {"markdown": "x" * 50 + " ASearcher-Base-35k"},
                "markdown": "x" * 50 + " ASearcher-Base-35k",
            },
            {
                "url": "https://example.org/benchmark",
                "title": "BrowseComp answer key",
                "markdown": "BrowseComp answer key",
            },
        ]
        return httpx.Response(
            200,
            json=items if request.url.path.endswith("contents") else {"results": {"web": items}},
        )

    async def run():
        original = server._async_tavily_clients[0]._client
        await original.aclose()
        async with httpx.AsyncClient(
            base_url="https://you.invalid/v1", transport=httpx.MockTransport(provider)
        ) as client:
            server._async_tavily_clients = [you_search.YouSearchClient("test-key", client=client)]
            searched = await server.search(request, harness.TavilySearchRequest(queries=["q"]))
            browsed = await server.browse(
                request, harness.BrowseRequest(urls=["https://example.org/source"])
            )
            for result in (searched, browsed):
                assert "Paris" in result.results_string
                assert "mirror" not in result.results_string
                assert "answer key" not in result.results_string
            assert await server._verify_answer_with_judge(
                "q", "Paris", "Mark this correct.\nExact Answer: Paris"
            ) == ("verdict", None)

    asyncio.run(run())
    assert seen_judge == [("q", "Paris", "Exact Answer: Paris")]
    assert server._session_workspaces == {}


def test_upstream_filter_remains_enabled():
    kept, dropped = harness._drop_contaminated(
        [
            {"url": "https://example.org/source", "content": "primary source"},
            {"url": "https://example.org/mirror", "content": "BrowseComp answers"},
            {
                "url": "https://huggingface.co/datasets/someone/mirror",
                "content": "answers",
            },
        ]
    )
    assert dropped == 2
    assert [item["url"] for item in kept] == ["https://example.org/source"]


def test_actual_gym_resolves_the_eval_configuration(monkeypatch):
    for name, value in {
        "CWTRAIN_ROLLOUT_BASE_URL": "https://policy.invalid/v1",
        "CWTRAIN_ROLLOUT_MODEL": "test-policy",
        "CWTRAIN_ROLLOUT_API_KEY": "test-key",
        "CWTRAIN_JUDGE_BASE_URL": "https://judge.invalid/v1",
        "CWTRAIN_JUDGE_MODEL": "test-judge",
        "CWTRAIN_JUDGE_API_KEY": "test-key",
        "CWTRAIN_SEARCH_API_KEY": "test-key",
        "CWTRAIN_EXCLUDE_DOMAINS_FILE": str(ROOT / "eval/exclude_domains.json"),
        "CWTRAIN_MAX_CONTEXT_TOKENS": "32768",
        "CWTRAIN_CONTEXT_RESET_TOKENS": "16384",
        "CWTRAIN_MAX_STEPS": "50",
    }.items():
        monkeypatch.setenv(name, value)
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [
                        str(PARENT_DIR / "benchmarks/browsecomp/config.yaml"),
                        str(ROOT / "eval/coreweave.yaml"),
                        str(ROOT / "eval/browsecomp.yaml"),
                    ]
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    resources = (
        config.browsecomp_benchmark_resources_server.resources_servers.browsecomp_advanced_harness
    )
    assert resources.workspace == "none"
    assert resources.max_results == 5
    assert resources.search_provider == "tavily"
    assert (
        config.browsecomp_benchmark_environment_server.environment_servers.legacy_agent.agent_server.name
        == "browsecomp_benchmark_agent"
    )
