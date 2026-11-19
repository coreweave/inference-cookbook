# Search tool access and destination policy

ASearcher training and BrowseComp evaluation use the same server-side
[`you_search.py`](you_search.py) adapter. The policy has `search` and `browse`,
not a shell or a general HTTP/file tool. This change addresses observable URL
bypasses at that boundary. It adds no search capability and claims no benchmark
score improvement. The existing named-benchmark contamination filter is unchanged;
answer-contract/judge hardening and reference-answer/failure handling are separate work.

## Audited boundary

Gym v0.7.0's BrowseComp harness passes exclusions into `search`, but its `browse`
precheck compares an unnormalized hostname. It calls `extract(urls=...)` without
passing exclusions. Previously the adapter trusted that precheck and did not apply
server exclusions to browse requests or returned source URLs. A direct adapter call,
a trailing-dot hostname, or an alternate dataset transport could bypass that layer.

Gym's pinned agent also constructs resource paths directly from model-produced tool names.
An advertised tool schema alone does not enforce the allowed operations. A task-local guard on
the agent's `ServerClient.request` now permits only exact `POST /search` and `POST /browse`
resource calls while the policy loop runs. Forged names such as `verify`, `bash_command`,
`../verify`, encoded names, queries and absolute URLs produce an ordinary unsupported-tool
result before any HTTP dispatch. The harness's trusted seed/verify calls outside that loop
retain their existing path; their implementation and reward semantics are unchanged.
This gate is scoped to policy-originated dispatch, not authentication for other clients that can
already reach the resources server. It relies on the recipe exposing no independent HTTP or
code-execution tool; adding such a capability requires a new access-boundary review.
The shared entrypoint is installed into both BrowseComp component directories for training
and evaluation, so the URL gate and dispatch gate ship together.

The adapter now binds the trusted exclude list to **every** key-rotated client.
Per-call search options can add exclusions but cannot remove those server defaults.
The same policy runs on search result URLs, browse requests before the Contents API,
and returned browse URLs, including empty/failed pages. A rejected browse batch
makes no provider call. The harness's early browse check uses this policy too.

| Input or result | Policy |
| --- | --- |
| Web destination | Require an absolute HTTP(S) URL, a DNS-shaped host and the scheme's default port; reject credentials, IP/numeric hosts, local names, controls, backslashes and malformed escapes. |
| Host spelling | Normalize IDNA, case, trailing dots and default ports. Reject escaped authorities rather than guessing the crawler's parser. Match domains on label boundaries. |
| Paths and query targets | Inspect a bounded percent-decoded view and normalize dot/slash segments for policy checks. Keep reserved path/query encodings in the URL sent to the provider. Check visible embedded HTTP(S) targets in wrappers and archive/redirect URLs. |
| Known artifact transport | Retain the configured HuggingFace and OpenAI storage exclusions; honor the `hf.co` alias when HuggingFace is excluded. Reject dataset/viewer/API and resolve-cache routes on HuggingFace, its short alias and `hf-mirror.com`. |
| Returned source identity | Check the documented `url` and any explicitly reported original/requested/source/final/resolved/redirect URL fields or redirect hops, including source envelopes under `contents` or `metadata`. |
| Research content | Preserve ordinary research sites, scientific PDFs, CSV/JSON sources, GitHub research repositories and API responses. A page's hyperlinks, favicons and thumbnails are not treated as the fetched page's source identity. |
| Tool selection | Pin the adapted Tavily path and inline workspace. Startup rejects a shell workspace or alternate provider; tool arguments cannot select an arbitrary API endpoint, method, headers or local file. |

Only accepted, canonical source URLs and the existing selected text fields enter
Tavily-shaped tool results. Named contamination still checks full provider results
before text caps. Blocked URLs/content are omitted without a benchmark-specific
message to the policy. Destination drops log only per-operation counts, without source URLs,
key paths or payload text. API keys and local task files remain in the trusted server
process; the policy is not given a route for reading them. This is not a redesign
of reference-answer storage or error/reward propagation.

## What the provider exposes

The audit used [Gym v0.7.0's harness](https://github.com/NVIDIA-NeMo/Gym/blob/v0.7.0/resources_servers/browsecomp_advanced_harness/app.py)
and You.com's public [Search schema](https://you.com/docs/api-reference/search/v1-search)
and [Contents schema](https://you.com/docs/api-reference/contents), checked on
2026-11-16. The adapter retains the supported legacy `GET /v1/search` interface
and `POST /v1/contents`; it does not migrate the recipe's extraction behavior.

Configured exclusions are validated and canonicalized at startup. A malformed hostname
fails startup with a configuration error.

The documented page identity is `url`; Contents does not specify whether it echoes the
request or reports the final URL. Successful returned pages are evaluated under the same
destination policy independently of exact URL matching. Missing and empty pages are
ordinary tool outcomes. Neither schema promises a complete redirect
chain or a separately verified final origin. Additional source/redirect fields are
checked defensively **if present**, not assumed to be available. The adapter does
not request or claim an unsupported provider-side redirect, DNS or IP restriction.

There are two different network hops:

1. The trusted Gym resources server calls the fixed You.com API endpoints. The
   adapter explicitly refuses HTTP redirects on those authenticated API requests,
   including when an injected client normally follows redirects. This avoids
   forwarding its API key through an API-origin redirect.
2. You.com's crawler accesses research URLs remotely. Application validation can
   reject a requested URL and withhold a returned page; it cannot enforce the
   crawler's DNS resolution, IP routing, unseen redirects, cache origin or parsing.
   Search/live-crawl can fetch a result before the adapter sees its URL. Pod or
   CoreWeave sandbox egress controls would not constrain that remote crawl.

Actual crawler containment would require supported provider enforcement, a trusted
fetch service with verified DNS/IP and redirect handling, or a controlled index.
None is added here. Opaque redirectors/proxies, hostname aliases not in the policy,
DNS rebinding, missing/misleading source metadata and unnamed or encoded answer-key
mirrors remain limitations. A clean-looking URL is not evidence that content is
uncontaminated. Live web changes and memorized answers remain additional limits.

## Validation

Run CPU-only fixtures with no keys or paid requests:

```bash
pytest training/case-studies/web-search-agent/tests/test_tool_access.py \
  training/case-studies/web-search-agent/tests/test_you_search.py
```

Fixtures exercise trailing dots, IDNA, credentials, alternate schemes and numeric
hosts, percent/double encoding, dot segments, dataset APIs, visible wrappers and
returned source URLs. Mocked Search/Contents calls check both admission and returned
results, API redirect refusal, fixed tool endpoints and continued PDF/CSV/JSON
research access. The suite also drives real Gym v0.7.0 model/tool loops with forged function names and
confirms normal search/browse and trusted verification remain usable. It runs real server methods with mocked
You.com transport when the pinned harness dependencies are installed. These tests
validate the application boundary, not live provider redirect or DNS behavior.

The standalone `pocketshell` package name on PyPI is not the dependency used by this Gym tag;
Gym v0.7.0 vendors its own `pocketshell` under the BrowseComp resource directory. Integration
checks must use that vendored package and the tag's declared harness requirements, not an
unrelated package with the same import name.
