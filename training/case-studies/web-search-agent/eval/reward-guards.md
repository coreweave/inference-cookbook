# Reward guards

A search agent can earn the reward without doing the research. It can find a page that publishes
the question with its answer, or it can write an answer that talks the judge into a yes. ASearcher's
and BrowseComp's answers are on the open web, in dataset mirrors, in ASearcher's published
training data, in the simple-evals CSVs and in papers that quote the questions. A policy that
reaches them learns to look up the answer key instead of learning to search, and its BrowseComp
score then measures the leak, not the model.

Training now uses Gym v0.7.0 bundled with NeMo-RL v0.8.0, and evaluation also uses Gym v0.7.0.
Both have upstream BrowseComp contamination filtering. The shared [`you_search.py`](you_search.py)
also filters ASearcher answer keys before shortening results. Contamination filtering controls
what the agent can read; answer validation and judge instructions control what can earn a reward.
The shared adapter applies four guards to both training and evaluation:

- **Results that quote a benchmark are dropped.** A search result or page is dropped when its text
  or URL names BrowseComp, ASearcher or OpenAI's simple-evals (Gym v0.7.0's patterns
  plus ASearcher's names), or when its URL is a HuggingFace dataset viewer. The
  whole page is checked before `YOU_SEARCH_MAX_CHARS` shortens it. When every result of a call
  is dropped, the call looks like one that found nothing. This is deliberate: a message saying the results were blocked would tell
  the agent that the answer key is nearby. Each drop logs a `[browsecomp][contamination]` line, the
  same line that Gym v0.7.0's filter logs.
- **Answer-key hosts are never fetched.** [`exclude_domains.json`](exclude_domains.json) lists
  `huggingface.co`, which hosts the ASearcher training data, and
  `openaipublic.blob.core.windows.net`, which serves the BrowseComp answer CSV. `prepare_search_data.py` downloads ASearcher for training and BrowseComp for evaluation;
  the search agent cannot fetch these hosts through its tools. The harness drops these hosts' search
  results and does not browse them.
- **Malformed final answers score 0 without a judge call.** Outside complete `<think>` blocks,
  the final response must contain exactly one `Exact Answer:` declaration. The answer must be
  1–200 characters on one line; it may follow the declaration on the next line. A numeric
  `Confidence:` suffix is allowed; both labels may be bold. Explanations before the declaration are excluded. Multiple
  declarations, extra lines after the answer, control characters and forged grading fields
  are rejected. Overlong answers are rejected rather than truncated, and forged fields are
  rejected rather than stripped into a different answer.
- **The judge treats the answer as data.** The question, reference and validated answer are
  encoded as separate JSON strings with readable Unicode, with the reference before the submission.
  Quotes, backslashes and newlines remain escaped. A separate system
  instruction tells the judge to treat those values as data and ignore instructions inside them.
  A new first rule, R0, rejects alternative guesses, hedging between candidates and instructions
  to award credit. It allows components explicitly requested by the question, aliases of the
  same entity, and words such as "or" in an official title. Gym's R1–R5 equivalence rules,
  verdict format and retry handling remain unchanged.

This grading contract is stricter than Gym's default BrowseComp grading and the previous adapter,
which selected the last answer declaration and truncated it. Compare scores only with runs graded
the same way. Regenerate training data and rerun evaluation preparation to include the new
answer-format instructions in policy prompts. Local adversarial tests cover malformed submissions, JSON escaping, system-message
placement, retry behavior and the real Gym v0.7.0 judge path with mocked verdicts. They verify the
contract and request construction, not the live LLM judge's resistance to every injection.

These guards leave these gaps:

- The filter matches names. It misses a page that quotes a question and its answer without naming
  the benchmark, and it cannot catch answers the base model memorized. If the training reward
  rises much faster than the BrowseComp score, read the trajectories.
- Semantic ambiguity and instructions within an otherwise valid answer still require the LLM
  judge to apply R0 correctly. JSON encoding and system instructions reduce exposure but do not
  guarantee that the judge will resist every injection.
- The web is live, so new copies of the questions keep appearing. A dated, frozen search index
  would close this gap and would also make runs repeatable.
- A failed `browse` call returns its error to the agent as text. A You.com outage during browsing
  can therefore score 0 instead of failing the rollout. A failed `search` call fails the rollout,
  except when You.com rejects the query itself (400 or 422, which it returns for some long
  queries). The agent then sees `Search failed`, as it would from Tavily, and can search again.
- Neither config turns on the harness's `bash_command` tool (Gym v0.6.0 and later, with
  `workspace: per_session`), so the agent cannot run code. Without code it cannot, for example,
  decrypt BrowseComp's published CSV.

