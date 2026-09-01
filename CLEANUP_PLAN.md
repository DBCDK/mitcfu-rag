# MitCFU-RAG Cleanup Plan

Purpose: get this repo into a state we're comfortable handing to partners as a
co-maintained project. The service stays in our environment and auth is
handled by the gateway — this plan is scoped to code health only: dead code,
unused/undeclared dependencies, broken tooling, and repo hygiene.

Root cause of most of the mess: the project was renamed from `fakta_chat` to
`mitcfu_rag` and the rename was never finished. `grep -r fakta_chat src/` hits
12 files — some are just stale docstrings, several are live `import`
statements pointing at a package that no longer exists.

Decision already made: **fix `evaluate`; delete `mitcfu-RAG-sh`,
`compare-retrievers`, and `evaluate-retrieval`.**

Work is ordered so each phase leaves the repo in a working state — safe to
stop after any phase and pick this back up later.

---

## Phase 0 — Baseline

- [ ] `uv sync --frozen` succeeds, `pytest` passes, `streaming-service-mitcfu --help`
      and `create-faiss-index --help` both run, as a pre-change baseline.

---

## Phase 1 — Delete orphaned modules ✅ done (branch `cleanup-dead-code-phase1`)

None of these are imported by the live pipeline (`service.py` →
`AgenticGraph`/`AgenticRAG` → `EmbeddingRetriever` / `AgentStreamingGenerator`).
Confirmed by grepping every reference to each symbol/module across `src/` and
`tests/` before listing it here.

- [x] Delete `src/mitcfu_rag/rag/dummy_rag.py`
- [x] Delete `src/mitcfu_rag/rag/parsers/dummyparser.py` (and the now-empty
      `rag/parsers/` dir if nothing else lives there — check first)
- [x] Delete `src/mitcfu_rag/rag/retrievers/dummy_retriever.py`
- [x] Delete `src/mitcfu_rag/rag/generators/dummy_generator.py`
  - These four only import each other; `dummy_rag.py` imports from the
    nonexistent `fakta_chat.rag.*` — the cluster is already broken, not just
    unused.
- [x] Delete `src/mitcfu_rag/rag/summarizers/general_summarizer.py`
  - Imports `from fakta_chat.rag.rag import Summarizer` (broken). Builds an
    unused `new_prompt` variable (lines 72-90) that's computed and discarded.
    `AgenticRAG.__init__` hardcodes `self.summarizer = None`
    (`agent_streaming_rag.py:63`) — summarization was never wired in.
  - Deleted the now-unused `Summarizer` ABC in `rag.py` at the same
    time — no concrete subclass remains.
  - If summarization is actually wanted later, treat that as a new feature,
    not a cleanup — write it against the current `mitcfu_rag` package, don't
    resurrect this file.
- [x] Delete `src/mitcfu_rag/rag/retrievers/ensemblers/` (directory contained
      only a 0-byte `__init__.py`)
  - Deleted the unused `Ensembler` ABC in `rag.py` at the same time — zero
    concrete subclasses existed; only referenced in a commented-out import
    in `config.py:170` (`ReciprocalEnsembler`).
- [x] Delete `src/mitcfu_rag/rag/utils.py` (0 bytes, zero references anywhere)
- [x] (found during execution, same pattern) Delete
      `src/mitcfu_rag/rag/parsers/__init__.py` and
      `src/mitcfu_rag/rag/summarizers/__init__.py` — both dirs became empty
      scaffolding once their one real file was removed. Deleted the unused
      `Parser` ABC in `rag.py` too — `DummyParser` was its only subclass.

**Verify:** `grep -rn "dummy_rag\|dummy_retriever\|dummy_generator\|dummyparser\|general_summarizer\|Summarizer\|Ensembler\|Parser" src/` returns nothing outside files intentionally kept (only stale commented-out imports in `config.py`, addressed in Phase 6). `pytest` — 18 passed. `python -c "import mitcfu_rag.rag.rag, mitcfu_rag.rag.agent_streaming_rag, mitcfu_rag.rag.langgraph_graphs, mitcfu_rag.rag.index_vector_db, mitcfu_rag.tools"` — all import cleanly.

---

## Phase 2 — Delete the three broken CLI tools

Per decision: delete, don't fix, these three. All three are dead on arrival —
either `NameError` or `ImportError` the instant they're invoked — and none are
referenced by README, Jenkinsfiles, or tests.

- [ ] Delete `src/mitcfu_rag/term_ui.py`
  - `term_ui.py:9,19` does `from mitcfu_rag.config import RAG; RAG = RAG()` —
    `RAG` was never defined in `config.py`.
- [ ] Delete `src/mitcfu_rag/evaluation_tools/compare_retrievers.py`
  - `compare_retrievers.py:21` loops over `Compare_Retrievers`, which only
    exists commented-out in `config.py:167-175`.
- [ ] Delete `src/mitcfu_rag/evaluation_tools/evaluate_retrieval.py`
  - `evaluate_retrieval.py:9-10` imports `fakta_chat.config` and
    `fakta_chat.evaluation_tools.evaluation` — package doesn't exist.
- [ ] Remove their entries from `pyproject.toml:[project.scripts]`:
  ```
  mitcfu-RAG-sh = "mitcfu_rag.term_ui:cli"
  evaluate-retrieval = "mitcfu_rag.evaluation_tools.evaluate_retrieval:run"
  compare-retrievers = "mitcfu_rag.evaluation_tools.compare_retrievers:run"
  ```
- [ ] `notes_eval.txt:38` references "compare-retrievers" by name in a dev
      note about future retrieval-evaluation work — update or drop that line
      so it doesn't point at a tool that no longer exists.
- [ ] Confirm nothing else imports `sample_evaluation_questions` etc. from
      `compare_retrievers.py` (it doesn't define anything imported elsewhere —
      only consumes from `evaluation.py`, one-directional).

**Verify:** `pyproject.toml` has exactly 3 scripts left (`create-faiss-index`,
`evaluate`, `streaming-service-mitcfu`). `uv sync --frozen` succeeds.

---

## Phase 3 — Fix `evaluate` (`evaluation_tools/evaluation.py`)

This is the one eval tool we're keeping. It currently fails outright and has
a real correctness bug in the request it sends. Fix all of the following:

- [ ] **Missing dependency.** `evaluation.py:11` does
      `from openai import APIConnectionError, OpenAI, RateLimitError`, but
      `openai` is not declared in `pyproject.toml` and is not in `uv.lock`
      (`grep -c 'name = "openai"' uv.lock` → 0). Under the pinned/`--frozen`
      env every CI and Docker build uses, this is `ModuleNotFoundError` on
      import. Add `openai` to `[project].dependencies`, re-lock with
      `uv lock`.
- [ ] **CWD-relative path bug.** `sample_evaluation_questions`
      (`evaluation.py:26`) does
      `pd.read_csv("src/mitcfu_rag/evaluation_tools/testset/query_answer.csv")`
      — only works if the process CWD happens to be the repo root. Replace
      with a path relative to the module: `Path(__file__).parent / "testset" / "query_answer.csv"`.
- [ ] **Real bug: malformed request content.** `generate_predictions`
      (`evaluation.py:116,122`):
      ```python
      query = [line["query"]]
      ...
      "messages": [{"role": "user", "content": query}],
      ```
      `content` ends up as a one-element list containing a raw string. The
      live service's message parser (`service.py:68-75`) expects `content`
      to be either a string or a list of `{"type": "text", "text": ...}`
      dicts, and does `content["text"]` on each item — a plain string item
      will raise `TypeError: string indices must be integers`. This means
      `evaluate` currently cannot successfully call the real service at all.
      Fix: `query = line["query"]` (plain string), and send
      `"content": query` directly.
      - Downstream, `predictions.append({"query": query, ...})` and the
        `f"Question: {query}\n..."` formatting in `evaluate_responses`
        (line 196) will also stop printing Python list reprs once `query` is
        a plain string.
- [ ] **Bare except swallows real bugs.** `parse_response` (`evaluation.py:109`)
      has `except:` catching everything, including bugs in the parsing code
      itself, and returning `-1` — indistinguishable from a legitimate
      "hallucination" score. Narrow to
      `except (json.JSONDecodeError, ValueError, KeyError):`.
- [ ] **Debug print cruft.** `evaluation.py:132,134` (`print(prediction)`,
      `print(references)`) — leftover debug prints inside
      `generate_predictions`. Remove, or convert to `logger.debug(...)`.
- [ ] **Dead commented-out blocks.** Remove the large commented-out code
      blocks in `evaluate_responses` (`evaluation.py:199-228`, `266-270`,
      `302-311`) — they reference variables (`n_miss`, `n_correct_exact`)
      that no longer exist elsewhere in the function; keeping them as
      comments only invites someone to "helpfully" uncomment broken code
      later.
- [ ] Document in README that `evaluate` calls the OpenAI API directly
      (`OpenAI()` at `evaluation.py:338` picks up `OPENAI_API_KEY` from the
      environment) — this is an external, billed dependency a partner needs
      to know about before running it.

**Verify:** `evaluate --help` runs. Point `evaluate <staging-url>` at a real
running instance of the service (staging) with a small `-n` and confirm it
completes a full round trip without the `TypeError` above, and that a
deliberately malformed model response gets logged (not silently swallowed) if
you temporarily break `parse_response`'s regex/JSON parsing to check the
narrowed except actually surfaces it.

---

## Phase 4 — Dependency list cleanup (`pyproject.toml`)

Verified by grepping for direct imports of every declared dependency across
`src/`:

- [ ] **Remove `langchain`** (bare package) — zero direct imports anywhere;
      only `langchain_community`, `langchain_text_splitters`, and
      `langchain_unstructured` are actually imported.
- [ ] **Remove `accelerate`** — zero direct imports, and no `device_map="auto"`
      usage anywhere (both embedding-loading call sites pass an explicit
      `torch.device`). Remove, `uv sync`, and smoke-test that
      `streaming-service-mitcfu` still starts and loads models correctly —
      this one is a "verify before trusting" removal, not a slam dunk.
- [ ] **Remove `tokenizers`** — zero direct imports; already pulled
      transitively and unpinned by `transformers`, so the explicit top-level
      entry does nothing today.
- [ ] **Add `langchain-core`** explicitly — used directly
      (`tests/test_generic_parser.py:4`: `from langchain_core.documents import Document`;
      `generic_parser.py`'s loaders return `Document` objects). Currently only
      installs because `langchain-community` happens to pull it in
      transitively — an undeclared direct dependency is a latent break
      waiting for someone to bump `langchain-community`.
- [ ] **Add `openai`** — see Phase 3.
- [ ] Re-run `uv lock` after all the above and commit the updated `uv.lock`.

**Verify:** `uv sync --frozen`, `pytest`, and a full local start of
`streaming-service-mitcfu` (or at minimum importing every module in
`mitcfu_rag` — `python -c "import mitcfu_rag.service"` etc.) all still work.

---

## Phase 5 — Smaller correctness fixes

- [ ] `langgraph_graphs.py:52-54` — `create_graph(type)` only handles
      `type == "service"`; any other value (including `"evaluate_router"`,
      which `service.py`'s own `--graph-type` help text advertises as valid)
      silently returns `None`, and the service only blows up later, on the
      first request. Either implement `create_evaluate_router_graph()` or
      strip the dead option from the CLI help text in `service.py`.
- [ ] Narrow the two remaining bare `except:` blocks:
  - `langgraph_graphs.py:95` — falls back to `agent = None` on *any*
    exception, not just a JSON parse failure. Narrow to
    `except (json.JSONDecodeError, KeyError, AttributeError):`.
  - `langgraph_graphs.py:138` — same pattern for reformulated queries;
    narrow similarly.
- [ ] `ms_marco_minilm_validator.py:5-8` — docstring header describes an
      unrelated class (`EmbeddingRetriever`), a copy-paste leftover from a
      different file. Rewrite to describe the actual validator.
- [ ] `service.py:36-38` — replace the hardcoded personal path defaults
      (`/data/rani/mitcfu-data/...`) for `path_to_embeddings`,
      `path_to_labels`, `path_to_JEDs` with environment-variable-backed
      config, or drop the defaults and make the CLI args required.

---

## Phase 6 — Repo hygiene sweep

- [ ] Delete `src/mitcfu_rag/__init__ copy.py` (stray tracked 0-byte file).
- [ ] Delete `README_TODO_UV.md` — it's a pasted `ruff` lint-error dump, not
      documentation. All of its listed errors are addressed by this plan
      (the F821 items go away with Phase 2's deletions; the bare-except items
      in Phase 3/5; the `tools/__init__.py` star-import issues below).
- [ ] Clean up `src/mitcfu_rag/tools/__init__.py` star-import pattern
      (`from .embedder import *` etc. — flagged by ruff as F403/F405/E402).
      Replace with explicit imports of the actual public names, or add
      `# noqa` with a one-line reason if the star-export is intentional.
- [ ] Fix `MANIFEST.in` — currently
      `recursive-include src/topic_pages/data *`, referencing a
      `topic_pages` package that doesn't exist in this repo (copy-pasted
      from a sibling project). Either remove the line or point it at real
      package data this project actually ships.
- [ ] Sweep stale `:mod:`fakta_chat...`` docstring headers left over from
      the rename, in: `tools/embedder.py:3`, `tools/knn_searcher.py:2`,
      `tools/semantic_splitter.py:3`, `evaluation_tools/notes_eval.txt`
      references, and any surviving after Phases 1-2 remove the files that
      hold the rest. Update to `mitcfu_rag.*`.
- [ ] Sweep dead commented-out imports referencing the old package name /
      unimplemented retrievers, e.g. `config.py:1-2,166-175`,
      `agent_streaming_rag.py:29,34`, `streamlit_ui.py:9,11-12` — either
      delete outright or replace with a single-line "not implemented yet"
      note if there's a genuine near-term plan to build them.
- [ ] Delete `error.json` from the repo root (untracked debug artifact; not
      in git history) and add a rule to `.gitignore` (e.g. `error.json`,
      `api_responses/` — the latter is where `evaluate`'s `log_responses`
      writes output per `evaluation.py:76`) so these stop accumulating in
      working trees.
- [ ] Add `.pre-commit-config.yaml` wiring up `ruff` — `pre-commit` is
      already declared as a dev dependency in `pyproject.toml` but nothing
      actually configures it.

---

## Phase 7 — Fill in README

Not code cleanup, but blocking for partner onboarding. `README.md` currently
has empty headers for: Unit/Validation/Performance tests, Sanity checks
before MR, Artifacts built, related Jenkins jobs, related Artifactory
artifacts, related repositories, production version, Confluence docs. Fill
these in or remove the headers — an empty section is worse than no section
for someone trying to onboard.

---

## Suggested execution order

Phases 1 → 2 → 4 → 3 → 5 → 6 can be done as one PR each (or combined into 2-3
PRs) with `pytest` + a manual service smoke-test as the gate between phases.
Phase 7 (README) can happen in parallel, it has no code dependencies.
