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

## Phase 3 — Delete `evaluate` too ✅ done (branch `cleanup-dead-code-phase1`)

Superseded decision: originally planned to fix `evaluate`
(`evaluation_tools/evaluation.py`) rather than delete it — see
commit `a1a1dba` on the now-abandoned `cleanup-fix-evaluate-phase3` branch for
that work (kept for reference, not merged). Team decided afterward to drop
it entirely instead, so no partner-maintained eval tooling ships with this
repo at all for now.

- [x] Delete the entire `src/mitcfu_rag/evaluation_tools/` package:
      `evaluation.py`, `prompt_template.py`, `notes_eval.txt`, and
      `testset/*.csv` (`edge_cases.csv`, `query_answer.csv`,
      `question_type_with_explanation.csv`,
      `summarize_and_no_information.csv`, `top_10_searches.csv`). Confirmed
      via grep that nothing outside this package imports any of it — the
      `testset/*.csv` files other than `query_answer.csv` weren't even
      referenced by `evaluation.py` itself, they were already dead data.
- [x] Remove the `evaluate` entry from `pyproject.toml:[project.scripts]`.

**Verify:** grep for `evaluation_tools`, `sample_evaluation_questions`,
`IN_CONTEXT_EXAMPLES`, `INSTRUCTIONS` across `src/`, `tests/`,
`pyproject.toml`, README, both Jenkinsfiles, `Dockerfile`, `MANIFEST.in` —
zero hits. `uv sync --frozen` succeeds. `pytest` — 18 passed. `.venv/bin`
now contains exactly `create-faiss-index` and `streaming-service-mitcfu` —
the only two working CLI tools this project ships.

---

## Phase 4 — Dependency list cleanup (`pyproject.toml`) ✅ done (branch `cleanup-dead-code-phase1`, not yet committed)

Verified by grepping for direct imports of every declared dependency across
`src/`:

- [x] **Removed `langchain`** (bare package) — zero direct imports anywhere;
      only `langchain_community`, `langchain_text_splitters`, and
      `langchain_unstructured` are actually imported. Confirmed gone from
      the resolved lockfile entirely (`grep -c 'name = "langchain"$'
      uv.lock` → 0) — nothing else pulls it in transitively either.
- [x] **Kept `accelerate` — do NOT remove.** Static analysis (zero direct
      imports, no literal `device_map="auto"`) suggested this was unused,
      but actually removing it and loading a real model
      (`AutoModel.from_pretrained(name, device_map=torch.device("cpu"))`,
      the exact pattern used in `retrievers/indexes/multilinguale5.py` and
      `streaming_multilingual_retriever.py`) failed hard:
      `ValueError: Using a device_map ... requires accelerate`. Current
      `transformers` requires `accelerate` for *any* `device_map` argument,
      not just `"auto"` — the plan's original assumption was wrong. Restored
      it immediately and re-verified the same load + forward pass succeeds
      with it present. Left as an explicit dependency, unchanged.
- [x] **Removed `tokenizers`** — zero direct imports; still present in the
      resolved lockfile as a transitive dependency of `transformers` (`grep
      -c 'name = "tokenizers"' uv.lock` → 2), so removing the redundant
      explicit top-level pin changed nothing observable.
- [x] **Added `langchain-core`** explicitly — used directly
      (`tests/test_generic_parser.py:4`: `from langchain_core.documents import Document`;
      `generic_parser.py`'s loaders return `Document` objects). Was already
      installing transitively via `langchain-community`; now declared
      directly so a future `langchain-community` bump can't silently drop it.
- [x] Re-ran `uv lock`; `uv.lock` updated accordingly (net: `langchain` and
      `psutil`-via-`accelerate` briefly dropped then `accelerate`+`psutil`
      restored; `langchain-core` already present, now also pinned directly).

**Verify:** `uv sync --frozen` succeeds. `pytest` — 18 passed. Real
model-load smoke test (`sentence-transformers/all-MiniLM-L6-v2` via
`AutoTokenizer`/`AutoModel.from_pretrained(..., device_map=torch.device("cpu"))`
plus a forward pass) succeeds with the final dependency set — this is the
load path the actual retrievers use, not just an import check.

---

## Phase 5 — Smaller correctness fixes ✅ done (branch `cleanup-dead-code-phase1`, not yet committed)

- [x] `langgraph_graphs.py:52-54` — `create_graph(type)` only handled
      `type == "service"`; any other value (including `"evaluate_router"`,
      which `service.py`'s own `--graph-type` help text advertised as valid)
      silently returned `None`, deferring the crash to the first request.
      Didn't implement `evaluate_router` (no spec for it, and the eval
      tooling it would have served is gone as of Phase 3) — instead made
      `create_graph` raise `ValueError` immediately for any unsupported type,
      and stripped the dead `evaluate_router` mention from `service.py`'s
      `--graph-type` help text. Verified: constructing `AgenticGraph(type="evaluate_router", ...)`
      now raises immediately with a clear message; `type="service"` is
      unaffected.
- [x] Narrowed the two remaining bare `except:` blocks:
  - `langgraph_graphs.py:95` → `except (json.JSONDecodeError, AttributeError):`
    (`JSONDecodeError` for unparseable output, `AttributeError` for the case
    where the parsed JSON is valid but not a dict, e.g. a bare list).
  - `langgraph_graphs.py:138` → `except (AttributeError, TypeError):` —
    `_extract_json` already narrowly catches `JSONDecodeError` internally and
    always returns a dict, so this outer catch only needs to guard the
    `.get()` call itself.
  - Verified by replicating both try/except blocks standalone against
    malformed JSON, JSON-that's-a-list, missing keys, and valid input — every
    case produces the identical fallback behavior as the original bare
    except, confirming no legitimate path regressed.
- [x] `ms_marco_minilm_validator.py:5-8` — docstring header described an
      unrelated class (`EmbeddingRetriever`), a copy-paste leftover. Rewrote
      to describe `MsValidator` (the ms-marco-MiniLM-L-6-v2 cross-encoder
      reference validator it actually is).
- [x] `service.py:36-38` — `path_to_labels` and `path_to_JEDs` were dead
      module-level variables (personal hardcoded path, zero references
      anywhere else in the file) — deleted outright rather than moved to
      env vars, since nothing reads them.
      `path_to_embeddings` was only used as `default=path_to_embeddings` on
      the required positional `faiss_path` CLI argument — verified with a
      throwaway argparse script that **`default=` on a plain positional
      argument is a no-op**; argparse still requires the value regardless,
      so this line never did anything. Removed the dead variable and the
      inert `default=` kwarg; `faiss_path` remains a required positional arg
      exactly as it behaved before, just without a dead reference to a
      colleague's home directory sitting in the source.

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
