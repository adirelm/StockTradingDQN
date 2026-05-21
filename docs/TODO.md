# TODO — phased task list (TradeDQN)

Task tracker (§2.2). **Definition of Done (DoD)** for every build task: behaviour
implemented + a test asserting it + gates green (ruff, ≤150 lines, coverage ≥85%)
+ an evidence pointer (file / test / screenshot) recorded + committed.

## Build phases — all ✅ done (one PRD + commit each)
| # | Phase | Status | Evidence |
|---|---|---|---|
| 1 | Data layer (client + §5 gatekeeper + cache) | ✅ | `data/`, tests; commit Phase 1 |
| 2 | Features (indicators + normalize + split) | ✅ | `features/`, tests; Phase 2 |
| 3 | Trading environment (state/reward) | ✅ | `env/`, tests; Phase 3 |
| 4 | Dueling Conv1D DQN | ✅ | `model/network.py`, tests; Phase 4 |
| 5 | Training (replay + target + loop) | ✅ | `model/agent.py`, `services/training.py`; Phase 5 |
| 6 | Backtest + inference + metrics | ✅ | `services/`, tests; Phase 6 |
| 7 | SDK facade | ✅ | `sdk.py`, tests; Phase 7 |
| 8 | Terminal interface | ✅ | `cli/menu.py`, `main.py`; Phase 8 |
| 9 | GUI dashboard | ✅ | `gui/`, tests; Phase 9 |
| 10 | Docs + real results | ✅ | README, PROMPTS, COST_ANALYSIS, charts; Phase 10 |

Status snapshot: **130 tests / 100% coverage**, ruff clean, ≤150 code lines/file,
secret-scan clean, wheel imports from a clean build.

## Pre-submission review remediation (guideline-audit findings)
(Detailed findings live in the local review notes, kept outside the submission.)
| Item | Status |
|---|---|
| **Blocker** — `.gitignore env/` excluded `src/tradedqn/env/` | ✅ fixed |
| Version 1.0.0 + config `version` + startup validation (§7/§8/§19) | ✅ fixed |
| ISO/IEC 25010 — add Compatibility + Portability (§13) | ✅ fixed |
| README Configuration / Extending / Contributing / Credits (§2.1/§12) | ✅ fixed |
| References / bibliography (§18) | ✅ fixed |
| Concurrency & thread-safety write-up (§15) | ✅ fixed |
| `docs/PRD.md` + `PLAN.md` + `TODO.md` (§2.2) | ✅ fixed (this batch) |
| §5 gatekeeper `execute()` + retry + logging | ⬜ batch 3 |
| §14 installed-wheel config lookup | ⬜ batch 3 |
| §11 token-cost table + budget management | ⬜ batch 4 |
| §10 Nielsen heuristics + accessibility (+ screenshots) | ⬜ batch 4 |
| §9 Jupyter analysis notebook + parameter sweep | ⬜ batch 5 |
| §4.2 base-class/EpisodeRunner for the shared rollout | ⬜ |
| Minors: chart DPI, I/O/Setup docstrings, CI test-report, dangling `PRD_docs` link | ⬜ |

## Known limitations (deliberate, documented)
- DQN underperforms Buy & Hold out-of-sample (overfits) — reported honestly; not the goal.
- No remote/PRs yet (solo repo); the review loop is human PRD-approval + AI-output review.
- Single-threaded by design (see PLAN concurrency).
