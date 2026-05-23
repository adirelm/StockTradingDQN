# TODO — phased task list (TradeDQN)

Task tracker (§2.2). **Definition of Done (DoD)** for every build task: behaviour
implemented + a test asserting it + gates green (ruff, ≤150 lines, coverage ≥85%)
+ an evidence pointer (file / test / screenshot) recorded + committed.

**Ownership (§2.2.3c).** Solo project — every task below is owned end-to-end by the
solo developer (architect role: scope, architecture, reward design, acceptance
criteria, review sign-off), with the AI as implementer against each approved PRD.
No per-task hand-off, so ownership is stated once here rather than per row.

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

Status snapshot: **178 tests / 100% coverage** (statement + branch), ruff clean,
≤150 code lines/file, secret-scan clean, wheel imports from a clean build.

## Post-review remediation log (guideline-audit findings)
This section is a remediation record, not open work — every item below is already
closed. It logs fixes made after the pre-submission audit so the trail is auditable.
(Detailed findings live in the local review notes, kept outside the submission.)
| Item | Status |
|---|---|
| **Blocker** — `.gitignore env/` excluded `src/tradedqn/env/` | ✅ fixed |
| Version 1.0.0 + config `version` + startup validation (§7/§8/§19) | ✅ fixed |
| ISO/IEC 25010 — add Compatibility + Portability (§13) | ✅ fixed |
| README Configuration / Extending / Contributing / Credits (§2.1/§12) | ✅ fixed |
| References / bibliography (§18) | ✅ fixed |
| Concurrency & thread-safety write-up (§15) | ✅ fixed |
| `docs/PRD.md` + `PLAN.md` + `TODO.md` (§2.2) | ✅ fixed |
| §5 gatekeeper `execute()` + retry + logging | ✅ `data/gatekeeper.py` |
| §14 installed-wheel config lookup | ✅ `config.py` importlib.resources fallback + wheel force-include |
| §11 token-cost table + budget management | ✅ `docs/COST_ANALYSIS.md` |
| §10 Nielsen heuristics + accessibility (+ screenshots) | ✅ README §UX + `assets/` |
| §9 Jupyter analysis notebook + parameter sweep | ✅ `notebooks/analysis.ipynb`, `scripts/parameter_sweep.py` |
| §4.2 base-class/EpisodeRunner for the shared rollout | ✅ `services/rollout.py` (`RolloutService`) |
| §6 reproducible train-from-scratch (seed Torch init) | ✅ `seeding.py` + `test_seeding.py` |
| Backtest benchmark/marker alignment (anchor at first execution price) | ✅ `services/backtest.py` |
| Minors: chart DPI, §16 I/O/Setup docstrings, CI test-report | ✅ `charts.py` `_DPI`, gatekeeper/env docstrings, `.github/workflows/ci.yml` |

## Known limitations (deliberate, documented)
- DQN underperforms Buy & Hold out-of-sample (overfits) — reported honestly; not the goal.
- Remote on GitHub; dependabot PRs open, no human feature PRs (solo dev); review loop is human PRD-approval + AI-output review.
- Single-threaded by design (see PLAN concurrency).
