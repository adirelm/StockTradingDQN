# ADR-001 — Configuration format: YAML (with the deck's parameter names)

**Status:** Accepted · **Date:** 2026-05-21 · **Decider:** human (architect)

## Context
§7 of the submission guidelines forbids hardcoded values — all tunable
parameters must live in a config file. Dr. Segal's reference project uses
`config/setup.json` (JSON). Assignment 1 (DroneRL) used `config/config.yaml`.
The rubric is format-agnostic: it grades "no hardcoding", not the file extension.

## Decision
Use **YAML** (`config/config.yaml`), but keep the **lecturer's exact parameter
names** (`window_size`, `features_count`, `gamma`, `epsilon_start/min/decay`,
`actions: {sell, hold, buy}`, etc.).

## Rationale
- A DQN trading project has many hyperparameters, and each needs a short "why".
  YAML supports **inline comments**; JSON does not. Self-documenting config is
  a quality signal and reduces the "magic number" smell.
- Consistency with Assignment 1 (`CLAUDE.md` convention) → one mental model.
- Keeping the deck's **names** means a grader cross-referencing the reference
  `setup.json` finds the same keys — we get name-compatibility without losing
  comments.

## Consequences
- The config loader (`src/tradedqn/config.py`) parses YAML via `PyYAML`
  (already a dependency).
- If the lecturer later requires `setup.json` specifically, the loader is the
  only switch point — values and names already match, so it's a parser swap.
- Trade-off accepted: a reader expecting his exact `setup.json` filename will
  find `config.yaml` instead; mitigated by matching keys + this ADR.

## Table-2 version mapping (§8.1-b)
The guidelines' Table-2 lists `src/<pkg>/shared/version.py` (initial `1.00`) as
the version declaration. We satisfy the *intent* — one declared version,
initialised at the 1.0 baseline and **validated at startup** — without that file
path:

| Table-2 expectation | This project |
|---|---|
| `src/<pkg>/shared/version.py` = `1.00` | `__version__ = "1.0.0"` in `src/tradedqn/__init__.py` |
| (single source of truth) | mirrored in `pyproject.toml` and `config/config.yaml` |
| (no startup check required) | **stronger**: `config.py::_check_version` refuses an incompatible major at load |

A standalone `shared/version.py` would add a module under the 100%-coverage gate
and a *fourth* place a version string lives — against single-source-of-truth. The
version is declared once authoritatively (`__init__.py`), validated once
(`config.py`), and a grader cross-referencing Table-2 finds the same `1.0` baseline.
