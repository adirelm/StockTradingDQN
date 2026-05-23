# ADR-002 — Intra-package imports: absolute (`from tradedqn.…`)

**Status:** Accepted · **Decider:** human (architect)

## Context
§14 of the submission guidelines recommends *relative* intra-package imports
(e.g. `from .model import DQNAgent`). This project instead uses **absolute**
imports throughout (`from tradedqn.model.agent import DQNAgent`) — every
intra-package reference, zero relative. This ADR records why the deviation is
deliberate.

## Decision
Use **absolute imports** rooted at the `tradedqn` package for all intra-package
references, in `src/`, `tests/`, and `scripts/`.

## Rationale
- **`src/` layout + installed wheel.** `tradedqn` ships as a wheel
  (`[tool.hatch.build.targets.wheel] packages = ["src/tradedqn"]`). Absolute
  imports resolve identically whether the code runs from the checkout
  (`pythonpath = ["src"]`) or from an installed package — no dependence on a
  module's position in the tree.
- **Readability at the SDK boundary.** `sdk.py` wires eight subpackages; `from
  tradedqn.services.training import TrainingService` states the layer at a
  glance, reinforcing the layer boundaries the §1.4 contract treats as
  human-decided architecture.
- **Refactor-safe.** Moving a module between subpackages does not silently break
  sibling relative imports.
- **Tooling-neutral.** Ruff's `I` (isort) rule sorts import groups but does not
  convert styles; no `flake8-tidy-imports` rule is enabled, so the convention is
  enforced by consistency, not by a linter that would fight it.

## Consequences
The guideline's relative-import recommendation is consciously **not** followed;
this ADR is the record. Only the import *spelling* differs — package structure,
public API, and behaviour are unaffected. Trade-off accepted: a reader expecting
relative imports finds absolute ones, mitigated by project-wide consistency (no
mixed styles).
