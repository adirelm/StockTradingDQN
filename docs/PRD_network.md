# PRD — Phase 4: Dueling Conv1D DQN (PyTorch)

**Phase:** 4 of 10 · **Status:** approved-to-build · Covers `REQUIREMENTS.md`
**B8** (network). Architecture is taken directly from the deck (low judgment).

## Goal
A `nn.Module` that maps the **30×10** state to **3 Q-values**
`[Q(Sell), Q(Hold), Q(Buy)]` via a Dueling head over Conv1D features.

## Architecture (deck slide "ארכיטקטורת DQN")
```
input  (B, window=30, features=10)
  → transpose → (B, 10, 30)         # Conv1D convolves the TIME axis (length=30);
  → Conv1D(10→32, k=3, pad=1) ReLU  # features are channels, never convolved across
  → Conv1D(32→64, k=3, pad=1) ReLU
  → Flatten                         # (B, 64*30)
  → Linear(→ dense_units=128) ReLU  # shared trunk
  → split:
       Value     head: Linear(128 → 1)        → V(s)      (B,1)
       Advantage head: Linear(128 → n_act=3)  → A(s,a)    (B,3)
  → aggregate:  Q(s,a) = V(s) + A(s,a) − mean_a' A(s,a')
  → output (B, 3)
```
`padding = kernel_size // 2` keeps the time length = `window`, so the flatten
dim is `conv_channels[-1] * window` (deterministic, no dummy forward needed).

All dims come from config: `network.conv_channels [32,64]`, `network.kernel_size`,
`network.dense_units`, `features.window_size`, `features.features_count`,
and `len(actions)` for the output size.

## Why Dueling (deck "למה מפצלים")
`V(s)` learns "how good is this market state" regardless of action; `A(s,a)`
learns "which action is relatively better". Mean-centering `A` makes the
decomposition identifiable (otherwise V and A could drift by a constant).

## Dueling ablation (B8 second half · §9 research)
The `dueling` flag is a config-toggled ablation deliverable: with
`network.dueling: false` the head collapses to a plain DQN (`Q = A`, the value
stream unused) over the *same* Conv1D trunk, isolating the contribution of the
dueling split. It is surfaced end-to-end — `TradingSDK.compare()` trains both
arms ("Dueling DQN" vs "Plain DQN") on identical data and returns per-arch
reward histories, and the GUI exposes a Dueling-vs-plain comparison that
overlays both reward curves.

## Public API
```
DuelingDQN(window, n_features, n_actions, conv_channels, kernel_size, dense_units,
           dueling=True)               # dueling=False → plain DQN (Q = A), the §9 ablation
DuelingDQN.from_config(cfg)            # reads config dims + network.dueling flag
.forward(x: Tensor[B,window,features]) -> Tensor[B, n_actions]
.value_advantage(x) -> (V: Tensor[B,1], A: Tensor[B,n_actions])   # for inspection/tests
```
~253k learnable params at config dims (`conv_channels [32,64]`, `dense_units=128`).

## Acceptance criteria (tests assert)
- `forward` on a `(B, 30, 10)` batch returns `(B, 3)`; works for `B=1`.
- Aggregation identity: `Q == V + A − A.mean(dim=1, keepdim=True)` (uses
  `value_advantage`); advantage stream is mean-zero in its Q contribution.
- Determinism: same `torch.manual_seed` + same input → identical output.
- `from_config` builds a net whose output width == `len(cfg.actions)` and whose
  first conv `in_channels == features_count`; it honours `network.dueling`.
- Ablation: `dueling=False` makes `forward` return the advantage stream verbatim
  (`Q == A`, no value/mean-centering), so it diverges from the `dueling=True` output.
- Params are learnable (non-empty `parameters()`, output requires grad).

## Gates
≤150 code lines/file · TDD · coverage ≥85% · ruff clean. CPU/MPS both fine
(small net — see ADR cost note later).
