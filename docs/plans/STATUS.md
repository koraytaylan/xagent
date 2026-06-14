# Plan status

Roll-up board for `docs/plans/` — **one row per plan, no per-task detail**. Task-level status lives in each plan's own `STATUS.md` (linked in the last column). This board exists to answer "what is done, in flight, or blocked?" across all plans at a glance, and stays small as plan count grows.

**It is only useful if it is accurate.** Update the relevant row in the *same change* that moves a plan, and keep it in sync with that plan's `STATUS.md` (see [README → STATUS.md](README.md#statusmd)).

_Last updated: 2026-06-14, against `develop`._

| Plan | Title | Status | Tasks | Outcome | Detail |
|---|---|---|---|---|---|
| 0001 | Survival Signal Grounding | ✅ Complete | 10/10 | Deaths-per-food 4.02 → 2.23 (−45%); `lag100` confirmed default. | [status](0001-Survival-Signal-Grounding/STATUS.md) |
| 0002 | CPU/GPU Runtime Decoupling | ✅ Complete | 12/12 | Sim decoupled from redraw onto a GPU worker thread; shared-GPU render path rejected. | [status](0002-CPU-GPU-Runtime-Decoupling/STATUS.md) |

**Status legend:** 📋 Planned (authored, not started) · 🚧 In progress (some tasks merged) · ✅ Complete (all tasks merged to `develop`) · ⛔ Blocked (waiting on a gate or upstream task) · 🗄️ Superseded (replaced or abandoned).
