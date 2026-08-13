# ggpui

This repository is a fork of GPUI.

- Upstream GPUI: https://github.com/zed-industries/zed/tree/main/crates/gpui
- Upstream site/docs: https://www.gpui.rs/

## What this fork adds

- Custom draw docs: [`crates/gpui/docs/custom_draw.md`](crates/gpui/docs/custom_draw.md)
- Superellipse corners docs: [`crates/gpui/docs/superellipse.md`](crates/gpui/docs/superellipse.md)
- Overflow fade masks docs: [`crates/gpui/docs/overflow_fade.md`](crates/gpui/docs/overflow_fade.md)

## Upstream revisions

Each dated revision identifies the matching Zed commit.

| Revision | Zed commit |
| --- | --- |
| [`2026-08-13`](https://github.com/ahkohd/ggpui/tree/2026-08-13) | [`03e5ad8a630c`](https://github.com/zed-industries/zed/commit/03e5ad8a630c84c3990055905d0444ea0a519b7f) |
| [`2026-03-17`](https://github.com/ahkohd/ggpui/tree/2026-03-17) | [`50ca710f515a`](https://github.com/zed-industries/zed/commit/50ca710f515ada2d00803ccc9a8900981ed5eb50) |

## Upstream sync plan

We plan to sync this fork with upstream GPUI once per month.

Each sync will:

- pull the latest upstream GPUI changes
- reapply and verify fork-specific additions
- run checks and examples, then fix any breakages
- update the docs when behaviour or APIs change
- create a dated revision tag

