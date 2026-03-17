# ggpui

This repository is a fork of GPUI.

- Upstream GPUI: https://github.com/zed-industries/zed/tree/main/crates/gpui
- Upstream site/docs: https://www.gpui.rs/

## What this fork adds

- Custom draw docs: [`crates/gpui/docs/custom_draw.md`](crates/gpui/docs/custom_draw.md)
- Superellipse corners docs: [`crates/gpui/docs/superellipse.md`](crates/gpui/docs/superellipse.md)
- Overflow fade masks docs: [`crates/gpui/docs/overflow_fade.md`](crates/gpui/docs/overflow_fade.md)

## Upstream sync plan

We plan to sync this fork with upstream GPUI once per month.

Each sync will:

- pull latest upstream GPUI changes
- re-apply and verify fork-specific additions
- run checks/examples and fix any breakages
- update docs if any behavior/API changes

