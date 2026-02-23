# Action Builder Module

Converts raw mouse clicks to `Discrete(2305)` action indices (ActionBuilder) and converts action indices back to PyAutoGUI execution (ActionExecutor). Handles both the recording side (BC data collection) and the playback side (live agent).

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/action/action_builder.py` | 164 | `ClickEvent` and `ActionEvent` dataclasses. `classify_click()`, `clicks_to_actions()` (state machine), `build_action_timeline()`. |
| `src/action/action_executor.py` | 101 | `ActionExecutor.play_card()` two-click sequence: card slot then arena position with 0.15s delay. `execute()` for full action dispatch. |
| `src/action/action_constants.py` | 51 | Card slot normalized positions, arena Y bounds for click classification, timing constants. |
| `src/action/CLAUDE.md` | - | Detailed technical reference (data structures, card slot regions, state machine, dependencies). |

## Click Classification

Checks card slots first (4 regions), then arena (`y_norm` in 0.0521-0.7813), otherwise "other". Card slots are priority -- if a click lands in a card slot region, it is always classified as a card click.

## State Machine (Click Pairing)

```
idle -> card click    -> card_selected(card_id)
idle -> arena/other   -> ignore, stay idle

card_selected -> arena click    -> emit ActionEvent, return to idle
card_selected -> other card     -> update card_id, stay card_selected
card_selected -> other click    -> cancel, return to idle
```

## Action Encoding

```
action = card_id * 576 + row * 18 + col    (0..2303)
action = 2304                                (no-op)
```

Uses `placement_to_action()` / `action_to_placement()` from `src.encoder.coord_utils`.

## ActionExecutor

Takes an action index, decodes to `(card_id, col, row)` via `action_to_placement()`, converts grid cell to normalized screen coords via `cell_to_norm()`, then executes a PyAutoGUI two-click sequence (card slot click, 0.15s delay, arena position click).

## Tests

46 tests across 2 files:
- `tests/test_action_builder.py` - Click classification, state machine, timeline building
- `tests/test_action_executor.py` - Card play execution, encode/decode roundtrips, corner cells

```bash
python -m pytest docs/josh/action_builder_module/tests/ -v
```

## Documentation

- `docs/action-builder-docs.md` - Full module documentation
- `docs/bc-data-collection-guide.md` - End-to-end BC data collection guide

## Dependencies

- `src.encoder.coord_utils` (norm_to_cell, cell_to_norm, placement_to_action, action_to_placement)
- `src.encoder.encoder_constants` (NOOP_ACTION, GRID_COLS, GRID_ROWS, arena fractions)
- `pyautogui` (for ActionExecutor only, optional for recording)

## Used By

- **dataset_builder_module** - DatasetBuilder uses ActionBuilder for click-to-action encoding
- **Live game agent** - ActionExecutor converts policy outputs to in-game card placements
