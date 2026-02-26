"""Manual env step test -- bypasses SB3 to test the env directly.

Run while IN A MATCH with the game window visible.
Executes 10 real steps and prints exactly what happens.
"""

import sys
import os
import time

_here = os.path.dirname(os.path.abspath(__file__))
_project = os.path.abspath(os.path.join(_here, ".."))
sys.path.insert(0, _project)

import types
_src_pkg = types.ModuleType("src")
_src_pkg.__path__ = [
    os.path.join(_here, "src"),
    os.path.join(_project, "bc_model_module", "src"),
    os.path.join(_project, "state_encoder_module", "src"),
    os.path.join(_project, "action_builder_module", "src"),
]
_src_pkg.__package__ = "src"
sys.modules["src"] = _src_pkg

for pkg, path in {
    "src.ppo": os.path.join(_here, "src", "ppo"),
    "src.bc": os.path.join(_project, "bc_model_module", "src", "bc"),
    "src.encoder": os.path.join(_project, "state_encoder_module", "src", "encoder"),
}.items():
    mod = types.ModuleType(pkg)
    mod.__path__ = [path]
    mod.__package__ = pkg
    sys.modules[pkg] = mod

for sub in ("generation", "detection", "classification", "data", "pipeline"):
    pkg = f"src.{sub}"
    mod = types.ModuleType(pkg)
    mod.__path__ = [os.path.join(_project, "src", "src", sub)]
    mod.__package__ = pkg
    sys.modules[pkg] = mod

import numpy as np
from src.ppo.clash_royale_env import ClashRoyaleEnv, EnvConfig

WINDOW_TITLE = "Clash Royale - thegoodpersonplayer2"

print("Creating env...")
env = ClashRoyaleEnv(EnvConfig(
    window_title=WINDOW_TITLE,
    dry_run=False,
))

print(f"\nGame bounds: x={env._game_x_offset}, y={env._game_y_offset}, "
      f"w={env._game_w}, h={env._game_h}")
print(f"Dispatcher offset: left={env._dispatcher._window_left}, "
      f"top={env._dispatcher._window_top}")
print(f"Dispatcher frame: {env._dispatcher._frame_w}x{env._dispatcher._frame_h}")
print(f"Dispatcher pyautogui loaded: {env._dispatcher._pyautogui is not None}")
print(f"Dispatcher dry_run: {env._dispatcher._config.dry_run}")
print(f"Dispatcher game_hwnd: {env._dispatcher._game_hwnd}")

print(f"\nCalling env.reset() -- will block until game detected...")
t0 = time.time()
obs, info = env.reset()
print(f"reset() returned in {time.time() - t0:.1f}s")
print(f"Initial obs vector: elixir={obs['vector'][0]:.3f}, "
      f"ally_towers={obs['vector'][9]:.3f}, enemy_towers={obs['vector'][10]:.3f}")

print(f"\n{'='*60}")
print("Running 10 manual steps with random card actions...")
print(f"{'='*60}\n")

for i in range(10):
    # Pick a random CARD action (not noop) from the mask
    mask = env.action_masks()
    card_mask = mask[:2304].astype(bool)  # exclude noop
    valid_card_actions = np.where(card_mask)[0]

    if len(valid_card_actions) > 0:
        action = int(np.random.choice(valid_card_actions))
        action_type = f"card {action // 576}, cell {action % 576}"
    else:
        action = 2304
        action_type = "noop (no card actions valid)"

    t0 = time.time()
    obs, reward, terminated, truncated, step_info = env.step(action)
    dt = time.time() - t0

    executed = step_info.get("action_executed", False)
    reason = step_info.get("action_reason", "")
    phase = step_info.get("phase", "?")
    focused = env._dispatcher._is_game_focused()

    print(f"Step {i+1}: action={action} ({action_type})")
    print(f"  executed={executed}, reason='{reason}', phase={phase}, "
          f"focused={focused}, dt={dt:.2f}s")
    print(f"  reward={reward:.3f}, terminated={terminated}, truncated={truncated}")

    if terminated or truncated:
        print(f"  *** Episode ended: terminated={terminated}, truncated={truncated}")
        print(f"  truncation_reason={step_info.get('truncation_reason', '')}")
        print(f"  anomaly={step_info.get('anomaly_detected', False)}")
        break

    time.sleep(0.3)

env.close()
print("\nDone.")
