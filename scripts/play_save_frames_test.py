import argparse
import sys
import pygame

import pickle
import dill
import jax
import jax.random as jrandom
import numpy as np
import json
import os

from PIL import Image

from jaxatari.games.jax_kangaroo import KangarooState
from jaxatari.environment import JAXAtariAction
from utils import get_human_action, load_game_environment, load_game_mods, update_pygame

UPSCALE_FACTOR = 4

# Map action names to their integer values
ACTION_NAMES = {
    v: k
    for k, v in vars(JAXAtariAction).items()
    if not k.startswith("_") and isinstance(v, int)
}

DEFAULT_SIZES = {
    "Platform": (128, 4),
    "Ladder":   (8, 35),   
    "Player":   (8, 24),
    "Child":    (8, 15),
    "Fruit":    (7, 11),
    "Bell":     (6, 11),
}

# eventually not relevant?
def objects_to_string(struct_obs, defaults=DEFAULT_SIZES):
    parts = []

    # singletons (if present)
    if "player_x" in struct_obs and "player_y" in struct_obs:
        w,h = defaults["Player"]
        parts.append(f"Player at ({int(struct_obs['player_x'])}, {int(struct_obs['player_y'])}), ({w}, {h})")

    if "child_x" in struct_obs and "child_y" in struct_obs:
        w,h = defaults["Child"]
        parts.append(f"Child at ({int(struct_obs['child_x'])}, {int(struct_obs['child_y'])}), ({w}, {h})")

    # multi-instance lists
    for key, label in [
        ("platform_positions", "Platform"),
        ("ladder_positions",   "Ladder"),
        ("fruit_positions",    "Fruit"),
        ("bell_positions",     "Bell"),
    ]:
        if key in struct_obs:
            w,h = defaults[label]
            for x,y in struct_obs[key]:
                if x == -1 or y == -1:
                    continue
                parts.append(f"{label} at ({int(x)}, {int(y)}), ({w}, {h})")

    return "[" + ", ".join(parts) + "]"

def xy(pos):
    if pos is None:
        return None

    # unwrap [[x,y]] -> [x,y]
    if isinstance(pos, (list, tuple)) and len(pos) == 1 and isinstance(pos[0], (list, tuple)):
        pos = pos[0]

    if not (isinstance(pos, (list, tuple)) and len(pos) >= 2):
        return None

    x, y = pos[0], pos[1]
    if x == -1 or y == -1:
        return None
    return int(x), int(y)

def objects_from_struct_obs(struct_obs: dict, default_sizes: dict):
    """
    Generic object extraction:
    - *_positions: list of (x,y)
    - *_position: single (x,y)
    - player_x/player_y: single (x,y)
    Everything becomes a list of dicts with category, x,y,w,h.
    """
    objs = []

    def add_obj(category, x, y):
        if x is None or y is None:
            return
        x, y = int(x), int(y)
        if x == -1 or y == -1:
            return
        w, h = default_sizes.get(category, (0, 0))
        objs.append({"category": category, "x": x, "y": y, "w": int(w), "h": int(h)})

    # Special case: player_x/player_y naming (common)
    if "player_x" in struct_obs and "player_y" in struct_obs:
        add_obj("Player", struct_obs["player_x"], struct_obs["player_y"])

    for key, val in struct_obs.items():
        # skip player_x/player_y since handled above
        if key in ("player_x", "player_y"):
            continue

        # list of positions: e.g., fruit_positions, ladder_positions
        if key.endswith("_positions") and isinstance(val, (list, tuple)):
            category = key[:-10].replace("_", " ").title().replace(" ", "")  # fruit_positions -> Fruit
            for xy in val:
                if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                    add_obj(category, xy[0], xy[1])

        # single position: e.g., child_position, bell_position
        elif key.endswith("_position") and isinstance(val, (list, tuple)):
            category = key[:-9].replace("_", " ").title().replace(" ", "")  # child_position -> Child

            # Case A: [x, y]
            if len(val) >= 2 and isinstance(val[0], (int, float)) and isinstance(val[1], (int, float)):
                add_obj(category, val[0], val[1])

            # Case B: [[x, y]] or [[x, y], [x2, y2], ...]
            elif len(val) >= 1 and isinstance(val[0], (list, tuple)) and len(val[0]) >= 2:
                for xy in val:
                    if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                        add_obj(category, xy[0], xy[1])

    return objs


def objects_to_string_from_list(objs):
    parts = []
    for o in objs:
        cat = o.get("category", "Unknown")
        x, y = o.get("x", -1), o.get("y", -1)
        w, h = o.get("w", 0), o.get("h", 0)
        parts.append(f"{cat} at ({int(x)}, {int(y)}), ({int(w)}, {int(h)})")
    return "[" + ", ".join(parts) + "]"


def objects_to_list(struct_obs, defaults=DEFAULT_SIZES):
    parts = []

    if "player_x" in struct_obs and "player_y" in struct_obs:
        w,h = defaults["Player"]
        parts.append(f"Player at ({int(struct_obs['player_x'])}, {int(struct_obs['player_y'])}), ({w}, {h})")

    if "child_x" in struct_obs and "child_y" in struct_obs:
        w,h = defaults["Child"]
        parts.append(f"Child at ({int(struct_obs['child_x'])}, {int(struct_obs['child_y'])}), ({w}, {h})")

    for key, label in [
        ("platform_positions", "Platform"),
        ("ladder_positions",   "Ladder"),
        ("fruit_positions",    "Fruit"),
        ("bell_positions",     "Bell"),
    ]:
        if key in struct_obs:
            w,h = defaults[label]
            for x,y in struct_obs[key]:
                if x == -1 or y == -1:
                    continue
                parts.append(f"{label} at ({int(x)}, {int(y)}), ({w}, {h})")

    return parts

def to_serializable(obj):
    if isinstance(obj, (np.ndarray, jax.Array)):
        return obj.tolist()
    elif hasattr(obj, "_asdict"):  # NamedTuple / dataclass-like
        return {k: to_serializable(v) for k, v in obj._asdict().items()}
    elif hasattr(obj, "__dict__"):
        return {k: to_serializable(v) for k, v in vars(obj).items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    else:
        return obj


def obs_to_json(obs, path="obs.json"):
    """Convert observation (NamedTuple-like) into JSON using to_serializable."""
    obs_dict = to_serializable(obs._asdict())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obs_dict, f, indent=2)


def save_frame(image, path):
    import numpy as np
    img = Image.fromarray(np.array(image).astype(np.uint8))
    img.save(path)

def expand_mods(game: str, mods):
    """
    Expands modpacks into a flat list of concrete mod names.
    Assumes there is a mod registry accessible via load_game_mod_registry(game)
    or inside load_game_mods module. Adjust `get_registry(...)` accordingly.
    """
    # Accept string or list
    if mods is None:
        return []
    if isinstance(mods, str):
        mods = [mods]

    registry = load_game_mod_registry(game)  # <-- you need this accessor (see below)

    expanded = []
    seen = set()

    def _expand(name):
        if name in seen:
            raise ValueError(f"Cycle detected in modpacks at: {name}")
        seen.add(name)

        entry = registry.get(name)
        if entry is None:
            # not a known modpack; assume it's a concrete mod name
            expanded.append(name)
            seen.remove(name)
            return

        # If registry maps name -> list, it's a modpack
        if isinstance(entry, list):
            for sub in entry:
                _expand(sub)
        else:
            # registry maps name -> concrete mod class/function
            expanded.append(name)

        seen.remove(name)

    for m in mods:
        _expand(m)

    # de-duplicate preserving order
    out = []
    for m in expanded:
        if m not in out:
            out.append(m)
    return out


def main():
    # --- dataset logging setup ---
    rows = []                         # will hold dict rows for the pkl DataFrame
    episode = 0


    parser = argparse.ArgumentParser(
        description="Play a JAXAtari game, record your actions or replay them.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "-g",
        "--game",
        type=str,
        required=True,
        help="Name of the game to play (e.g. 'freeway', 'pong'). The game must be in the src/jaxatari/games directory.",
    )
    parser.add_argument(
        "-m", "--mod",
        type=str,
        action="append",
        required=False,
        help="Name of the mod class.",
    )

    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        "--record",
        type=str,
        metavar="FILE",
        help="Record your actions and save them to the specified file (e.g. actions.npy).",
    )
    mode_group.add_argument(
        "--replay",
        type=str,
        metavar="FILE",
        help="Replay recorded actions from the specified file (e.g. actions.npy).",
    )
    mode_group.add_argument(
        "--random",
        action="store_true",
        help="Play the game with random actions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for JAX PRNGKey and random action generation.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Frame rate for the game.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose mode.",
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        default="",
        help="Subfolder inside 'frames/' where frames and dataset will be saved (default: '2_normal_4').",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=4,
        help="Save every n-th environment step (default: 4).",
    )
    parser.add_argument(
        "--max-saved-frames",
        type=int,
        default=None,
        help="Maximum number of saved frames (rows in dataset). If None, run until quit.",
    )

    args = parser.parse_args()

    save_stride = args.save_every     # every nth frame
    save_dir = os.path.join(os.getcwd(), "frames", args.output_folder)
    os.makedirs(save_dir, exist_ok=True)

    execute_without_rendering = False
    # Load the game environment
    try:
        env, renderer = load_game_environment(args.game)
        if args.mod is not None:
            mod = load_game_mods(args.game, args.mod)
            env = mod(env)

        if renderer is None:
            execute_without_rendering = True
            print("No renderer found, running without rendering.")

    except (FileNotFoundError, ImportError) as e:
        print(f"Error loading game: {e}")
        sys.exit(1)

    # Initialize the environment
    key = jrandom.PRNGKey(args.seed)
    jitted_reset = jax.jit(env.reset)
    jitted_step = jax.jit(env.step)
    jitted_render = jax.jit(renderer.render)

    # initialize the environment
    obs, state = jitted_reset(key)
    obs_to_json(obs, os.path.join(save_dir, f"initial_obs.json"))

    def struct_from_obs(o):
        return {name: (value.tolist() if isinstance(value, (np.ndarray, jax.Array)) else value)
                for name, value in o._asdict().items()}

    prev_struct = struct_from_obs(obs)
    prev_image = np.array(jitted_render(state)).astype(np.uint8)
    save_idx = 0

    # setup pygame if we are rendering
    if not execute_without_rendering:
        pygame.init()
        pygame.display.set_caption(f"JAXAtari Game {args.game}")
        env_render_shape = jitted_render(state).shape[:2]
        window = pygame.display.set_mode(
            (env_render_shape[1] * UPSCALE_FACTOR, env_render_shape[0] * UPSCALE_FACTOR)
        )
        clock = pygame.time.Clock()

    action_space = env.action_space()

    save_keys = {}
    running = True
    pause = False
    frame_by_frame = False
    frame_rate = args.fps
    next_frame_asked = False
    total_return = 0
    if args.replay:
        with open(args.replay, "rb") as f:
            # Load the saved data
            save_data = np.load(f, allow_pickle=True).item()

            # Extract saved data
            actions_array = save_data["actions"]
            seed = save_data["seed"]
            loaded_frame_rate = save_data["frame_rate"]

            frame_rate = loaded_frame_rate

            # Reset environment with the saved seed
            key = jrandom.PRNGKey(seed)
            obs, state = jitted_reset(key)

            def struct_from_obs(o):
                return {name: (value.tolist() if isinstance(value, (np.ndarray, jax.Array)) else value)
                        for name, value in o._asdict().items()}

            prev_struct = struct_from_obs(obs)
            prev_image = np.array(jitted_render(state)).astype(np.uint8)


        # loop over all the actions and play the game
        for action in actions_array:
            # Convert numpy action to JAX array
            action = jax.numpy.array(action, dtype=jax.numpy.int32)
            if args.verbose:
                print(f"Action: {ACTION_NAMES[int(action)]} ({int(action)})")

            obs, state, reward, done, info = jitted_step(state, action)
            image = jitted_render(state)

            update_pygame(window, image, UPSCALE_FACTOR, 160, 210)
            clock.tick(frame_rate)

            # Check for quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    pygame.quit()
                    sys.exit(0)

        pygame.quit()
        sys.exit(0)

    # main game loop
    step_idx = 0
    max_saved = args.max_saved_frames

    while running:
        # check for external actions
        if (max_saved is not None) and (save_idx >= max_saved):
            print(f"Reached max saved frames: {max_saved}. Stopping.")
            break
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:  # pause
                    pause = not pause
                elif event.key == pygame.K_r:  # reset
                    obs, state = jitted_reset(key)
                elif event.key == pygame.K_f:
                    frame_by_frame = not frame_by_frame
                elif event.key == pygame.K_n:
                    next_frame_asked = True
        if pause or (frame_by_frame and not next_frame_asked):
            continue
        if args.random:
            # sample an action from the action space array
            action = action_space.sample(key)
            key, subkey = jax.random.split(key)
        else:
            # get the pressed keys
            action = get_human_action()

            # Save the action to the save_keys dictionary
            if args.record:
                # Save the action to the save_keys dictionary
                save_keys[len(save_keys)] = action

        if not frame_by_frame or next_frame_asked:
            action = get_human_action()
            prev_obs = obs
            
            # Step environment
            next_obs, state, reward, done, info = jitted_step(state, action)
            total_return += reward

            if next_frame_asked:
                next_frame_asked = False

            # Render next frame
            next_image = np.array(jitted_render(state)).astype(np.uint8)

            # Every 50th frame, log transition
            if step_idx % save_stride == 0:
                if save_idx < 210:
                    prev_struct = struct_from_obs(prev_obs)
                    next_struct = struct_from_obs(next_obs)

                    prev_objs = objects_from_struct_obs(prev_struct, DEFAULT_SIZES)
                    next_objs = objects_from_struct_obs(next_struct, DEFAULT_SIZES)

                    cocos = prev_struct.get("coco_positions", [])
                    if any((x != -1 and y != -1) for x, y in cocos):
                        print("FOUND COCO:", cocos)

                    falling = prev_struct.get("falling_coco_position")
                    if isinstance(falling, (list, tuple)) and len(falling) >= 2:
                        if falling[0] != -1 and falling[1] != -1:
                            print("FOUND FALLING COCO:", falling)

                    # Build one row matching the dataset structure
                    row = {
                        "index": f"{episode:05d}_{save_idx:05d}",
                        "obs": prev_image,
                        "next_obs": next_image,
                        "objects": objects_to_string_from_list(prev_objs),
                        "next_objects": objects_to_string_from_list(next_objs),
                        "action": np.float32(action),              
                        "reward": np.float32(reward),              
                        "original_reward": np.float32(getattr(info, "raw_reward", reward)),
                        "done": bool(done),
                    }
                    rows.append(row)

                    save_frame(next_image, os.path.join(save_dir, f"frame_{save_idx:05d}.png"))
                    #state_pkl_path = os.path.join(save_dir, f"state_{save_idx:05d}.pkl")

                    #with open(state_pkl_path, "wb") as f:
                    #    dill.dump(state, f)

                    save_idx += 1

                else:
                    import ipdb; ipdb.set_trace()
                    prev_struct = struct_from_obs(prev_obs)

            # Advance to next step
            obs = next_obs
            prev_struct = struct_from_obs(next_obs)
            prev_image = next_image
            step_idx += 1

            if done:
                print(f"Done. Total return {total_return}")
                total_return = 0
                episode += 1
                obs, state = jitted_reset(key)
                prev_struct = struct_from_obs(obs)
                prev_image = np.array(jitted_render(state)).astype(np.uint8)
                step_idx = 0


        # Render the environment
        if not execute_without_rendering:
            image = jitted_render(state)

            update_pygame(window, image, UPSCALE_FACTOR, 160, 210)

            clock.tick(frame_rate)

    if args.record:
        # Convert dictionary to array of actions
        save_data = {
            "actions": np.array(
                [action for action in save_keys.values()], dtype=np.int32
            ),
            "seed": args.seed,  # The random seed used
            "frame_rate": frame_rate,  # The frame rate for consistent replay
        }
        with open(args.record, "wb") as f:
            np.save(f, save_data)

    # --- Save dataset ---
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows, columns=[
            "index", "obs", "next_obs", "objects", "next_objects",
            "action", "reward", "original_reward", "done"
        ])
        out_pkl = os.path.join(save_dir, f"dataset_{episode}.pkl")
        df.to_pickle(out_pkl)
        print(f"Wrote {len(df)} transitions to {out_pkl}")

    pygame.quit()


if __name__ == "__main__":
    main()