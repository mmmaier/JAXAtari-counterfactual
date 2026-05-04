import argparse
import sys
import pygame

import jax
import jax.random as jrandom
import numpy as np
import json
import os

from PIL import Image

from jaxatari.games.jax_freeway import FreewayState
from jaxatari.environment import JAXAtariAction
from utils import get_human_action, load_game_environment, load_game_mods, update_pygame

UPSCALE_FACTOR = 4

ACTION_NAMES = {
    v: k
    for k, v in vars(JAXAtariAction).items()
    if not k.startswith("_") and isinstance(v, int)
}

# Freeway step() expects compact indices (0=NOOP, 1=UP, 2=DOWN), not raw Atari values.
# get_human_action() returns raw Atari integers (NOOP=0, FIRE=1, UP=2, DOWN=5, ...).
# FIRE (space=1) is kept as UP since Freeway has no fire action and space is intuitive.
_FREEWAY_UP   = {1, 2, 6, 7, 10, 14, 15}   # FIRE, UP, UPRIGHT, UPLEFT, UPFIRE, ...
_FREEWAY_DOWN = {5, 8, 9, 13, 16, 17}       # DOWN, DOWNRIGHT, DOWNLEFT, DOWNFIRE, ...

def to_freeway_action(raw_action) -> jax.Array:
    """Map raw JAXAtariAction integer → Freeway compact index (0=NOOP, 1=UP, 2=DOWN)."""
    raw = int(raw_action)
    if raw in _FREEWAY_UP:
        return jax.numpy.array(1, dtype=jax.numpy.int32)
    if raw in _FREEWAY_DOWN:
        return jax.numpy.array(2, dtype=jax.numpy.int32)
    return jax.numpy.array(0, dtype=jax.numpy.int32)


def _fields(obj) -> dict:
    """Return field dict for both NamedTuples (_asdict) and flax struct.dataclasses (__dataclass_fields__)."""
    if hasattr(obj, '_asdict'):
        return obj._asdict()
    if hasattr(obj, '__dataclass_fields__'):
        return {f: getattr(obj, f) for f in obj.__dataclass_fields__}
    return {}


def objects_from_obs(obs) -> list:
    """
    Extract objects from all ObjectObservation fields in the obs struct.
    ObjectObservation has x, y, width, height, active — used directly by Freeway.
    Handles both single objects (scalar arrays) and batches (1-D arrays).
    """
    objs = []

    for field_name, val in _fields(obs).items():
        if not (hasattr(val, 'x') and hasattr(val, 'y') and
                hasattr(val, 'width') and hasattr(val, 'height')):
            continue

        category = field_name.replace("_", " ").title().replace(" ", "")

        x = np.array(val.x)
        y = np.array(val.y)
        w = np.array(val.width)
        h = np.array(val.height)
        active = np.array(val.active) if hasattr(val, 'active') else np.ones_like(x, dtype=bool)

        if x.ndim == 0:  # single object
            if bool(active):
                objs.append({"category": category, "x": int(x), "y": int(y),
                             "w": int(w), "h": int(h)})
        else:  # multiple objects
            for i in range(len(x)):
                if bool(active[i]):
                    objs.append({"category": category, "x": int(x[i]), "y": int(y[i]),
                                 "w": int(w[i]), "h": int(h[i])})

    return objs


def objects_to_string_from_list(objs):
    parts = []
    for o in objs:
        cat = o.get("category", "Unknown")
        x, y = o.get("x", -1), o.get("y", -1)
        w, h = o.get("w", 0), o.get("h", 0)
        parts.append(f"{cat} at ({int(x)}, {int(y)}), ({int(w)}, {int(h)})")
    return "[" + ", ".join(parts) + "]"


def to_serializable(obj):
    if isinstance(obj, (np.ndarray, jax.Array)):
        return obj.tolist()
    elif hasattr(obj, '_asdict'):
        return {k: to_serializable(v) for k, v in obj._asdict().items()}
    elif hasattr(obj, '__dataclass_fields__'):
        return {f: to_serializable(getattr(obj, f)) for f in obj.__dataclass_fields__}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    else:
        return obj


def obs_to_json(obs, path="obs.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(obs), f, indent=2)


def save_frame(image, path):
    img = Image.fromarray(np.array(image).astype(np.uint8))
    img.save(path)


def main():
    rows = []
    episode = 0

    parser = argparse.ArgumentParser(
        description="Play Freeway, save frames and dataset.",
        allow_abbrev=False,
    )
    parser.add_argument("-g", "--game", type=str, required=True,
                        help="Name of the game (use 'freeway').")
    parser.add_argument("-m", "--mods", nargs='+', type=str, required=False,
                        help="Mod name(s).")
    parser.add_argument("--allow_conflicts", action="store_true",
                        help="Allow conflicting mods.")

    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument("--record", type=str, metavar="FILE",
                            help="Record actions to FILE.")
    mode_group.add_argument("--replay", type=str, metavar="FILE",
                            help="Replay actions from FILE.")
    mode_group.add_argument("--random", action="store_true",
                            help="Play with random actions.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--output-folder", type=str, default="",
                        help="Subfolder inside 'frames/' for output.")
    parser.add_argument("--save-every", type=int, default=4,
                        help="Save every n-th step (default: 4).")
    parser.add_argument("--max-saved-frames", type=int, default=None,
                        help="Stop after saving this many frames.")

    args = parser.parse_args()

    save_stride = args.save_every
    save_dir = os.path.join(os.getcwd(), "frames", args.output_folder)
    os.makedirs(save_dir, exist_ok=True)

    execute_without_rendering = False
    try:
        env, renderer = load_game_environment(args.game)
        if args.mods is not None:
            mod_applier = load_game_mods(
                game_name=args.game,
                mods_config=args.mods,
                allow_conflicts=args.allow_conflicts,
            )
            env = mod_applier(env)
            if hasattr(env, 'renderer'):
                renderer = env.renderer
        if renderer is None:
            execute_without_rendering = True
            print("No renderer found, running without rendering.")
    except (FileNotFoundError, ImportError) as e:
        print(f"Error loading game: {e}")
        sys.exit(1)

    key = jrandom.PRNGKey(args.seed)
    jitted_reset = jax.jit(env.reset)
    jitted_step = jax.jit(env.step)
    jitted_render = jax.jit(renderer.render)

    obs, state = jitted_reset(key)
    prev_state = state
    obs_to_json(obs, os.path.join(save_dir, "initial_obs.json"))

    prev_image = np.array(jitted_render(state)).astype(np.uint8)
    save_idx = 0

    if not execute_without_rendering:
        pygame.init()
        pygame.display.set_caption(f"JAXAtari {args.game}")
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
            save_data = np.load(f, allow_pickle=True).item()
            actions_array = save_data["actions"]
            key = jrandom.PRNGKey(save_data["seed"])
            frame_rate = save_data["frame_rate"]
            obs, state = jitted_reset(key)
            prev_image = np.array(jitted_render(state)).astype(np.uint8)

        for action in actions_array:
            action = jax.numpy.array(action, dtype=jax.numpy.int32)
            obs, state, reward, done, info = jitted_step(state, to_freeway_action(action))
            image = jitted_render(state)
            update_pygame(window, image, UPSCALE_FACTOR, 160, 210)
            clock.tick(frame_rate)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    pygame.quit()
                    sys.exit(0)
        pygame.quit()
        sys.exit(0)

    step_idx = 0
    max_saved = args.max_saved_frames

    while running:
        if max_saved is not None and save_idx >= max_saved:
            print(f"Reached max saved frames: {max_saved}. Stopping.")
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_p:
                    pause = not pause
                elif event.key == pygame.K_r:
                    obs, state = jitted_reset(key)
                elif event.key == pygame.K_f:
                    frame_by_frame = not frame_by_frame
                elif event.key == pygame.K_n:
                    next_frame_asked = True

        if pause or (frame_by_frame and not next_frame_asked):
            continue

        if args.random:
            action = action_space.sample(key)
            key, _ = jax.random.split(key)
        else:
            action = get_human_action()
            if args.record:
                save_keys[len(save_keys)] = action

        if not frame_by_frame or next_frame_asked:
            action = get_human_action()
            prev_obs = obs
            prev_state = state

            next_obs, state, reward, done, info = jitted_step(state, to_freeway_action(action))
            total_return += reward

            if next_frame_asked:
                next_frame_asked = False

            next_image = np.array(jitted_render(state)).astype(np.uint8)

            if step_idx % save_stride == 0:
                prev_objs = objects_from_obs(prev_obs)
                next_objs = objects_from_obs(next_obs)

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

                # save_frame(next_image, os.path.join(save_dir, f"frame_{save_idx:05d}.png"))
                save_idx += 1

            obs = next_obs
            prev_image = next_image
            step_idx += 1

            if done:
                print(f"Done. Total return {total_return}")
                total_return = 0
                episode += 1
                obs, state = jitted_reset(key)
                prev_image = np.array(jitted_render(state)).astype(np.uint8)
                step_idx = 0

        if not execute_without_rendering:
            image = jitted_render(state)
            update_pygame(window, image, UPSCALE_FACTOR, 160, 210)
            clock.tick(frame_rate)

    if args.record:
        save_data = {
            "actions": np.array(list(save_keys.values()), dtype=np.int32),
            "seed": args.seed,
            "frame_rate": frame_rate,
        }
        with open(args.record, "wb") as f:
            np.save(f, save_data)

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows, columns=[
            "index", "obs", "next_obs", "objects", "next_objects",
            "action", "reward", "original_reward", "done",
        ])
        out_pkl = os.path.join(save_dir, f"dataset_{episode}.pkl")
        df.to_pickle(out_pkl)
        print(f"Wrote {len(df)} transitions to {out_pkl}")

    pygame.quit()


if __name__ == "__main__":
    main()
