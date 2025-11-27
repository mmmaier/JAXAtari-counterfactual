from jaxatari.renderers import JAXGameRenderer

import argparse
import json
import os
import dill

import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import jax.numpy as jnp

from utils import load_game_environment  


class CounterfactualRenderer(JAXGameRenderer):
    def __init__(self, base_renderer: JAXGameRenderer):
        self.base_renderer = base_renderer

    def render(self, state, extra_objects=None):
        # 1. Render base frame
        frame = self.base_renderer.render(state)

        # 2. Overlay PNGs if given
        if extra_objects:
            frame = _paste_pngs_onto_frame(frame, extra_objects)
            print("Extra objects!", extra_objects)

        return frame
    

    def export_counterfactual(state, extra_objects, frame, filename_prefix="cf"):
        state_json = {
            "game_state": state, 
            "inserted_objects": extra_objects,
        }
        with open(f"{filename_prefix}.json", "w") as f:
            json.dump(state_json, f, indent=2)

        img = Image.fromarray(frame.astype("uint8"))
        img.save(f"{filename_prefix}.png")


def from_serializable(state_json):
    """
    Convert JSON-loaded state back into a JAX/numpy-friendly format.
    This should mirror your `to_serializable` function.
    For now, assume arrays are stored as lists.
    """
    if isinstance(state_json, list):
        return jnp.array(state_json)
    if isinstance(state_json, dict):
        return {k: from_serializable(v) for k, v in state_json.items()}
    return state_json

from PIL import Image
import numpy as np
import jax.numpy as jnp

def _paste_pngs_onto_frame(frame, extra_objects):
    # 1) Normalize to uint8 RGB for compositing
    orig_dtype = np.array(frame).dtype
    is_float = np.issubdtype(orig_dtype, np.floating)

    frame_np = np.array(frame)
    if is_float:
        frame_np = (np.clip(frame_np, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        frame_np = frame_np.astype(np.uint8)

    # Ensure 3 channels
    if frame_np.ndim == 2:
        frame_np = np.stack([frame_np]*3, axis=-1)
    elif frame_np.shape[-1] == 1:
        frame_np = np.repeat(frame_np, 3, axis=-1)

    H, W = frame_np.shape[:2]
    base_img = Image.fromarray(frame_np, mode="RGB").convert("RGBA")

    print("Frame size (W x H):", W, "x", H)
    for obj in (extra_objects or []):
        print("Placing PNG:", obj["path"], "at", obj["coords"], "with size", obj.get("size"))

    # 2) Paste each PNG with alpha and clipping
    for obj in (extra_objects or []):
        if "path" not in obj:
            continue
        x, y = map(int, obj["coords"])
        overlay = Image.open(obj["path"]).convert("RGBA")
        if "size" in obj:
            overlay = overlay.resize(tuple(obj["size"]), resample=Image.NEAREST)

        ow, oh = overlay.size
        # Completely outside?
        if x >= W or y >= H or x + ow <= 0 or y + oh <= 0:
            continue

        # Clip overlay if it goes out of bounds
        left_clip   = max(0, -x)
        top_clip    = max(0, -y)
        right_clip  = max(0, (x + ow) - W)
        bottom_clip = max(0, (y + oh) - H)
        crop_box = (left_clip, top_clip, ow - right_clip, oh - bottom_clip)
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            continue
        overlay_cropped = overlay.crop(crop_box)

        paste_x = max(0, x)
        paste_y = max(0, y)
        base_img.paste(overlay_cropped, (paste_x, paste_y), overlay_cropped)

    # 3) Convert back to original dtype/scale
    result = np.array(base_img.convert("RGB"), dtype=np.uint8)
    if is_float:
        result = (result.astype(np.float32) / 255.0).astype(orig_dtype)
    else:
        result = result.astype(orig_dtype)

    return jnp.asarray(result)



def show_frame(frame):
    """Display frame with matplotlib."""
    if isinstance(frame, jnp.ndarray):
        frame = np.array(frame)

    # Rescale if needed (check your renderer output)
    if frame.max() <= 1.0:
        frame = (frame * 255).astype(np.uint8)
    else:
        frame = frame.astype(np.uint8)

    plt.imshow(frame)
    plt.axis("off")
    plt.show()


def main(game: str, state_path: str, extra_objects: list):
    # 1. Load env + renderer
    game_env, base_renderer = load_game_environment(game)
    
    renderer = CounterfactualRenderer(base_renderer)

    with open(state_path, "rb") as f:
        state = dill.load(f)

    # 3. Render
    frame = renderer.render(state, extra_objects=extra_objects)

    # 4. Show frame
    show_frame(frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a saved JAXAtari state with optional PNG overlays.")
    parser.add_argument("game", type=str, help="Game name (e.g., Pong, Breakout)")
    parser.add_argument("state_path", type=str, help="Path to saved state file (.pkl or .json)")
    parser.add_argument(
        "--extra-png",
        action="append",
        nargs=3,
        metavar=("X", "Y", "PATH"),
        help="Overlay a PNG at (X,Y). Example: --extra-png 50 60 assets/rope.png",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=(32, 32),
        help="Resize the PNG to this width and height (default: 32 32)",
    )
    args = parser.parse_args()

    # Build list of extra objects
    extra_objects = []
    if args.extra_png:
        for obj in args.extra_png:
            x, y = map(int, obj[:2])
            path = obj[2]
            extra_objects.append({
                "coords": (x, y),
                "path": path,
                "size": tuple(args.size),
            })

    main(args.game, args.state_path, extra_objects)


