import numpy as np
import matplotlib.pyplot as plt
import os

def create_wavy_rope_from_sprite(input_sprite_path, output_path, wave_amplitude=2, wave_frequency=0.6, rope_thickness=3):
    """
    Creates a fully colored wavy rope sprite using the color from an input sprite.

    Parameters:
    - input_sprite_path: path to the original sprite (to extract color)
    - output_path: where to save the new rope .npy
    - wave_amplitude: horizontal amplitude of the wave
    - wave_frequency: frequency of the wave
    - rope_thickness: width of the rope in pixels
    """
    sprite = np.load(input_sprite_path)
    rows, cols = sprite.shape[:2]

    # Extract first non-transparent pixel color
    rope_color = None
    for r in range(rows):
        for c in range(cols):
            if sprite[r, c, 3] > 0:  # alpha > 0
                rope_color = sprite[r, c]
                break
        if rope_color is not None:
            break
    if rope_color is None:
        rope_color = np.array([255, 255, 255, 255], dtype=sprite.dtype)  # fallback to white

    # Start with fully transparent array
    rope_sprite = np.zeros_like(sprite)

    # Draw the wavy rope
    for r in range(rows):
        c_center = int(cols // 2 + wave_amplitude * np.sin(wave_frequency * r * 2 * np.pi))
        for t in range(-rope_thickness // 2, rope_thickness // 2 + 1):
            c = c_center + t
            if 0 <= c < cols:
                rope_sprite[r, c] = rope_color

    # Save the new rope
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, rope_sprite)
    return rope_sprite

def show_sprite(arr, title="Sprite"):
    plt.imshow(arr)
    plt.title(title)
    plt.axis('off')
    plt.show()

def main():
    cwd = os.getcwd()
    
    input_path = os.path.join(cwd, "src", "jaxatari", "games", "modified_sprites", "kangaroo", "ladder.npy")
    output_path = os.path.join(cwd, "src", "jaxatari", "games", "modified_sprites", "kangaroo", "rope.npy")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    rope_sprite = create_wavy_rope_from_sprite(input_path, output_path, wave_amplitude=2, wave_frequency=0.2, rope_thickness=2)
    
    show_sprite(rope_sprite, "Wavy Rope Sprite")
    

if __name__ == "__main__":
    main()
