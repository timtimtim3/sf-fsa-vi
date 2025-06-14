#!/usr/bin/env python
import warnings

from matplotlib.figure import Figure
import sys

warnings.filterwarnings("ignore", "Glyph.*missing.*", UserWarning)
from omegaconf import DictConfig
import hydra
import gym
import numpy as np
from envs.utils import get_rbf_activation_data
from sfols.plotting.plotting import plot_all_rbfs

import os
import zipfile
import urllib.request
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties


# Font constants for the Font Awesome *web* package
FONT_DIR = "visualize_env"
ZIP_NAME = "fontawesome-free-5.15.4-web.zip"
ZIP_URL = (
    "https://github.com/FortAwesome/Font-Awesome/"
    "releases/download/5.15.4/fontawesome-free-5.15.4-web.zip"
)
ZIP_PATH = os.path.join(FONT_DIR, ZIP_NAME)
TARGET_TTF = "fa-solid-900.ttf"
TTF_PATH = os.path.join(FONT_DIR, TARGET_TTF)
LETTER_FONT_DIR = FONT_DIR  # reuse your symbol_font folder


def ensure_letter_font(letter_font: str) -> str:
    """
    Download and register a LaTeX-like letter font (lmr or cmr),
    or any URL, into LETTER_FONT_DIR. Returns the local path.
    """
    os.makedirs(LETTER_FONT_DIR, exist_ok=True)

    # built‐in presets for Latin Modern (lmr) and Computer Modern Unicode (cmr)
    presets = {
        "lmr": "https://mirrors.ircam.fr/pub/CTAN/fonts/lm/fonts/opentype/public/lm/lmroman10-regular.otf",
        "cmr": "https://mirrors.ibiblio.org/CTAN/fonts/cm-unicode/fonts/otf/cmunrm.otf",
    }

    if letter_font in presets:
        url = presets[letter_font]
        fname = os.path.basename(url)
    elif letter_font.startswith("http"):
        url = letter_font
        fname = os.path.basename(url)
    else:
        raise ValueError(f"Unknown letter_font '{letter_font}', must be one of {list(presets)} or a URL")

    path = os.path.join(LETTER_FONT_DIR, fname)
    if not os.path.isfile(path):
        print(f"Downloading letter font '{letter_font}' from {url} …")
        urllib.request.urlretrieve(url, path)
        print(f"Saved letter font to {path}")
    # register with Matplotlib
    font_manager.fontManager.addfont(path)
    return path


def ensure_font_ttf():
    """Download the Font Awesome web ZIP (with TTFs) and extract/register fa-solid-900.ttf."""
    os.makedirs(FONT_DIR, exist_ok=True)

    # If font already extracted, just register and return
    if os.path.isfile(TTF_PATH):
        print(f"Font already present: {TTF_PATH}")
        font_manager.fontManager.addfont(TTF_PATH)
        return

    # Download the ZIP if it's not already present
    if not os.path.isfile(ZIP_PATH):
        print(f"Downloading {ZIP_NAME}…")
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
        print("Download complete.")

    # Extract only the desired TTF from the webfonts/ directory
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for member in zf.namelist():
            if member.endswith(f"webfonts/{TARGET_TTF}"):
                print(f"Extracting {member}…")
                zf.extract(member, FONT_DIR)
                src = os.path.join(FONT_DIR, member)
                os.makedirs(os.path.dirname(TTF_PATH), exist_ok=True)
                shutil.move(src, TTF_PATH)
                print(f"Saved TTF to {TTF_PATH}")
                break
        else:
            raise FileNotFoundError(f"{TARGET_TTF} not found in web ZIP!")

    # Register the font with Matplotlib so it can be used by name
    font_manager.fontManager.addfont(TTF_PATH)
    print("Font registered:", TTF_PATH)


def visualize_grid(env,
                   map_array: np.ndarray,
                   mapping: dict,
                   font_keys: list,
                   fa_solid: FontProperties,
                   letter_fp: FontProperties,
                   cell_size: float = 1.0,
                   border_width: float = 1.0,
                   obstacle_border_width: float = 3.0,
                   outer_border_width: float = 6.0,
                   fontsize: int = 20,
                   tele_fontsize: int = 17) -> Figure:
    """
    Render the 2D 'map_array' grid with custom borders and symbols.

    - Empty cells: thin black border
    - Obstacles ('X'): white fill, thicker border
    - Outer boundary: thick black border
    - Symbols via 'mapping'; entries in 'font_keys' use FA font
    """
    nrows, ncols = map_array.shape
    fig, ax = plt.subplots(figsize=(ncols * cell_size * 0.5,
                                    nrows * cell_size * 0.5))

    teleport_coords_to_color = {}
    teleport_coords_to_symbol = {}
    if hasattr(env, "TELEPORT_COORDS"):
        for coords in env.teleport_start_coords:
            teleport_coords_to_color[coords] = [1.0, 0.0, 0.0, 0.4]
            teleport_coords_to_symbol[coords] = env.TELEPORT_MAP[coords]
        for coords in env.teleport_to_coords:
            teleport_coords_to_color[coords] = [0, 0.1, 0.9, 0.4]
            teleport_coords_to_symbol[coords] = env.TELEPORT_MAP[coords]

    # Draw each cell
    for r in range(nrows):
        for c in range(ncols):
            coords = (r, c)

            # if this is a teleport cell, draw a semi-transparent background
            color = teleport_coords_to_color.get(coords)
            if color is not None:
                ax.add_patch(patches.Rectangle(
                    (c, r), 1, 1,
                    facecolor=color,  # RGBA tuple
                    edgecolor='none', # no border here
                    zorder=0          # beneath the border/text
                ))

            cell = map_array[r, c]

            # Determine border width
            lw = obstacle_border_width if cell == 'X' else border_width
            rect = patches.Rectangle(
                (c, r), 1, 1,
                fill=False,
                linewidth=lw,
                edgecolor='black'
            )
            ax.add_patch(rect)

            # Draw symbol if mapped
            glyph = mapping.get(cell, '')
            if glyph:
                fp = fa_solid if cell in font_keys else letter_fp
                ax.text(c + 0.5, r + 0.5, glyph,
                        fontproperties=fp,
                        fontsize=fontsize,
                        ha="center", va="center_baseline")

            if coords in teleport_coords_to_symbol:
                tele_glyph = teleport_coords_to_symbol[coords]
                ax.text(
                    c + 0.5, r + 0.5, tele_glyph,
                    fontproperties=letter_fp,
                    fontsize=tele_fontsize,
                    ha='center', va='center_baseline',
                    zorder=3
                )

    # Outer border
    outer = patches.Rectangle(
        (0, 0), ncols, nrows,
        fill=False,
        linewidth=outer_border_width,
        edgecolor='black'
    )
    ax.add_patch(outer)

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')
    plt.tight_layout()
    return fig


@hydra.main(version_base=None, config_path="conf", config_name="visualize_env")
def main(cfg: DictConfig) -> None:
    env_params = dict(cfg.env)
    env_name = env_params.pop("env_name")
    train_env_kwargs = {
        k: v for k, v in {
            "add_obj_to_start": env_params.get("add_obj_to_start"),
            "add_empty_to_start": env_params.get("add_empty_to_start"),
            "level_name": env_params.get("level_name"),
            "only_rbf": env_params.get("only_rbf")
        }.items() if v is not None
    }
    train_env = gym.make(env_name, **train_env_kwargs)

    # rbf_data, grid_size = get_rbf_activation_data(train_env, exclude={"X"})
    # plot_all_rbfs(rbf_data, grid_size, train_env, skip_non_goal=False)

    # Ensure the Font Awesome TTF is available
    ensure_font_ttf()
    fa_solid = FontProperties(fname=TTF_PATH)

    letter_fp = None
    lf = getattr(cfg.visualize_env_mapping, "letter_font", None)
    if lf:
        letter_path = ensure_letter_font(lf)
        letter_fp = FontProperties(fname=letter_path)

    # Visualize the grid
    fig = visualize_grid(train_env, train_env.MAP, cfg.visualize_env_mapping.mapping, cfg.visualize_env_mapping.font_keys,
                         fa_solid, letter_fp)

    mapping_name = next(
        (arg.split("=",1)[1]
         for arg in sys.argv
         if arg.startswith("visualize_env_mapping=")),
        "grid"
    )

    # ← NEW: ensure output dir exists and save
    out_dir = os.path.join(FONT_DIR, mapping_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{mapping_name}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved visualization to {out_path}")

    # still display interactively
    plt.show()


if __name__ == "__main__":
    main()
