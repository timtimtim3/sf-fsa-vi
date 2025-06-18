import fnmatch
import os
from typing import Union
import pandas as pd
from utils.utils import read_wandb_run_history
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.ticker as ticker
import matplotlib as mpl


n_tasks = None


def pull_data(run, patterns, x_axis="learning/timestep", samples=100000):
    # Pull *all* history so we can see which columns exist
    all_df = run.history(pandas=True, samples=samples)

    per_key_dfs = []
    for pattern in patterns:
        matched_keys = [col for col in all_df.columns if fnmatch.fnmatch(col, pattern)]

        # For each matched key, pull a tiny DF [timestep, that_key] and store it
        for key in matched_keys:
            df_key = run.history(
                pandas=True,
                keys=[key],
                x_axis=x_axis,
                samples=samples
            )

            # Rename
            prefix, rest = key.rsplit("/", 1)  # rest is something like "OfficeAreas-v0-task1"
            if "Areas" in rest:
                new_rest = rest.replace("Areas", "")
            else:
                new_rest = rest
            new_name = f"{prefix}/{new_rest}"  # e.g. "learning/fsa_neg_reward/Office-v0-task1"
            df_key = df_key.rename(columns={key: new_name})

            per_key_dfs.append(df_key)

    merged = per_key_dfs[0]
    for df in per_key_dfs[1:]:
        merged = pd.merge(merged, df, on=x_axis, how="outer")

    # Sort by timestep and optionally reset index
    merged = merged.sort_values(by=x_axis).reset_index(drop=True)

    return merged


def average_cols(df: pd.DataFrame, cols: list[str], name: str) -> pd.DataFrame:
    """
    Compute a new column `name` that is the row-wise mean of all columns in `cols`.
    """
    # Compute row-wise mean (skip NaNs)
    df[name] = df[cols].mean(axis=1)
    return df


def pull_runs(api, run_ids, entity, project, patterns, x_axis, 
              compute_average_from_fsa_reward=False, stretch_x_axis=False, repeat_in_stretch=False):
    if len(run_ids) < 1:
        return None
    
    run_dfs = {}
    for run_id in run_ids:
        run_path = f"{entity}/{project}/{run_id}"
        run      = api.run(run_path)

        run_df = pull_data(run=run, x_axis=x_axis, patterns=patterns)

        fsa_reward_cols = [col for col in run_df.columns if fnmatch.fnmatch(col, patterns[0])]
        fsa_neg_reward_cols = [col for col in run_df.columns if fnmatch.fnmatch(col, patterns[1])]
        N = len(fsa_neg_reward_cols)

        if compute_average_from_fsa_reward:
            # Get average run_df performance over tasks
            run_df = average_cols(run_df, cols=fsa_reward_cols, name="learning/fsa_reward_average")
            run_df = average_cols(run_df, cols=fsa_neg_reward_cols, name="learning/fsa_neg_reward_average")

            if stretch_x_axis and repeat_in_stretch and N > 0:
                # 1) Keep a copy of the original run_df so we can read its x‐axis
                orig_df   = run_df.copy()
                orig_x    = orig_df[x_axis].values
                n_orig    = len(orig_df)

                # 2) Repeat each row N times
                run_df = (
                    orig_df
                    .loc[orig_df.index.repeat(N)]
                    .reset_index(drop=True)
                )

                # 3) Build a new, linearly spaced x‐axis from
                #    orig_x.min() up to (orig_x.max() * N), with
                #    exactly orig_len * N entries.
                new_min = orig_x.min()
                new_max = orig_x.max() * N
                total   = n_orig * N

                new_x = np.linspace(new_min, new_max, total)

                # 4) Overwrite the x‐axis column
                run_df[x_axis] = new_x

        run_df = run_df.set_index(x_axis)

        run_dfs[run_id] = run_df

    # Concatenate them along the columns, using run_id as the top‐level key
    # Resulting columns are a MultiIndex: (run_id, metric_name)
    concat_df = pd.concat(run_dfs, axis=1)

    # Compute mean and std **across runs** for each metric_name. We group by
    # the second level of the MultiIndex (level=1) and take mean/std along axis=1.
    # df_mean = concat_df.groupby(axis=1, level=1).mean()
    # df_std  = concat_df.groupby(axis=1, level=1).std()
    df_mean = (concat_df.T.groupby(level=1).mean().T)
    df_std = (concat_df.T.groupby(level=1).std().T)

    # Rename the std‐columns to append "/std"
    df_std = df_std.rename(columns={col: f"{col}/std" for col in df_std.columns})

    # Combine mean and std back into one DataFrame (re‐add x_axis as a column)
    final_df = pd.concat([df_mean, df_std], axis=1).reset_index()

    # Sort columns so all means come before their corresponding std
    sorted_cols = []
    for col in df_mean.columns:
        sorted_cols.append(col)
        if f"{col}/std" in df_std.columns:
            sorted_cols.append(f"{col}/std")
    final_df = final_df[[x_axis] + sorted_cols]

    # # Drop any rows where all metric columns (i.e. everything except x_axis) are NaN:
    # metric_cols = [c for c in final_df.columns if c != x_axis]
    # final_df = final_df.dropna(subset=metric_cols, how="all")

    global n_tasks
    if n_tasks is None:
        n_tasks = N
    return final_df


def plot_metric_across_runs(
    dfs: dict[str, Union[pd.DataFrame, None]],
    x_axis: str,
    ycol: str,
    colors: Union[dict[str, str], None] = None,
    xlabel: Union[str, None] = None,
    ylabel: Union[str, None] = None,
    title: Union[str, None] = None,
    std_multiplier: float = 1.0,
    save_dir: str = "results",
    linewidth: float = 1,
    save: bool = False,
    timestamp: Union[str , None] = None,
    truncate_at_min: bool = True,
    fixate_y_ticks: bool = False,
    display_marker_n_times: Union[dict[str, int], None] = None,
    use_sci_x: bool = False
):
    """
    Given a dict of DataFrames `dfs` (keyed by label), each containing columns
    `x_axis`, `ycol`, and optionally `ycol + "/std"`, plot each series on the same
    figure. Now forces the y-axis to run from the next lower multiple of 25 up to +25,
    shows ticks at multiples of 25 up to 0, and optionally displays N markers
    per curve evenly spaced along the x-axis.
    """
    if colors is None:
        colors = {"lof_dqn": "blue", "flat_dqn": "green", "sfols_dqn": "red"}

    # Determine all max and min x-axis values across non-None DataFrames
    x_mins, x_maxs = [], []
    for df in dfs.values():
        if df is None or x_axis not in df.columns:
            continue
        xi = df[x_axis].dropna()
        if xi.empty:
            continue
        x_mins.append(xi.min())
        x_maxs.append(xi.max())

    if not x_mins or not x_maxs:
        raise RuntimeError("No valid x_axis data found in any DataFrame.")

    global_min = min(x_mins)
    global_max = min(x_maxs) if truncate_at_min else max(x_maxs)

    # Prepare figure
    fig, ax = plt.subplots()

    # Collect minima for y-axis adjustment
    ymins = []

    for label, df in dfs.items():
        if df is None or x_axis not in df.columns or ycol not in df.columns:
            continue

        xi = df[x_axis].values
        yi = df[ycol].values
        color = colors.get(label, None)

        # Plot mean line
        ax.plot(xi, yi, label=label, color=color, linewidth=linewidth)

        # Track y-min from mean±std (if std exists)
        std_col = f"{ycol}/std"
        if std_col in df.columns and not np.all(np.isnan(df[std_col].values)):
            ystd = df[std_col].values
            ymins.append(np.min(yi - std_multiplier * ystd))
            lower = yi - std_multiplier * ystd
            upper = yi + std_multiplier * ystd
            ax.fill_between(xi, lower, upper, color=color, alpha=0.15)
        else:
            ymins.append(np.min(yi))

        # place N markers starting at x=global_min, last one one step before global_max
        if display_marker_n_times is not None:
            N = display_marker_n_times.get(label)
            if isinstance(N, int) and N > 0:
                # compute step size so markers sit at:
                #   x = global_min + k*(global_max-global_min)/N,  k=0..N-1
                step_x = (global_max - global_min) / N
                x_marks = global_min + step_x * np.arange(N)
                y_marks = np.interp(x_marks, xi, yi)

                ax.scatter(
                    x_marks, y_marks,
                    color=color,
                    marker='*',
                    s=30,      # smaller dot size
                    alpha=0.8,   # 50% opacity
                    zorder=3
                )

    # 3) x-axis limits and optional scientific scaling
    ax.set_xlim(global_min, global_max)
    if use_sci_x:
        # let matplotlib do the scaling and draw the "×10^6" offset text for us
        ax.ticklabel_format(style="sci", axis="x", scilimits=(6,6))
        # use math‐text for the offset
        ax.xaxis.get_major_formatter().set_useMathText(True)
    # always label normally; the offset text will appear automatically if use_sci_x
    ax.set_xlabel(xlabel or x_axis)

    # === Updated: extend y-axis up to +25, but only tick from bottom to 0 ===
    if fixate_y_ticks:
        step = 25
        raw_min = min(ymins)
        y_bottom = step * np.floor(raw_min / step)
        y_top = 25
        ax.set_ylim(y_bottom, y_top)
        # Only show ticks at multiples of 25 up to 0
        ticks = np.arange(-200, 1, step)   # e.g. [-200, -175, ..., -25, 0]
        ax.set_yticks(ticks)

    # Finish up
    ax.legend(loc="upper left")
    ax.grid(True)
    ax.set_xlabel(xlabel or x_axis)
    ax.set_ylabel(ylabel or ycol)
    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save:
        os.makedirs(save_dir, exist_ok=True)
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ycol_safe = ycol.replace("/", "_")
        filename = f"{ycol_safe}_{timestamp}.png"
        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path, dpi=300)

    plt.show()


def smooth_dfs(
    dfs: dict[str, Union[pd.DataFrame, None]],
    window_size: int,
    x_axis: str
) -> dict[str, pd.DataFrame]:
    """
    For each (key, df) in `dfs`, if df is not None, replace every column
    except `x_axis` with its rolling mean over 'window_size' rows,
    then drop the first (window_size-1) rows (where rolling mean is NaN).
    Returns a new dict with the same keys, where each DataFrame has been
    smoothed & truncated.  If dfs[k] was None, the result[k] is still None.
    """
    smoothed: dict[str, pd.DataFrame] = {}

    for label, df in dfs.items():
        if df is None:
            smoothed[label] = None
            continue

        # Make sure x_axis exists
        if x_axis not in df.columns:
            raise ValueError(f"DataFrame for '{label}' has no column '{x_axis}'")

        # Identify all metric columns (everything except x_axis)
        metric_cols = [c for c in df.columns if c != x_axis]
        if not metric_cols:
            # Nothing to smooth—just keep x_axis
            smoothed[label] = df.copy()
            continue

        # Compute rolling mean over the metric columns; keep x_axis untouched
        rolled_metrics = df[metric_cols].rolling(window=window_size, min_periods=window_size).mean()

        # Truncate the first (window_size-1) rows
        truncated_metrics = rolled_metrics.iloc[window_size - 1 :].reset_index(drop=True)
        truncated_x = df[[x_axis]].iloc[window_size - 1 :].reset_index(drop=True)

        # Reconstruct a new DataFrame with x_axis first, then the smoothed columns
        new_df = pd.concat([truncated_x, truncated_metrics], axis=1)
        smoothed[label] = new_df

    return smoothed


def normalize_run_ids(run_ids_specs):
    """
    Given a list like [flatdqn_run_ids, sfols_run_ids, lof_run_ids],
    where each element is either:
      - a list of run‐IDs, or
      - a string path to a save‐folder
    return a new list of lists, where any string has been replaced
    by load_ids_from_folder(path).
    """
    return [load_ids_from_folder_if_str(spec) for spec in run_ids_specs]


def load_ids_from_folder_if_str(run_ids):
    """
    If `run_ids` is a string, treat it as a save_folder path
    and load the most recent WandB IDs from that folder.
    Otherwise assume it's already a list and just return it.
    """
    if isinstance(run_ids, str):
        return load_ids_from_folder(run_ids)
    return run_ids


def load_ids_from_folder(save_folder):
    """
    Look inside `save_folder` for subdirectories, read their WandB history,
    and collect the most recent run ID from each. If there are no subdirs,
    look only in `save_folder` itself.
    """
    ids = []

    # 1) List immediate subdirectories (if the folder doesn't exist, error out)
    try:
        entries = os.listdir(save_folder)
    except FileNotFoundError:
        raise RuntimeError(f"Save‐folder not found: {save_folder}")

    sub_dirs = [e for e in entries
                if os.path.isdir(os.path.join(save_folder, e))]

    # 2) If no subdirectories, just look in the folder itself
    if not sub_dirs:
        sub_dirs = [""]

    # 3) For each subdirectory, read the WandB history and grab the last run ID
    for sub in sub_dirs:
        path = os.path.join(save_folder, sub)
        hist = read_wandb_run_history(path)      # returns a list of (time, run_id) tuples
        if not hist:
            continue
        _, most_recent_id = hist[-1]
        ids.append(most_recent_id)

    return ids


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    # 1) Plain‐old dict (with all interpolations resolved)
    plain = OmegaConf.to_container(cfg, resolve=True)

    # 2) Extract the nested group
    nested = plain.pop("wandb_gather_run_ids", {})

    # 3) Merge it back into the top level
    plain.update(nested)

    # 4) Reconstruct a DictConfig
    cfg = OmegaConf.create(plain)

    save = cfg.get("save", False)
    truncate_at_min = cfg.get("truncate_at_min", True)
    save_dir = cfg.get("save_dir", None)
    smooth = cfg.get("smooth", True)
    title = cfg.get("title", None)
    font_scale = cfg.get("font_scale", 1.0)  # e.g., 1.2 = 20% larger fonts

    # Global font size scaling (increase as needed)
    base_font_size = 10 
    mpl.rcParams.update({
        "axes.titlesize": base_font_size * font_scale,
        "axes.labelsize": base_font_size * font_scale,
        "xtick.labelsize": base_font_size * font_scale,
        "ytick.labelsize": base_font_size * font_scale,
        "legend.fontsize": base_font_size * font_scale,
        "figure.titlesize": base_font_size * font_scale,
        "mathtext.fontset": "stix",
    })

    display_marker_n_times = {
        "sfols_dqn": cfg.get("display_marker_n_times_sfols", None), 
        "lof_dqn": cfg.get("display_marker_n_times_lof", None),
        "flat_dqn": cfg.get("display_marker_n_times_flatq", None),
        }

    if save_dir is None:
        save_dir = os.path.join("results", "wandb_plots", cfg.run_ids.env_name)
    else:
        save_dir = os.path.join("results", "wandb_plots", save_dir)

    api = wandb.Api()

    patterns = [
        "learning/fsa_reward/*-task*",
        "learning/fsa_neg_reward/*-task*",
        "learning/fsa_*reward_average"
    ]
    x_axis = "learning/total_timestep"

    flatdqn_run_ids = cfg.run_ids.get("flatdqn_run_ids", [])
    sfols_run_ids   = cfg.run_ids.get("sfols_run_ids", [])
    lof_run_ids     = cfg.run_ids.get("lof_run_ids", [])

    flatdqn_run_ids, sfols_run_ids, lof_run_ids = normalize_run_ids([
        flatdqn_run_ids,
        sfols_run_ids,
        lof_run_ids
    ])

    flatdqn = pull_runs(api, flatdqn_run_ids, cfg.wandb.entity, cfg.wandb.project, patterns, 
                        x_axis="learning/timestep", compute_average_from_fsa_reward=True, stretch_x_axis=True, 
                        repeat_in_stretch=True)
    if flatdqn is not None:
        flatdqn = flatdqn.rename(columns={"learning/timestep": x_axis})
    sfols = pull_runs(api, sfols_run_ids, cfg.wandb.entity, cfg.wandb.project, patterns, x_axis=x_axis)
    lof = pull_runs(api, lof_run_ids, cfg.wandb.entity, cfg.wandb.project, patterns, x_axis=x_axis)

    dfs = {
        "flat_dqn": flatdqn,
        "sfols_dqn": sfols,
        "lof_dqn": lof
    }
    if smooth:
        dfs = smooth_dfs(dfs, window_size=10, x_axis=x_axis)
    colors = None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plot_metric_across_runs(
        dfs,
        x_axis="learning/total_timestep",
        ycol="learning/fsa_neg_reward_average",
        colors=colors,
        xlabel="Total Timesteps",
        ylabel=f"Mean Neg. Step Reward",
        title=title if title is not None else f"Mean Neg. Step Reward over {n_tasks} FSA tasks ±1 STD",
        std_multiplier=1.0,
        save=save,
        timestamp=timestamp,
        truncate_at_min=truncate_at_min,
        save_dir=save_dir,
        fixate_y_ticks=True,
        display_marker_n_times=display_marker_n_times,
        use_sci_x=True
    )
    plot_metric_across_runs(
        dfs,
        x_axis="learning/total_timestep",
        ycol="learning/fsa_reward_average",
        colors=colors,
        xlabel="Total Timesteps",
        ylabel=f"Mean FSA Reward (Success)",
        title=title if title is not None else f"Mean FSA Reward (Success) over {n_tasks} FSA tasks ±1 STD",
        std_multiplier=1.0,
        save=save,
        timestamp=timestamp,
        truncate_at_min=truncate_at_min,
        save_dir=save_dir,
        display_marker_n_times=display_marker_n_times,
        use_sci_x=True
    )


if __name__ == "__main__":
    main()
