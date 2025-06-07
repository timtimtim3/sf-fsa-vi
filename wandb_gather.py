import fnmatch
import os
from typing import Union
import pandas as pd
import wandb
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


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
    truncate_at_min: bool = True
):
    """
    Given a dict of DataFrames `dfs` (keyed by label), each containing columns
    `x_axis`, `ycol`, and optionally `ycol + "/std"`, plot each series on the same
    figure. If `ycol + "/std"` is present and not all NaN, shade ±(std_multiplier * std)
    around the mean line, using the same color but transparent. Optionally save to PNG.

    - dfs: e.g. {"flatdqn": flatdqn_df, "sfols": sfols_df, "lof": lof_df} or some values may be None
    - x_axis: name of the column to use for the x-axis
    - ycol: name of the mean column to plot
    - colors: optional dict mapping each key in `dfs` to a matplotlib color
              e.g. {"flatdqn": "green", "sfols": "red", "lof": "blue"}
    - xlabel, ylabel, title: optional labels
    - std_multiplier: how many standard deviations to shade (default = 1.0)
    - save_dir: directory in which to save the figure (default "results")
    - linewidth: width of the plot lines
    - save: if True, write out a PNG under `save_dir`
    - timestamp: override for filename timestamp (otherwise uses now)
    - truncate_at_min: if True, find the smallest “maximum x_axis” among dfs and set x_max to that;
                       if False, use the overall maximum x_axis
    """
    if colors is None:
        colors = {"lof_dqn": "blue", "flat_dqn": "green", "sfols_dqn": "red"}

    # Determine all max and min x-axis values across non-None DataFrames
    x_mins = []
    x_maxs = []
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
    if truncate_at_min:
        global_max = min(x_maxs)
    else:
        global_max = max(x_maxs)

    fig, ax = plt.subplots()

    for label, df in dfs.items():
        if df is None or x_axis not in df.columns or ycol not in df.columns:
            continue

        xi = df[x_axis].values
        yi = df[ycol].values
        color = colors.get(label, None)

        ax.plot(xi, yi, label=label, color=color, linewidth=linewidth)

        std_col = f"{ycol}/std"
        if std_col in df.columns:
            ystd = df[std_col].values
            if not np.all(np.isnan(ystd)):
                lower = yi - std_multiplier * ystd
                upper = yi + std_multiplier * ystd
                ax.fill_between(xi, lower, upper, color=color, alpha=0.2)

    ax.set_xlim(global_min, global_max)
    ax.legend()
    ax.grid(True)
    ax.set_xlabel(xlabel or x_axis)
    ax.set_ylabel(ylabel or ycol)
    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save:
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Sanitize ycol for filename (replace slashes)
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


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    save = cfg.get("save", False)
    truncate_at_min = cfg.get("truncate_at_min", True)
    save_dir = cfg.get("save_dir", None)
    if save_dir is None:
        save_dir = os.path.join("results", "wandb_plots", cfg.env_name)
    else:
        save_dir = os.path.join("results", "wandb_plots", save_dir)

    api = wandb.Api()

    patterns = [
        "learning/fsa_reward/*-task*",
        "learning/fsa_neg_reward/*-task*",
        "learning/fsa_*reward_average"
    ]
    x_axis = "learning/total_timestep"

    flatdqn_run_ids = cfg.get("flatdqn_run_ids", [])
    sfols_run_ids   = cfg.get("sfols_run_ids", [])
    lof_run_ids     = cfg.get("lof_run_ids", [])

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
    dfs = smooth_dfs(dfs, window_size=10, x_axis=x_axis)
    # colors = {
    #     "lof_dqn":     "#1f77b4",  # blue
    #     "flat_dqn":    "#2ca02c",  # green
    #     # "sfols_dqn":   "#d62728",  # red
    #     "sfols_dqn":   "#ff7f0e",
    # }
    colors = None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plot_metric_across_runs(
        dfs,
        x_axis="learning/total_timestep",
        ycol="learning/fsa_neg_reward_average",
        colors=colors,
        xlabel="Total Timestep",
        ylabel=f"Mean Neg. Step Reward",
        title=f"Mean Neg. Step Reward over {n_tasks} FSA tasks ±1 STD",
        std_multiplier=1.0,
        save=save,
        timestamp=timestamp,
        truncate_at_min=truncate_at_min,
        save_dir=save_dir
    )
    plot_metric_across_runs(
        dfs,
        x_axis="learning/total_timestep",
        ycol="learning/fsa_reward_average",
        colors=colors,
        xlabel="Total Timestep",
        ylabel=f"Mean FSA Reward (Success)",
        title=f"Mean FSA Reward (Success) over {n_tasks} FSA tasks ±1 STD",
        std_multiplier=1.0,
        save=save,
        timestamp=timestamp,
        truncate_at_min=truncate_at_min,
        save_dir=save_dir
    )


if __name__ == "__main__":
    main()
