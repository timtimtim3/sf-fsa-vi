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

            if stretch_x_axis:
                # Stretch x_axis by number of goal‐tasks
                run_df[x_axis] *= N

                if repeat_in_stretch and N > 0:
                    # Repeat each row N times, preserving order
                    run_df = run_df.loc[run_df.index.repeat(N)].reset_index(drop=True)

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
    return final_df, N


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
    timestamp: Union[str , None] = None
):
    """
    Given a dict of DataFrames `dfs` (keyed by label), each containing columns
    `x_axis`, `ycol`, and optionally `ycol + "/std"`, plot each series on the same
    figure. If `ycol + "/std"` is present and not all NaN, shade ±(std_multiplier * std)
    around the mean line, using the same color but transparent. Finally, save to PNG.

    - dfs: e.g. {"flatdqn": flatdqn_df, "sfols": sfols_df, "lof": lof_df} or some values may be None
    - x_axis: name of the column to use for the x-axis
    - ycol: name of the mean column to plot
    - colors: optional dict mapping each key in `dfs` to a matplotlib color
              e.g. {"flatdqn": "green", "sfols": "red", "lof": "blue"}
    - xlabel, ylabel, title: optional labels
    - std_multiplier: how many standard deviations to shade (default = 1.0)
    - save_dir: directory in which to save the figure (default "results")
    """
    if colors is None:
        colors = {"lof": "blue", "flatdqn": "green", "sfols": "red"}

    # Determine global x-axis range across all non-None DataFrames
    x_min, x_max = None, None
    for label, df in dfs.items():
        if df is None or x_axis not in df.columns or ycol not in df.columns:
            continue
        xi = df[x_axis].dropna()
        if xi.empty:
            continue
        local_min, local_max = xi.min(), xi.max()
        if x_min is None or local_min < x_min:
            x_min = local_min
        if x_max is None or local_max > x_max:
            x_max = local_max

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

    if x_min is not None and x_max is not None:
        ax.set_xlim(x_min, x_max)

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

    flatdqn, n_tasks = pull_runs(api, flatdqn_run_ids, cfg.wandb.entity, cfg.wandb.project, patterns, 
                        x_axis="learning/timestep", compute_average_from_fsa_reward=True, stretch_x_axis=True, 
                        repeat_in_stretch=True)
    flatdqn = flatdqn.rename(columns={"learning/timestep": x_axis})
    sfols, _ = pull_runs(api, sfols_run_ids, cfg.wandb.entity, cfg.wandb.project, patterns, x_axis=x_axis)
    lof, _ = pull_runs(api, lof_run_ids, cfg.wandb.entity, cfg.wandb.project, patterns, x_axis=x_axis)

    dfs = {
    "flatdqn": flatdqn,
    "sfols": sfols,
    "lof": lof
    }
    dfs = smooth_dfs(dfs, window_size=10, x_axis=x_axis)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_metric_across_runs(
        dfs,
        x_axis="learning/total_timestep",
        ycol="learning/fsa_neg_reward_average",
        colors={"flatdqn": "green", "sfols": "red", "lof": "blue"},
        xlabel="Total Timestep",
        ylabel=f"Mean Neg. FSA Reward",
        title=f"Mean Neg. Reward over {n_tasks} FSA tasks (average) ±1 STD",
        std_multiplier=1.0,
        save=save,
        timestamp=timestamp
    )
    plot_metric_across_runs(
        dfs,
        x_axis="learning/total_timestep",
        ycol="learning/fsa_reward_average",
        colors={"flatdqn": "green", "sfols": "red", "lof": "blue"},
        xlabel="Total Timestep",
        ylabel=f"Mean FSA Reward",
        title=f"Mean Reward over {n_tasks} FSA tasks (average) ±1 STD",
        std_multiplier=1.0,
        save=save,
        timestamp=timestamp
    )


if __name__ == "__main__":
    main()
