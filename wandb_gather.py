import fnmatch
import pandas as pd
import wandb
import hydra
from omegaconf import DictConfig


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
              compute_average_from_fsa_reward=False, stretch_x_axis=False):
    if len(run_ids) < 1:
        return None
    
    run_dfs = {}
    for run_id in run_ids:
        run_path = f"{entity}/{project}/{run_id}"
        run      = api.run(run_path)

        run_df = pull_data(run=run, x_axis=x_axis, patterns=patterns)

        if compute_average_from_fsa_reward:
            # Get average run_df performance over tasks
            fsa_reward_cols = [col for col in run_df.columns if fnmatch.fnmatch(col, patterns[0])]
            fsa_neg_reward_cols = [col for col in run_df.columns if fnmatch.fnmatch(col, patterns[1])]
            run_df = average_cols(run_df, cols=fsa_reward_cols, name="learning/fsa_reward_average")
            run_df = average_cols(run_df, cols=fsa_neg_reward_cols, name="learning/fsa_neg_reward_average")

        if stretch_x_axis:
            # Strech x_axis by n tasks
            run_df[x_axis] *= len(fsa_neg_reward_cols)

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
    return final_df


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
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
                        x_axis="learning/timestep", compute_average_from_fsa_reward=True, stretch_x_axis=True)
    flatdqn = flatdqn.rename(columns={"learning/timestep": x_axis})
    sfols = pull_runs(api, sfols_run_ids, cfg.wandb.entity, cfg.wandb.project, patterns, 
                      x_axis=x_axis, compute_average_from_fsa_reward=False, stretch_x_axis=False)
    lof = pull_runs(api, lof_run_ids, cfg.wandb.entity, cfg.wandb.project, patterns, 
                    x_axis=x_axis, compute_average_from_fsa_reward=False, stretch_x_axis=False)
    
    print(flatdqn)
    print(sfols)
    print(lof)


if __name__ == "__main__":
    main()
