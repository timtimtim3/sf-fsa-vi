import json
import random, os
import shutil
from typing import List, Optional

import torch as th
import numpy as np 
import os

from omegaconf import OmegaConf


def __init__():
    pass


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = True


def save_config(cfg, base_dir, type='run'):
    # save the full config as YAML
    OmegaConf.save(
        config=cfg,
        f=os.path.join(base_dir, f"{type}_config.yaml"),
        resolve=True,  # will interpolate any ${...} nodes
    )

    # convert the config to plain Python containers
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # write it out as JSON
    json_path = os.path.join(base_dir, f"{type}_config.json")
    with open(json_path, "w") as fp:
        json.dump(cfg_dict, fp, indent=2)


def save_wandb_run_name(
        base_dir: str,
        run_name: str,
        history: Optional[List[str]] = None,
        filename: str = "wandb_runs.txt"
):
    """
    Ensures `base_dir` exists and writes out the full run-history:
    first any names in `history`, then the current `run_name`.

    Args:
        base_dir:   folder to create/ensure
        run_name:   the newly-started WandB run name
        history:    list of past run names to preserve (in order)
        filename:   the file under base_dir to write into
    """
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, filename)

    # Default to empty list if no history provided
    history = history or []

    # Write all old names, then the new one
    with open(path, "w") as f:
        for name in history:
            f.write(f"{name}\n")
        f.write(f"{run_name}\n")


def setup_run_dir(base_save_dir, cfg, run_name = None):
    run_name_history = read_wandb_run_history(base_save_dir)
    shutil.rmtree(base_save_dir, ignore_errors=True)
    os.makedirs(base_save_dir, exist_ok=True)
    save_config(cfg, base_dir=base_save_dir, type='run')
    save_wandb_run_name(base_save_dir, run_name, history=run_name_history)


def read_wandb_run_history(base_dir: str,
                           filename: str = "wandb_runs.txt"
                          ) -> List[str]:
    """
    Reads and returns the list of previous WandB run names
    from `<base_dir>/<filename>`.  If the file doesn’t exist,
    returns an empty list.
    """
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        # strip newline characters
        return [line.strip() for line in f.readlines()]


def do_planning(planning, gpi_agent, eval_env, wb=None, n_iters=5, eval_episodes=1, use_regular_gpi_exec=True):
    W = None
    times = []

    for j in range(n_iters):
        print(f'Iter: {j}')

        if W is not None:
            w_old_arr = np.asarray(list(W.values())).reshape(-1)
        else:
            w_old_arr = None

        W, time = planning.traverse(W, num_iters=1)
        times.append(time)

        w_arr = np.asarray(list(W.values())).reshape(-1)

        rewards = []
        for _ in range(eval_episodes):
            acc_reward = gpi_agent.evaluate(gpi_agent, eval_env, W, psis_are_augmented=not use_regular_gpi_exec)
            rewards.append(acc_reward)

        if len(rewards) > 0:
            avg_reward = np.mean(rewards)
        else:
            avg_reward = 0

        log_dict = {
            "evaluation/acc_reward": avg_reward,
            "evaluation/iter": j,
            "evaluation/time": np.sum(times)
        }
        if wb is not None:
            wb.log(log_dict)

        if w_old_arr is not None:
            # Compute normalized difference between old and new weights
            diff = np.linalg.norm(w_arr - w_old_arr) / w_arr.size
            print(f"Normalized weight diff: {diff:.6f}")

        # Check for convergence (early stopping)
        if w_old_arr is not None and np.allclose(w_arr, w_old_arr):
            print(f"Stopping early at iter {j}")
            break
    return W
