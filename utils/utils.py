import json
import random, os
import shutil
from copy import deepcopy
from typing import List, Tuple, Optional
from omegaconf import OmegaConf
import torch as th
import numpy as np 
from fsa.planning import get_indicator_props_enable_transition


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


def read_wandb_run_history(
    base_dir: str,
    filename: str = "wandb_runs.txt"
) -> List[Tuple[str, str]]:
    """
    Reads and returns a list of (run_name, run_id) tuples from `<base_dir>/<filename>`.
    Lines that contain only `run_name` (no comma) are interpreted as `(run_name, "")`.
    If the file doesn’t exist, returns an empty list.
    """
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        return []

    history: List[Tuple[str, str]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "," in line:
                parts = line.split(",", 1)
                name, rid = parts[0], parts[1]
            else:
                name, rid = line, ""
            history.append((name, rid))
    return history


def save_wandb_run_name(
    base_dir: str,
    run_name: str,
    run_id: str,
    history: List[Tuple[str, str]],
    filename: str = "wandb_runs.txt"
):
    """
    Ensures `base_dir` exists and writes out the full run-history to `<filename>`.
    Each entry in `history` is a tuple (old_name, old_id).  Then the new (run_name, run_id)
    is appended.  Lines are written as:
      - "name,id"   if id != ""
      - "name"      if id == ""
    """
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, filename)

    with open(path, "w") as f:
        for old_name, old_id in history:
            if old_id:
                f.write(f"{old_name},{old_id}\n")
            else:
                f.write(f"{old_name}\n")
        # write the new run last
        if run_id:
            f.write(f"{run_name},{run_id}\n")
        else:
            f.write(f"{run_name}\n")


def setup_run_dir(
    base_save_dir: str,
    cfg,
    run_name: Optional[str] = None,
    run_id: Optional[str] = None
):
    """
    1) Reads prior (run_name, run_id) history from wandb_runs.txt (handling old lines that
       may contain only run_name).
    2) Deletes and recreates base_save_dir.
    3) Saves the Hydra config to YAML/JSON.
    4) Writes updated history including the new (run_name, run_id).
    """
    prev_history = read_wandb_run_history(base_save_dir)
    shutil.rmtree(base_save_dir, ignore_errors=True)
    os.makedirs(base_save_dir, exist_ok=True)

    # Save the full Hydra config under base_save_dir
    save_config(cfg, base_dir=base_save_dir, type="run")

    # Append the new (run_name, run_id) to wandb_runs.txt
    save_wandb_run_name(
        base_dir=base_save_dir,
        run_name=run_name or "",
        run_id=run_id or "",
        history=prev_history
    )


def do_planning(planning, gpi_agent, eval_env, wb=None, n_iters=5, eval_episodes=1, use_regular_gpi_exec=True,
                set_non_goal_zero=False, **kwargs):
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

    if set_non_goal_zero:
        W_new = deepcopy(W)
        for fsa_state, w_arr in list(W.items())[:-1]:
            indicator = get_indicator_props_enable_transition(w_arr, fsa_state, eval_env.fsa, eval_env.env)
            w_arr *= indicator
            W_new[fsa_state] = w_arr
        W = W_new
    return W
