import json
import random, os
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
