import glob
import json
import os

from sfols.rl.successor_features.gpi import GPI
from fsa.tasks_specification import load_fsa
from omegaconf import DictConfig
from envs.wrappers import GridEnvWrapper
import numpy as np
import wandb as wb
import hydra
import envs
import gym
import wandb
import matplotlib.pyplot as plt
from envs.utils import get_rbf_activation_data
from sfols.plotting.plotting import plot_q_vals, plot_all_rbfs, get_plot_arrow_params_from_eval
import pickle as pkl

EVAL_EPISODES = 20


def load_qtables_from_json(policy_dir: str):
    """
    Loads saved qtables from json files

    Parameters:
        policy_dir (str): The directory where the policy json files are
    """
    q_tables = {}  # Dictionary to store loaded policies
    # List all JSON files in the policy directory that match 'qtable_polX.json'
    for filename in sorted(os.listdir(policy_dir)):  # Sort ensures policies are loaded in order
        if filename.startswith("qtable_pol") and filename.endswith(".json"):
            policy_index = int(filename.split("qtable_pol")[1].split(".json")[0])  # Extract policy index
            q_table_path = os.path.join(policy_dir, filename)

            with open(q_table_path, "r") as f:
                q_table_serialized = json.load(f)

            # Convert back: keys from str to tuple, values from list to np.array (if necessary)
            q_table_original = {
                eval(k): np.array(v) if isinstance(v, list) else v
                for k, v in q_table_serialized.items()
            }

            q_tables[policy_index] = q_table_original  # Store in dict with policy index as key
    return q_tables

def load_qtables_from_pickles(policy_dir: str):
    """
    Loads saved qtables from pickle files

    Parameters:
        policy_dir (str): The directory where the policy pickle files are
    """

    # Load policy pickle files from the directory
    pkl_files = sorted(glob.glob(os.path.join(policy_dir, "discovered_policy_*.pkl")))
    print(f"Loading {len(pkl_files)} policies from {policy_dir}")
    q_tables = {}
    for i, pkl_path in enumerate(pkl_files):
        with open(pkl_path, "rb") as fp:
            policy_data = pkl.load(fp)
        q_tables[i] = policy_data["q_table"]
    return q_tables

@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    wandb.init(mode="disabled")

    # Create the train and eval environments
    env_params = dict(cfg.env)
    env_name = env_params.pop("env_name")

    # Default to env defaults if not specified
    train_env_kwargs = {
        k: v for k, v in {
            "add_obj_to_start": env_params.get("add_obj_to_start"),
            "add_empty_to_start": env_params.get("add_empty_to_start"),
            "level_name": env_params.get("level_name"),
            "only_rbf": env_params.get("only_rbf")
        }.items() if v is not None
    }
    excluded_keys = {"add_obj_to_start", "add_empty_to_start"}
    eval_env_kwargs = {k: v for k, v in train_env_kwargs.items() if k not in excluded_keys}

    train_env = gym.make(env_name, **train_env_kwargs)
    eval_env = gym.make(env_name, **eval_env_kwargs)

    if train_env.only_rbf:
        from fsa.planning import SFFSAValueIterationAreasRBFOnly as ValueIteration
    else:
        from fsa.planning import SFFSAValueIteration as ValueIteration

    # Create the FSA env wrapper, to evaluate the FSA
    fsa, T = load_fsa('-'.join([env_name, cfg.fsa_name]), eval_env) # Load FSA
    eval_env = GridEnvWrapper(eval_env, fsa, fsa_init_state="u0", T=T)

    # Create the GPI agent shell (no policies yet)
    def agent_constructor(log_prefix: str):
        return hydra.utils.call(config=cfg.algorithm, env=train_env, log_prefix=log_prefix, fsa_env=eval_env)

    gpi_agent = GPI(train_env,
                    agent_constructor,
                    **cfg.gpi.init,
                    planning_constraint=cfg.env.planning_constraint)

    # -----------------------------------------------------------------------------
    # 1) LOAD PREVIOUSLY SAVED POLICIES FROM .PKL FILES
    # -----------------------------------------------------------------------------
    dir_date_postfix = cfg.get("dir_postfix", "")
    if dir_date_postfix:
        dir_date_postfix = "-" + dir_date_postfix
    directory = train_env.unwrapped.spec.id + dir_date_postfix
    policy_dir = f"results/sfols/policies/{directory}"
    gpi_agent.load_tasks(policy_dir)

    # Load from json since pickles don't work
    q_tables = load_qtables_from_json(policy_dir)

    gpi_agent.load_policies(policy_dir, q_tables)

    if "RBF" in env_name:
        rbf_data, grid_size = get_rbf_activation_data(train_env, exclude={"X"})
        # plot_all_rbfs(rbf_data, grid_size, train_env)

    # -----------------------------------------------------------------------------
    # 2) PLOT ARROWS MAX Q
    # -----------------------------------------------------------------------------
    # for i, (policy, w) in enumerate(zip(gpi_agent.policies, gpi_agent.tasks)):
    #     print(i, w)
    #     plot_q_vals(w, train_env, q_table=policy.q_table, rbf_data=rbf_data)

    # -----------------------------------------------------------------------------
    # 2) Play singular policies on the tasks they were trained on
    # -----------------------------------------------------------------------------
    # gpi_agent.evaluate_all_single_policies(train_env, render=True, verbose=True, get_stuck_max=10)
    # gpi_agent.evaluate_single_policy(env=train_env, policy_index=9, render=True, verbose=True, get_stuck_max=10)
    # [0.21, 0.07, 0.01, 0.03, 0.21, 0.13, 0.13, 0.21]

    # -----------------------------------------------------------------------------
    # 3) PERFORM VALUE ITERATION AND LET THE AGENT PLAY IN THE ENV WITH RENDER
    # -----------------------------------------------------------------------------
    print("Performing value iteration")
    planning = ValueIteration(eval_env, gpi_agent, constraint=cfg.env.planning_constraint)
    W = None
    times = []

    for j in range(50):
        W, time = planning.traverse(W, num_iters=1)
        times.append(time)

        rewards = []
        for _ in range(EVAL_EPISODES):
            acc_reward = gpi_agent.evaluate(gpi_agent, eval_env, W)
            rewards.append(acc_reward)

        avg_reward = np.mean(rewards)
        log_dict = {
            "evaluation/acc_reward": avg_reward,
            "evaluation/iter": j,
            "evaluation/time": np.sum(times)
        }

    # final_reward = gpi_agent.evaluate(gpi_agent, eval_env, W, render=True)
    # print(f"Final reward (rendered): {final_reward}")

    for (i, w) in enumerate(W.values()):
        if i == len(W.keys()) - 1:
            break

        print(i, w)
        actions, policy_indices, qvals = gpi_agent.get_gpi_policy_on_w(w)
        arrow_data = get_plot_arrow_params_from_eval(actions, qvals, train_env)
        plot_q_vals(w, train_env, arrow_data=arrow_data, rbf_data=rbf_data, policy_indices=policy_indices)

    train_env.close()
    eval_env.close()  # Close the environment when done
    wb.finish()


if __name__ == "__main__":
    main()
