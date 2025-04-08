import glob
import json
import os
import importlib
from copy import deepcopy

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
from envs.utils import get_rbf_activation_data, get_fourier_activation_data
from sfols.plotting.plotting import (plot_q_vals, plot_all_rbfs, get_plot_arrow_params_from_eval, plot_maxqvals,
                                     plot_all_fourier)
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


def plot_gpi_qvals(w_dict, gpi_agent, train_env, activation_data, verbose=True, unique_symbol_for_centers=False):
    if verbose:
        print("\nPlotting GPI q-values:")
    w_arr = np.asarray(list(w_dict.values())).reshape(-1)
    for (uidx, w) in enumerate(w_dict.values()):
        if uidx == len(w_dict.keys()) - 1:
            break

        w_dot = w_arr if gpi_agent.psis_are_augmented else w

        if verbose:
            print(uidx, np.round(w, 2))
        actions, policy_indices, qvals = gpi_agent.get_gpi_policy_on_w(w_dot, uidx=uidx)
        arrow_data = get_plot_arrow_params_from_eval(actions, qvals, train_env)
        plot_q_vals(w, train_env, arrow_data=arrow_data, activation_data=activation_data,
                    policy_indices=policy_indices, unique_symbol_for_centers=unique_symbol_for_centers)


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    plot = cfg.get("plot", True)
    value_iter_type = cfg.get("value_iter_type", None)
    dir_date_postfix = cfg.get("dir_postfix", None)

    wandb.init(mode="disabled")

    # Create the train and eval environments
    env_params = dict(cfg.env)
    env_name = env_params.pop("env_name")
    env_level_name = env_params["level_name"]

    alg_params = dict(cfg.algorithm)
    gamma = alg_params.pop("gamma")
    print(gamma)

    train_env_kwargs = deepcopy(env_params)
    train_env_kwargs.pop("restriction")
    train_env_kwargs.pop("planning_constraint")

    excluded_keys = {"add_obj_to_start", "add_empty_to_start", "reset_probability_goals"}
    eval_env_kwargs = {k: v for k, v in train_env_kwargs.items() if k not in excluded_keys}

    train_env = gym.make(env_name, **train_env_kwargs)
    eval_env = gym.make(env_name, **eval_env_kwargs)

    psis_are_augmented = False if value_iter_type is None or "augmented" not in value_iter_type.lower() else True
    if value_iter_type:
        # Construct the full module path
        module_name = "fsa.planning"
        class_name = value_iter_type  # The string should match the class name

        # Dynamically import the module
        module = importlib.import_module(module_name)

        # Get the class from the module
        ValueIteration = getattr(module, class_name)
        print(f"Using {value_iter_type}")
    elif hasattr(train_env, "only_rbf") and train_env.only_rbf:
        print("Defaulting to SFFSAValueIterationAugmented")
        psis_are_augmented = True
        from fsa.planning import SFFSAValueIterationAugmented as ValueIteration
    else:
        print("Defaulting to SFFSAValueIteration")
        from fsa.planning import SFFSAValueIteration as ValueIteration

    # Create the FSA env wrapper, to evaluate the FSA
    fsa, T = load_fsa('-'.join([env_name, cfg.fsa_name]), eval_env)  # Load FSA
    eval_env = GridEnvWrapper(eval_env, fsa, fsa_init_state="u0", T=T)
    n_fsa_states = len(fsa.states)
    feat_dim = train_env.feat_dim

    # Create the GPI agent shell (no policies yet)
    def agent_constructor(log_prefix: str):
        return hydra.utils.call(config=cfg.algorithm, env=train_env, log_prefix=log_prefix, fsa_env=eval_env)

    gpi_agent = GPI(train_env,
                    agent_constructor,
                    **cfg.gpi.init,
                    psis_are_augmented=psis_are_augmented,
                    planning_constraint=cfg.env.planning_constraint)

    # -----------------------------------------------------------------------------
    # 1) LOAD PREVIOUSLY SAVED POLICIES FROM .PKL FILES
    # -----------------------------------------------------------------------------
    directory = train_env.unwrapped.spec.id
    if dir_date_postfix is not None:
        dir_date_postfix = "-" + dir_date_postfix
        directory += dir_date_postfix
    policy_dir = f"results/sfols/policies/{directory}"
    gpi_agent.load_tasks(policy_dir)

    # Load from json since pickles don't work
    q_tables = load_qtables_from_json(policy_dir)

    gpi_agent.load_policies(policy_dir, q_tables)

    unique_symbol_for_centers = False
    grid_size = train_env.MAP.shape
    if "rbf" in env_level_name:
        activation_data, _ = get_rbf_activation_data(train_env, exclude={"X"})
        plot_all_rbfs(activation_data, grid_size, train_env, skip_non_goal=False, save_dir=policy_dir)
        unique_symbol_for_centers = True
    elif "fourier" in env_level_name:
        activation_data, _ = get_fourier_activation_data(train_env)
        plot_all_fourier(activation_data, grid_size, train_env, save_dir=policy_dir)
    else:
        activation_data = None

    # -----------------------------------------------------------------------------
    # 2) PLOT ARROWS MAX Q
    # -----------------------------------------------------------------------------
    print("Loaded policy weights:")
    for i, (policy, w) in enumerate(zip(gpi_agent.policies, gpi_agent.tasks)):
        print(i, np.round(w, 2))
        if plot:
            plot_q_vals(w, train_env, q_table=policy.q_table, activation_data=activation_data,
                        unique_symbol_for_centers=unique_symbol_for_centers)
            # plot_maxqvals(w, train_env, q_table=policy.q_table, rbf_data=rbf_data)
            # row 5 col 3, 4

    # -----------------------------------------------------------------------------
    # 2) Play singular policies on the tasks they were trained on
    # -----------------------------------------------------------------------------
    # gpi_agent.evaluate_all_single_policies(train_env, render=True, verbose=True, get_stuck_max=10)
    # gpi_agent.evaluate_single_policy(env=train_env, policy_index=9, render=True, verbose=True, get_stuck_max=10)
    # [0.21, 0.07, 0.01, 0.03, 0.21, 0.13, 0.13, 0.21]

    # -----------------------------------------------------------------------------
    # 3) PERFORM VALUE ITERATION AND LET THE AGENT PLAY IN THE ENV WITH RENDER
    # -----------------------------------------------------------------------------
    print("\nPerforming value iteration...")
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

        if len(rewards) > 0:
            avg_reward = np.mean(rewards)
        else:
            avg_reward = 0

        log_dict = {
            "evaluation/acc_reward": avg_reward,
            "evaluation/iter": j,
            "evaluation/time": np.sum(times)
        }

    print("\nValue iterated weight vector: ")
    print(np.round(np.asarray(list(W.values())), 2))

    # gpi_agent.evaluate(gpi_agent, eval_env, W, render=True, sleep_time=0.1)
    # all_max_q, all_v, all_gamma_t_v_values, all_gamma_t_q_values = (
    #     gpi_agent.do_rollout(gpi_agent, eval_env, W, n_fsa_states=n_fsa_states,
    #                          feat_dim=feat_dim, gamma=gamma, render=False, sleep_time=0.1))
    # matrix = np.column_stack((all_gamma_t_q_values, all_max_q, all_gamma_t_v_values, all_v))
    # print(gamma)
    # print(np.round(matrix, 3))

    # Enable this to do GPI like in original paper
    # gpi_agent.psis_are_augmented = False

    plot_gpi_qvals(W, gpi_agent, train_env, activation_data, unique_symbol_for_centers=unique_symbol_for_centers)

    train_env.close()
    eval_env.close()  # Close the environment when done
    wb.finish()


if __name__ == "__main__":
    main()
