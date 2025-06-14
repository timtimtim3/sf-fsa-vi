import numpy as np

import wandb as wb
import hydra
import envs
import gym
import os
import importlib
from copy import deepcopy
from sfols.rl.utils.utils import policy_eval_exact
from sfols.rl.successor_features.gpi import GPI
from sfols.rl.successor_features.ols import OLS
from fsa.tasks_specification import load_fsa
from omegaconf import DictConfig, OmegaConf, ListConfig
from envs.wrappers import GridEnvWrapper
from utils.utils import get_base_save_dir, seed_everything, do_planning, setup_run_dir
from sfols.plotting.plotting import plot_all_rbfs, plot_all_fourier, plot_gpi_qvals
from envs.utils import get_rbf_activation_data, get_fourier_activation_data

EVAL_EPISODES = 20
n_iters = 10


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    value_iter_type = cfg.get("value_iter_type", None)
    learn_all_extremum = cfg.get("learn_all_extremum", False)
    subtract_constant = cfg.get("subtract_constant", None)
    use_regular_gpi_exec = cfg.get("use_regular_gpi_exec", True)
    fsa_symbols_from_env = cfg.get("fsa_symbols_from_env", False)
    learn_weights = cfg.get("learn_weights", None)
    dir_postfix = cfg.get("dir_postfix", None)
    use_batch_dir = cfg.get("use_batch_dir", False)
    batch_dir_postfix = cfg.get("batch_dir_postfix", None)
    batch_run_name = cfg.get("batch_run_name", None)
    os.environ["WANDB_SYMLINKS"] = "False"

    # Init Wandb
    run = wb.init(
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        entity=cfg.wandb.entity, project=cfg.wandb.project,
        group="sfols", tags=["sfols"],
        # mode = "disabled"

    )
    run.tags = run.tags
    using_dqn = ("SFDQN" in cfg.algorithm["_target_"])

    # Set seeds
    seed_everything(cfg.seed)

    # Create the train and eval environments
    env_params = dict(cfg.env)
    env_name = env_params.pop("env_name")
    env_level_name = "" if "level_name" not in env_params else env_params["level_name"]

    train_env_kwargs = deepcopy(env_params)
    train_env_kwargs.pop("restriction")
    train_env_kwargs.pop("planning_constraint")

    excluded_keys = {"add_obj_to_start", "add_empty_to_start", "reset_probability_goals"}
    eval_env_kwargs = {k: v for k, v in train_env_kwargs.items() if k not in excluded_keys}

    train_env = gym.make(env_name, **train_env_kwargs)
    eval_env = gym.make(env_name, **eval_env_kwargs)

    if learn_weights is not None:
        for w in learn_weights:
            assert len(w) == train_env.feat_dim

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

    eval_envs = []
    fsa_to_load = cfg.fsa_name if isinstance(cfg.fsa_name, ListConfig) else [cfg.fsa_name]
    for fsa_name in fsa_to_load:
        # Create the FSA env wrapper, to evaluate the FSA
        fsa, T = load_fsa('-'.join([env_name, fsa_name]), eval_env,
                          fsa_symbols_from_env=fsa_symbols_from_env)  # Load FSA
        fsa_env = GridEnvWrapper(eval_env, fsa, fsa_init_state="u0", T=T)
        eval_envs.append(fsa_env)

    # Define the agent constructor and gpi agent
    def agent_constructor(log_prefix: str):
        kwargs = {}
        return hydra.utils.call(config=cfg.algorithm, env=train_env, log_prefix=log_prefix, fsa_env=eval_envs, **kwargs)

    planning_kwargs = {}
    if subtract_constant is not None:
        planning_kwargs["subtract_constant"] = subtract_constant
    if psis_are_augmented:
        planning_kwargs["set_non_goal_zero"] = True
    gpi_agent = GPI(train_env,
                    agent_constructor,
                    **cfg.gpi.init,
                    fsa_env=eval_envs,
                    psis_are_augmented=psis_are_augmented,
                    planning_constraint=cfg.env.planning_constraint,
                    ValueIteration=ValueIteration,
                    planning_kwargs=planning_kwargs)

    # m = number of predicates
    # Need to add the constraint, which sets add some restriction to the extrema weights.
    ols = OLS(m=train_env.feat_dim, **cfg.ols, restriction=cfg.env.restriction)

    base_save_dir = get_base_save_dir(train_env, dir_postfix, use_batch_dir, batch_run_name, batch_dir_postfix, 
                                      method="sfols")
    setup_run_dir(base_save_dir, cfg, run_name=run.name, run_id=run.id)

    unique_symbol_for_centers = False
    grid_size = train_env.MAP.shape
    if "rbf" in env_level_name:
        activation_data, _ = get_rbf_activation_data(train_env)
        plot_all_rbfs(activation_data, grid_size, train_env, skip_non_goal=False, save_dir=base_save_dir)
        unique_symbol_for_centers = True
    elif "fourier" in env_level_name:
        activation_data, _ = get_fourier_activation_data(train_env)
        plot_all_fourier(activation_data, grid_size, train_env, save_dir=base_save_dir)
    else:
        activation_data = None

    def learn_loop(w):
        gpi_agent.learn(w=w, reuse_value_ind=ols.get_set_max_policy_index(w), **cfg.gpi.learn)
        value = policy_eval_exact(agent=gpi_agent, env=train_env, w=w,
                                  using_dqn=using_dqn)  # Do the expectation analytically
        # Value here is the average SF over initial starting states
        # under the current GPI policy under current w=w (including the policy that was just learned)
        remove_policies = ols.add_solution(value, w, gpi_agent=gpi_agent, env=train_env,
                                           learn_all_extremum=learn_all_extremum)
        gpi_agent.delete_policies(remove_policies)

        gpi_agent.save_policies(base_save_dir)
        gpi_agent.plot_q_vals(activation_data, base_save_dir, unique_symbol_for_centers=unique_symbol_for_centers,
                              show=False)

        gpi_agent.save_tasks(base_save_dir, as_json=True, as_pickle=True)

    if learn_weights is not None:
        for w in learn_weights:
            print(f"Training pre-defined weights {w}")
            learn_loop(np.array(w, dtype=np.float64))
    else:
        for ols_iter in range(cfg.max_iter_ols):
            print(f"ols_iter: {ols_iter}")

            if ols.ended():
                print("ended at iteration", ols_iter)
                break

            w = ols.next_w()
            print(f"Training {w}")

            learn_loop(w)

    # DONE
    gpi_agent.save_policies(base_save_dir)
    gpi_agent.plot_q_vals(activation_data, base_save_dir, unique_symbol_for_centers=unique_symbol_for_centers,
                          show=False)
    gpi_agent.save_tasks(base_save_dir, as_json=True, as_pickle=True)

    # Once the low-level policies have been obtained we can retrain the high-level
    # policy and keep track of the results.
    run.summary["policies_obtained"] = len(gpi_agent.policies)
    wb.define_metric("evaluation/acc_reward", step_metric="evaluation/iter")

    for eval_env in eval_envs:
        planning = ValueIteration(eval_env, gpi_agent, constraint=cfg.env.planning_constraint, **planning_kwargs)
        W, _ = planning.traverse(num_iters=n_iters, verbose=True)

        # # Render
        # _ = gpi_agent.evaluate(gpi_agent, eval_env, W, render=True, sleep_time=0.1)

        plot_gpi_qvals(W, gpi_agent, train_env, activation_data, fsa_name=eval_env.fsa.name,
                       unique_symbol_for_centers=unique_symbol_for_centers,
                       base_dir=base_save_dir, psis_are_augmented=not use_regular_gpi_exec)

    wb.finish()


if __name__ == "__main__":
    main()
