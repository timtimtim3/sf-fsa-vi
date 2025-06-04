import numpy as np
import wandb as wb
import hydra
import gym
import importlib
from copy import deepcopy
from sfols.rl.successor_features.gpi import GPI
from fsa.tasks_specification import load_fsa
from omegaconf import DictConfig, ListConfig
import envs
from envs.wrappers import GridEnvWrapper
from envs.utils import get_rbf_activation_data, get_fourier_activation_data
from sfols.plotting.plotting import plot_all_rbfs, plot_all_fourier, plot_gpi_qvals, plot_trajectories
from utils.utils import save_config, do_planning


EVAL_EPISODES = 5
n_iters = 10


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    plot = cfg.get("plot", True)
    plot_trajs = cfg.get("plot_trajs", False)
    plot_qvals = cfg.get("plot_qvals", True)

    value_iter_type = cfg.get("value_iter_type", None)
    subtract_constant = cfg.get("subtract_constant", None)
    use_regular_gpi_exec = cfg.get("use_regular_gpi_exec", True)
    fsa_symbols_from_env = cfg.get("fsa_symbols_from_env", False)

    dir_date_postfix = cfg.get("dir_postfix", None)

    wb.init(mode="disabled")

    # Create the train and eval environments
    env_params = dict(cfg.env)
    env_name = env_params.pop("env_name")
    env_level_name = "" if "level_name" not in env_params else env_params["level_name"]

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

    eval_envs = []
    fsa_to_load = cfg.fsa_name if isinstance(cfg.fsa_name, ListConfig) else [cfg.fsa_name]
    for fsa_name in fsa_to_load:
        # Create the FSA env wrapper, to evaluate the FSA
        fsa, T = load_fsa('-'.join([env_name, fsa_name]), eval_env,
                          fsa_symbols_from_env=fsa_symbols_from_env)  # Load FSA
        fsa_env = GridEnvWrapper(eval_env, fsa, fsa_init_state="u0", T=T)
        eval_envs.append(fsa_env)

    # Create the GPI agent shell (no policies yet)
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

    # -----------------------------------------------------------------------------
    # 1) LOAD PREVIOUSLY SAVED POLICIES FROM .PKL FILES
    # -----------------------------------------------------------------------------
    directory = train_env.unwrapped.spec.id
    if dir_date_postfix is not None:
        dir_date_postfix = "-" + dir_date_postfix
        directory += dir_date_postfix
    base_save_dir = f"results/sfols/{directory}"
    save_config(cfg, base_dir=base_save_dir, type='play')

    gpi_agent.load_tasks(base_save_dir)
    gpi_agent.load_policies(base_save_dir)

    unique_symbol_for_centers = False
    grid_size = train_env.MAP.shape
    if "rbf" in env_level_name:
        activation_data, _ = get_rbf_activation_data(train_env, exclude={"X"})
        plot_all_rbfs(activation_data, grid_size, train_env, skip_non_goal=False, save_dir=base_save_dir)
        unique_symbol_for_centers = True
    elif "fourier" in env_level_name:
        activation_data, _ = get_fourier_activation_data(train_env)
        plot_all_fourier(activation_data, grid_size, train_env, save_dir=base_save_dir)
    else:
        activation_data = None

    # print(gpi_agent.evaluate_fsa(eval_env, render=True, base_dir=base_save_dir))

    # ROLLOUT
    w = gpi_agent.tasks[0]
    policy = gpi_agent.policies[0]
    # state = train_env.reset()
    # q_val = policy.q_values(state, w)
    # action = th.argmax(q_val, dim=1).item()
    #
    # action_2 = policy.eval(state, w)
    #
    # actions, q_max = policy.best_actions_and_q(state, w)
    # print(q_val, action)
    # print(action_2)
    # print(actions, q_max)

    # -----------------------------------------------------------------------------
    # 2) PLOT ARROWS MAX Q
    # -----------------------------------------------------------------------------
    print("Loaded policy weights:")
    for i, (policy, w) in enumerate(zip(gpi_agent.policies, gpi_agent.tasks)):
        print(i, np.round(w, 2))

    if plot:
        for i, (policy, w) in enumerate(zip(gpi_agent.policies, gpi_agent.tasks)):
            # if i == 0:
            #     state = (2.5, 6.5)
            #     print(np.round(policy.get_psis(state), 3))
            #     train_env.reset(state=state)
            #     print(train_env.step(action=4))

            if plot_trajs:
                save_path = f"{base_save_dir}/traj_{i}.png" if base_save_dir is not None else None
                trajectories = gpi_agent.policies[i].get_trajectories(w, n_trajectories=9, method="random", max_steps=40)
                plot_trajectories(train_env, trajectories, w=w, activation_data=activation_data,
                                  unique_symbol_for_centers=unique_symbol_for_centers, save_path=save_path)
            if plot_qvals:
                gpi_agent.plot_q_vals(activation_data, base_dir=base_save_dir,
                                      unique_symbol_for_centers=unique_symbol_for_centers, policy_id=i)

    # -----------------------------------------------------------------------------
    # 2) Play singular policies on the tasks they were trained on
    # -----------------------------------------------------------------------------
    # gpi_agent.evaluate_all_single_policies(train_env, render=True, verbose=True, get_stuck_max=10)
    # gpi_agent.evaluate_single_policy(env=train_env, policy_index=9, render=True, verbose=True, get_stuck_max=10)
    # [0.21, 0.07, 0.01, 0.03, 0.21, 0.13, 0.13, 0.21]

    # -----------------------------------------------------------------------------
    # 3) PERFORM VALUE ITERATION AND LET THE AGENT PLAY IN THE ENV WITH RENDER
    # -----------------------------------------------------------------------------
    for eval_env in eval_envs:
        print("\nPerforming value iteration...")
        planning = ValueIteration(eval_env, gpi_agent, constraint=cfg.env.planning_constraint, **planning_kwargs)
        W, _ = planning.traverse(num_iters=n_iters, verbose=True)
        print("\nValue iterated weight vector: ")
        print(np.round(np.asarray(list(W.values())), 2))

        # # Render
        # _ = gpi_agent.evaluate(gpi_agent, eval_env, W, render=True, sleep_time=0.1)

        plot_gpi_qvals(W, gpi_agent, train_env, activation_data, fsa_name=eval_env.fsa.name,
                       unique_symbol_for_centers=unique_symbol_for_centers,
                       base_dir=base_save_dir, psis_are_augmented=not use_regular_gpi_exec)

    wb.finish()


if __name__ == "__main__":
    main()
