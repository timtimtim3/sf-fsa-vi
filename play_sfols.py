from fsa.planning import SFFSAValueIteration as ValueIteration
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
from sfols.plotting.plotting import plot_q_vals, plot_all_rbfs

EVAL_EPISODES = 20


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    wandb.init(mode="disabled")

    # Create the train and eval environments
    env_params = dict(cfg.env)
    env_name = env_params.pop("env_name")

    # train_env = gym.make(env_name, add_obj_to_start=True if add_obj_to_start is None else add_obj_to_start)
    train_env = gym.make(env_name, add_obj_to_start=False, add_empty_to_start=False)
    eval_env = gym.make(env_name)

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
    gpi_agent.load_policies_and_tasks(policy_dir)

    if "RBF" in env_name:
        rbf_data, grid_size = get_rbf_activation_data(train_env, exclude={"X"})
        plot_all_rbfs(rbf_data, grid_size, train_env)

    # -----------------------------------------------------------------------------
    # 2) PLOT ARROWS MAX Q
    # -----------------------------------------------------------------------------
    for i, (policy, w) in enumerate(zip(gpi_agent.policies, gpi_agent.tasks)):
        print(w)
        plot_q_vals(i, policy, w, train_env, rbf_data)

    # plot_q_vals(9, gpi_agent.policies[9], gpi_agent.tasks[9], train_env, rbf_data)

    # -----------------------------------------------------------------------------
    # 2) Play singular policies on the tasks they were trained on
    # -----------------------------------------------------------------------------
    # gpi_agent.evaluate_all_single_policies(train_env, render=True, verbose=True, get_stuck_max=10)
    # gpi_agent.evaluate_single_policy(env=train_env, policy_index=9, render=True, verbose=True, get_stuck_max=10)
    # [0.21, 0.07, 0.01, 0.03, 0.21, 0.13, 0.13, 0.21]

    # -----------------------------------------------------------------------------
    # 3) PERFORM VALUE ITERATION AND LET THE AGENT PLAY IN THE ENV WITH RENDER
    # -----------------------------------------------------------------------------
    # print("Performing value iteration")
    # planning = ValueIteration(eval_env, gpi_agent, constraint=cfg.env.planning_constraint)
    # W = None
    # times = []
    #
    # for j in range(50):
    #     W, time = planning.traverse(W, num_iters=1)
    #     times.append(time)
    #
    #     rewards = []
    #     for _ in range(EVAL_EPISODES):
    #         acc_reward = gpi_agent.evaluate(gpi_agent, eval_env, W)
    #         rewards.append(acc_reward)
    #
    #     avg_reward = np.mean(rewards)
    #     log_dict = {
    #         "evaluation/acc_reward": avg_reward,
    #         "evaluation/iter": j,
    #         "evaluation/time": np.sum(times)
    #     }
    #
    # final_reward = gpi_agent.evaluate(gpi_agent, eval_env, W, render=True)
    # print(f"Final reward (rendered): {final_reward}")

    train_env.close()
    eval_env.close()  # Close the environment when done
    wb.finish()


if __name__ == "__main__":
    main()
