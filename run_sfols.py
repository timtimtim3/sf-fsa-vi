import json

from sfols.rl.utils.utils import policy_eval_exact
from sfols.rl.successor_features.gpi import GPI
from sfols.rl.successor_features.ols import OLS
from fsa.tasks_specification import load_fsa
from omegaconf import DictConfig, OmegaConf
from envs.wrappers import GridEnvWrapper

from utils.utils import seed_everything 

import pickle as pkl
import numpy as np
import wandb as wb
import shutil
import hydra
import envs
import gym
import os
from sfols.plotting.plotting import plot_q_vals
from envs.utils import get_rbf_activation_data


EVAL_EPISODES = 20


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    os.environ["WANDB_SYMLINKS"] = "False"

    # Init Wandb
    run = wb.init(
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        entity=cfg.wandb.entity, project=cfg.wandb.project,
        group="sfols", tags=["sfols"],
        # mode = "disabled"

    )
    run.tags = run.tags

    # Set seeds
    seed_everything(cfg.seed)

    # Create the train and eval environments
    env_params = dict(cfg.env)
    env_name = env_params.pop("env_name")
    add_obj_to_start = env_params.get("add_obj_to_start")

    if "RBFOnly" in env_name:
        from fsa.planning import SFFSAValueIterationAreasRBFOnly as ValueIteration
    else:
        from fsa.planning import SFFSAValueIteration as ValueIteration

    # Default to env defaults if not specified
    kwargs = {
        k: v for k, v in {
            "add_empty_to_start": env_params.get("add_empty_to_start"),
            "only_rbf": env_params.get("only_rbf")
        }.items() if v is not None
    }

    train_env = gym.make(env_name, add_obj_to_start=True if add_obj_to_start is None else add_obj_to_start,
                         **kwargs)
    eval_env = gym.make(env_name)

    # Create the FSA env wrapper, to evaluate the FSA
    fsa, T = load_fsa('-'.join([env_name, cfg.fsa_name]), eval_env) # Load FSA
    eval_env = GridEnvWrapper(eval_env, fsa, fsa_init_state="u0", T=T)

    # Define the agent constructor and gpi agent
    def agent_constructor(log_prefix: str):
        return hydra.utils.call(config=cfg.algorithm, env=train_env, log_prefix=log_prefix, fsa_env=eval_env)

    gpi_agent = GPI(train_env,
                    agent_constructor,
                    **cfg.gpi.init,
                    planning_constraint=cfg.env.planning_constraint)

    # m = number of predicates
    # Need to add the constraint, which sets add some restriction to the extrema weights.
    ols = OLS(m=train_env.feat_dim, **cfg.ols, restriction=cfg.env.restriction)

    # Directory for storing the policies
    directory = train_env.unwrapped.spec.id
    base_save_dir = f"results/sfols/policies/{directory}"
    shutil.rmtree(base_save_dir, ignore_errors=True)
    os.makedirs(base_save_dir, exist_ok=True)

    if "RBF" in env_name:
        rbf_data, grid_size = get_rbf_activation_data(train_env, exclude={"X"})

    for ols_iter in range(cfg.max_iter_ols):
        print(f"ols_iter: {ols_iter}")
        
        if ols.ended():
            print("ended at iteration", ols_iter)
            break
       
        w = ols.next_w()
        print(f"Training {w}")

        # gpi_agent.learn(w=w, **cfg.gpi.learn)
        gpi_agent.learn(w=w, reuse_value_ind=ols.get_set_max_policy_index(w), **cfg.gpi.learn)
        value = policy_eval_exact(agent=gpi_agent, env=train_env, w=w) # Do the expectation analytically
        remove_policies = ols.add_solution(value, w, gpi_agent=gpi_agent, env=train_env)
        gpi_agent.delete_policies(remove_policies)

        # Save Q-tables as .json files
        for i, (policy, w) in enumerate(zip(gpi_agent.policies, gpi_agent.tasks)):
            q_table_serializable = {
                str(k): v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in policy.q_table.items()
            }

            q_table_path = f"{base_save_dir}/qtable_pol{i}.json"
            with open(q_table_path, "w") as f:
                json.dump(q_table_serializable, f, indent=4)

            # plot and save q-vals
            plot_q_vals(i, policy.q_table, w, train_env, rbf_data, save_path=f"{base_save_dir}/qvals_pol{i}.png",
                        show=False)

    # Save Q-tables as .json files
    for i, (policy, w) in enumerate(zip(gpi_agent.policies, gpi_agent.tasks)):
        q_table_serializable = {
            str(k): v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in policy.q_table.items()
        }

        q_table_path = f"{base_save_dir}/qtable_pol{i}.json"
        with open(q_table_path, "w") as f:
            json.dump(q_table_serializable, f, indent=4)

        # plot and save q-vals
        plot_q_vals(i, policy.q_table, w, train_env, rbf_data, save_path=f"{base_save_dir}/qvals_pol{i}.png", show=False)

    # Save policies as .pkl files
    for i, pi in enumerate(gpi_agent.policies):
        d = vars(pi)
        d.pop("replay_buffer")
        d.pop("env")
        d.pop("gpi")

        policy_path = f"results/sfols/policies/{train_env.unwrapped.spec.id}/discovered_policy_{i + 1}.pkl"
        with open(policy_path, "wb") as fp:
            pkl.dump(d, fp)
        # wb.save(f"results/sfols/policies/{train_env.unwrapped.spec.id}/discovered_policy_{i + 1}.pkl")

        # Create an artifact
        artifact = wb.Artifact(
            name=f"policy_{train_env.unwrapped.spec.id}_{i + 1}",  # Unique artifact name
            type="policy-data",  # Define the type (model, dataset, etc.)
            description="Learned policy for the environment"
        )

        # Add the policy file to the artifact
        artifact.add_file(policy_path)

        # Log the artifact to the current run (uploading it to the cloud)
        wb.run.log_artifact(artifact)

        run.summary["policies_obtained"] = len(gpi_agent.policies)

    tasks = gpi_agent.tasks
    tasks_path = f"results/sfols/policies/{train_env.unwrapped.spec.id}/tasks.pkl"
    with open(tasks_path, "wb") as fp:
        pkl.dump(tasks, fp)

    # Once the low-level policies have been obtained we can retrain the high-level
    # policy and keep track of the results.

    wb.define_metric("evaluation/acc_reward", step_metric="evaluation/iter")

    planning = ValueIteration(eval_env, gpi_agent, constraint=cfg.env.planning_constraint)
    W = None

    times = []

    for j in range(50):

        W, time = planning.traverse(W, num_iters = 1)
        times.append(time)
        rewards = []
        for _ in range(EVAL_EPISODES):
            acc_reward = gpi_agent.evaluate(gpi_agent, eval_env, W)
            rewards.append(acc_reward)
            
        log_dict = {"evaluation/acc_reward": np.average(rewards),
                    "evaluation/iter": j,
                    "evaluation/time": np.sum(times)}
        
        wb.log(log_dict)

    acc_reward = gpi_agent.evaluate(gpi_agent, eval_env, W, render=True)

    wb.finish()



if __name__ == "__main__":
    main()
