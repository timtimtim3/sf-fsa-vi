import wandb as wb
import hydra
import envs
import gym
import os
from copy import deepcopy
from fsa.tasks_specification import load_fsa
from omegaconf import DictConfig, OmegaConf, ListConfig
from envs.wrappers import FlatQEnvWrapper
from utils.utils import seed_everything, setup_run_dir
from hydra.utils import instantiate

EVAL_EPISODES = 20
n_iters = 10


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    fsa_symbols_from_env = cfg.get("fsa_symbols_from_env", False)
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

    train_env_kwargs = deepcopy(env_params)
    train_env_kwargs.pop("restriction")
    train_env_kwargs.pop("planning_constraint")

    excluded_keys = {"add_obj_to_start", "add_empty_to_start", "reset_probability_goals"}
    eval_env_kwargs = {k: v for k, v in train_env_kwargs.items() if k not in excluded_keys}

    train_env = gym.make(env_name, **train_env_kwargs)
    eval_env = gym.make(env_name, **eval_env_kwargs)

    # Directory for storing the policies
    directory = train_env.unwrapped.spec.id
    base_save_dir = f"results/flatq/{directory}"
    setup_run_dir(base_save_dir, cfg, run_name=run.name)

    eval_envs = []
    train_envs = []
    fsa_to_load = cfg.fsa_name if isinstance(cfg.fsa_name, ListConfig) else [cfg.fsa_name]
    for fsa_name in fsa_to_load:
        # Create the FSA env wrapper, to evaluate the FSA
        fsa, T = load_fsa('-'.join([env_name, fsa_name]), eval_env,
                          fsa_symbols_from_env=fsa_symbols_from_env)  # Load FSA
        wrapped_eval_env = FlatQEnvWrapper(eval_env, fsa, fsa_init_state="u0")
        wrapped_train_env = FlatQEnvWrapper(train_env, fsa, fsa_init_state="u0")
        eval_envs.append(wrapped_eval_env)
        train_envs.append(wrapped_train_env)

    for train_env, eval_env in zip(train_envs, eval_envs):
        agent = instantiate(
            cfg.algorithm,
            env=train_env,
            eval_env=eval_env,
            log_prefix=f"{eval_env.fsa.name}/"
        )
        agent.learn(
            total_timesteps=cfg.algorithm.total_timesteps,
            eval_freq=cfg.algorithm.eval_freq
        )
        agent.save(base_save_dir)
        agent.plot_q_vals(base_dir=base_save_dir, show=True)

    wb.finish()


if __name__ == "__main__":
    main()
