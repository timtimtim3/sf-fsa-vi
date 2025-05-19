import wandb as wb
import hydra
import envs
import gym
import os
from copy import deepcopy
from fsa.tasks_specification import load_fsa
from omegaconf import DictConfig, OmegaConf
from envs.wrappers import FlatQEnvWrapper
from utils.utils import seed_everything, setup_run_dir, save_config
from hydra.utils import instantiate, get_class

EVAL_EPISODES = 20
n_iters = 10


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    fsa_symbols_from_env = cfg.get("fsa_symbols_from_env", False)
    dir_date_postfix = cfg.get("dir_postfix", None)

    # disable WANDB logging
    wb.init(mode="disabled")

    seed_everything(cfg.seed)

    # Create the train and eval environments
    env_params = dict(cfg.env)
    env_params.pop("env_name")

    train_env_kwargs = deepcopy(env_params)
    train_env_kwargs.pop("restriction")
    train_env_kwargs.pop("planning_constraint")

    excluded_keys = {"add_obj_to_start", "add_empty_to_start", "reset_probability_goals"}
    eval_env_kwargs = {k: v for k, v in train_env_kwargs.items() if k not in excluded_keys}

    # env setup
    env_name = cfg.env.env_name
    train_env = gym.make(env_name, **train_env_kwargs)
    eval_env = gym.make(env_name, **eval_env_kwargs)

    # Directory for storing the policies
    directory = train_env.unwrapped.spec.id
    if dir_date_postfix is not None:
        dir_date_postfix = "-" + dir_date_postfix
        directory += dir_date_postfix
    base_save_dir = f"results/flatq/{directory}"
    save_config(cfg, base_dir=base_save_dir, type='play')

    # Create the FSA env wrapper, to evaluate the FSA
    fsa, T = load_fsa('-'.join([env_name, cfg.fsa_name]), eval_env,
                      fsa_symbols_from_env=fsa_symbols_from_env)  # Load FSA
    eval_env = FlatQEnvWrapper(eval_env, fsa, fsa_init_state="u0")
    train_env = FlatQEnvWrapper(train_env, fsa, fsa_init_state="u0")
    n_fsa_states = fsa.num_states

    # Load the class object (either FlatQ or DQNContinuousFSA, etc)
    AlgClass = get_class(cfg.algorithm._target_)

    # Now call its `load(...)`:
    agent = AlgClass.load(
        env=train_env,
        eval_env=eval_env,
        n_fsa_states=n_fsa_states,
        path=base_save_dir,
        **{k: v for k, v in cfg.algorithm.items() if k != "_target_"}
    )

    agent.plot_q_vals(base_dir=base_save_dir, show=True)

    wb.finish()


if __name__ == "__main__":
    main()
