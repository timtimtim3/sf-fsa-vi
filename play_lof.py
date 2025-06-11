from copy import deepcopy
import gym
import hydra
import os
import wandb
from omegaconf import DictConfig, ListConfig
from envs.wrappers import GridEnvWrapper
from fsa.tasks_specification import load_fsa
from utils.utils import seed_everything, save_config


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    fsa_symbols_from_env = cfg.get("fsa_symbols_from_env", False)
    dir_date_postfix = cfg.get("dir_postfix", None)

    # disable WANDB logging
    wandb.init(mode="disabled")

    seed_everything(cfg.seed)

    env_params = dict(cfg.env)
    env_name = env_params.pop("env_name")

    train_env_kwargs = deepcopy(env_params)
    train_env_kwargs.pop("restriction")
    train_env_kwargs.pop("planning_constraint")

    excluded_keys = {"add_obj_to_start", "add_empty_to_start", "reset_probability_goals"}
    eval_env_kwargs = {k: v for k, v in train_env_kwargs.items() if k not in excluded_keys}

    # Load the environments (train and eval)
    train_env = gym.make(env_name, **train_env_kwargs)
    eval_env = gym.make(env_name, **eval_env_kwargs)

    # Directory for storing the policies
    directory = train_env.unwrapped.spec.id
    if dir_date_postfix is not None:
        dir_date_postfix = "-" + dir_date_postfix
        directory += dir_date_postfix
    base_save_dir = f"results/lof/{directory}"
    save_config(cfg, base_dir=base_save_dir, type='play')

    eval_envs = []
    Ts = []
    fsa_to_load = cfg.fsa_name if isinstance(cfg.fsa_name, ListConfig) else [cfg.fsa_name]
    for fsa_name in fsa_to_load:
        # Create the FSA env wrapper, to evaluate the FSA
        fsa, T = load_fsa('-'.join(["Office-v0", fsa_name]), eval_env,
                          fsa_symbols_from_env=fsa_symbols_from_env)  # Load FSA
        fsa_env = GridEnvWrapper(eval_env, fsa, fsa_init_state="u0", T=T)
        eval_envs.append(fsa_env)
        Ts.append(T)

    lof = hydra.utils.call(config=cfg.algorithm, env=train_env, eval_env=eval_envs, T=Ts)
    lof.load(base_dir=base_save_dir)

    lof.plot_q_vals(base_dir=os.path.join(base_save_dir, "options"))

    lof.train_metapolicies(iters=1000, reset_train=True)
    log_dict = lof.evaluate_fsa(reset=False)
    print(log_dict)
    
    lof.plot_meta_qvals(base_dir=base_save_dir)


if __name__ == "__main__":
    main()
