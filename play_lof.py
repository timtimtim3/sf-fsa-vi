from copy import deepcopy
import gym
import hydra
import os
import wandb
from omegaconf import DictConfig, ListConfig
from envs.wrappers import GridEnvWrapper
from fsa.tasks_specification import load_fsa
from utils.utils import get_base_save_dir, seed_everything, save_config
import matplotlib as mpl


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    fsa_symbols_from_env = cfg.get("fsa_symbols_from_env", False)
    dir_postfix = cfg.get("dir_postfix", None)
    use_batch_dir = cfg.get("use_batch_dir", False)
    batch_dir_postfix = cfg.get("batch_dir_postfix", None)
    batch_run_name = cfg.get("batch_run_name", None)
    font_scale = cfg.get("font_scale", 1.0)  # e.g., 1.2 = 20% larger fonts

    # Global font size scaling (increase as needed)
    base_font_size = 10 
    mpl.rcParams.update({
        "axes.titlesize": base_font_size * font_scale,
        "axes.labelsize": base_font_size * font_scale,
        "xtick.labelsize": base_font_size * font_scale,
        "ytick.labelsize": base_font_size * font_scale,
        "legend.fontsize": base_font_size * font_scale,
        "legend.title_fontsize": base_font_size * font_scale,
        "figure.titlesize": base_font_size * font_scale,
        "mathtext.fontset": "stix",
    })

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
    base_save_dir = get_base_save_dir(train_env, dir_postfix, use_batch_dir, batch_run_name, batch_dir_postfix, 
                                      method="lof")
    save_config(cfg, base_dir=base_save_dir, type='play')

    eval_envs = []
    Ts = []
    fsa_to_load = cfg.fsa_name if isinstance(cfg.fsa_name, ListConfig) else [cfg.fsa_name]
    for fsa_name in fsa_to_load:
        # Create the FSA env wrapper, to evaluate the FSA
        fsa, T = load_fsa('-'.join([env_name, fsa_name]), eval_env,
                    fsa_symbols_from_env=fsa_symbols_from_env, using_lof=True)  # Load FSA
        # fsa, T = load_fsa('-'.join(["Office-v0", fsa_name]), eval_env,
        #                   fsa_symbols_from_env=fsa_symbols_from_env)  # Load FSA
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
