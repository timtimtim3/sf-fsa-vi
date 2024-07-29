import hydra 
from omegaconf import DictConfig, OmegaConf
import wandb 
from utils.utils import seed_everything 
import gym 
import envs
from fsa.tasks_specification import load_fsa
import os
from lof.algorithms.options import MetaPolicyVI
import numpy as np

@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    # Init Wandb
    run = wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        entity=cfg.wandb.entity, project=cfg.wandb.project,
        tags=["lof"], group="lof",
        mode="disabled"
    )
    
    # Set seeds
    seed_everything(cfg.seed)

    env_cfg = dict(cfg.env)

    # Load the environments (train and eval)
    env_name = env_cfg.pop("env_name")
    train_env = gym.make(env_name)
    eval_env = gym.make(env_name)

    # Create the FSA env wrapper
    fsa_task = cfg.fsa_name
    fsa, T = load_fsa("-".join((env_name, fsa_task)), eval_env)
    eval_env = hydra.utils.call(config=env_cfg.pop("eval_env"), env=eval_env, fsa=fsa, fsa_init_state="u0", T=T)


    # Get ground truth
    vi = MetaPolicyVI(train_env, eval_env, fsa, T, 1, 50)
    vi.learn_options()
    vi.train_metapolicy(iters=50)

    # Load the algorithm and run it
    lof = hydra.utils.call(config=cfg.algorithm, env=train_env, eval_env=eval_env, fsa=fsa, T=T)
    
    lof.gt_options = vi.options 
    lof.gt_V = vi.V

    lof.learn_options()
    lof.train_metapolicy(record=True)

    # Create and save options and metapolicy
    os.makedirs(f"results/lof/{run.name}/options")
    lof.save(f"results/lof/{run.name}")


    with open("value_iteration_options.txt", "w+") as fp:
        lines = []
        for idx, option in enumerate(vi.options):
            V = option.Q.max(axis=1)
            lines.append(f"value_function_option{idx}\n")
            for coords, state in train_env.coords_to_state.items():

                lines.append(f"{coords}={np.round(V[state], 2)}\n")

        fp.writelines(lines)

    with open("value_iteration.txt", "w+") as fp:
        lines = []
        for coords, state in train_env.coords_to_state.items():
            lines.append(f"{coords} [{train_env.MAP[coords]}] = {np.round(vi.V[0, state], 2)}\n")
       
        fp.writelines(lines)


    with open("qlearning_options.txt", "w+") as fp:
        lines = []
        for idx, option in enumerate(lof.options):
            V = option.Q.max(axis=1)
            lines.append(f"value_function_option{idx}\n")
            for coords, state in train_env.coords_to_state.items():

                lines.append(f"{coords}={np.round(V[state], 2)}\n")

        fp.writelines(lines)

    with open("qlearning.txt", "w+") as fp:
        lines = []
        for coords, state in train_env.coords_to_state.items():
            lines.append(f"{coords} [{train_env.MAP[coords]}] = {np.round(lof.V[0, state], 2)}\n")
       
        fp.writelines(lines)



    wandb.finish()

if __name__ == "__main__":

    main()