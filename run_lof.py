import hydra 
from omegaconf import DictConfig, OmegaConf
import wandb 
from utils.utils import seed_everything 
import gym 
import envs
from fsa.tasks_specification import load_fsa
import os

@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    # Init Wandb
    run = wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        entity=cfg.wandb.entity, project=cfg.wandb.project,
        tags=["lof"], group="lof",
        # mode="disabled"
    )
    
    # Set seeds
    seed_everything(cfg.seed)

    env_cfg = dict(cfg.env)

    # Load the environments (train and eval)
    env_name = env_cfg.pop("env_name")
    train_env = gym.make(env_name)
    eval_env = gym.make(env_name)

    fsa_task = cfg.fsa_name

    fsa, T = load_fsa("-".join((env_name, fsa_task)), eval_env)

    eval_env = hydra.utils.call(config=env_cfg.pop("eval_env"), env=eval_env, fsa=fsa, fsa_init_state="u0", T=T)

    # Load the algorithm and run it
    lof = hydra.utils.call(config=cfg.algorithm, env=train_env, eval_env=eval_env, fsa=fsa, T=T)
    lof.learn_options()

    lof.train_metapolicy(record=True)

    # Create and save options and metapolicy
    os.makedirs(f"results/lof/{run.name}/options")
    lof.save(f"results/lof/{run.name}")

    wandb.finish()

if __name__ == "__main__":

    main()