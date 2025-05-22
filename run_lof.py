from copy import deepcopy
from omegaconf import DictConfig, OmegaConf
from envs.wrappers import GridEnvWrapper
from fsa.tasks_specification import load_fsa
from utils.utils import seed_everything, setup_run_dir
import hydra
import wandb
import envs
import gym


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
    base_save_dir = f"results/lof/{directory}"
    setup_run_dir(base_save_dir, cfg, run_name=run.name)

    # Create the FSA env wrapper
    fsa_task = cfg.fsa_name
    fsa, T = load_fsa("-".join((env_name, fsa_task)), eval_env)
    eval_env = GridEnvWrapper(eval_env, fsa, fsa_init_state="u0", T=T)
    # eval_env = hydra.utils.call(config=env_cfg.pop("eval_env"), env=eval_env, fsa=fsa, fsa_init_state="u0", T=T)

    # vi = MetaPolicyVI(train_env, eval_env, fsa, T)
    # vi.train_metapolicy()

    # Load the algorithm and run it
    lof = hydra.utils.call(config=cfg.algorithm, env=train_env, eval_env=eval_env, fsa=fsa, T=T)

    # lof.gt_options = vi.options

    lof.learn_options()

    # Once the base options have been learned, we can retrain the policy and keep track
    # of the results for the readaptation (planning), results
    lof.train_metapolicy(record=True)

    # Create and save options and metapolicy
    lof.save(base_save_dir)

    # for oidx, option in enumerate(lof.options):
    #     print(oidx)
    #     print(option)
    #     Vgt = lof.gt_options[oidx].Q.max(axis=1)
    #     V = option.Q.max(axis=1)
    #     errors = np.abs(Vgt - V).tolist()
    #
    #     data = [[str(state) + train_env.MAP[state], error] for (state, error) in zip(train_env.coords_to_state.keys(),
    #                                                                                  errors)]
    #     table = wandb.Table(data=data, columns=["state", "error"])
    #
    #     wandb.log({f"option_learning/option_{oidx}/final_errors_per_state":
    #               wandb.plot.bar(table, "state", "error", title="Custom Bar Chart")})

    wandb.finish()


if __name__ == "__main__":
    main()
