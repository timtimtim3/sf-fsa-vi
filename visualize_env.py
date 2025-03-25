from omegaconf import DictConfig
import hydra
import gym
from envs.utils import get_rbf_activation_data
from sfols.plotting.plotting import plot_all_rbfs


@hydra.main(version_base=None, config_path="conf", config_name="visualize_env")
def main(cfg: DictConfig) -> None:
    env_params = dict(cfg.env)
    env_name = env_params.pop("env_name")

    train_env_kwargs = {
        k: v for k, v in {
            "add_obj_to_start": env_params.get("add_obj_to_start"),
            "add_empty_to_start": env_params.get("add_empty_to_start"),
            "level_name": env_params.get("level_name"),
            "only_rbf": env_params.get("only_rbf")
        }.items() if v is not None
    }

    train_env = gym.make(env_name, **train_env_kwargs)

    rbf_data, grid_size = get_rbf_activation_data(train_env, exclude={"X"})
    plot_all_rbfs(rbf_data, grid_size, train_env, skip_non_goal=False)


if __name__ == "__main__":
    main()
