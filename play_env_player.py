import hydra
import gym
from copy import deepcopy
from omegaconf import DictConfig
import envs


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    # Create the train and eval environments
    env_params = dict(cfg.env)
    env_name = env_params.pop("env_name")

    train_env_kwargs = deepcopy(env_params)
    train_env_kwargs.pop("restriction")
    train_env_kwargs.pop("planning_constraint")

    env = gym.make(env_name, **train_env_kwargs)

    _ = env.reset()
    done = False

    try:
        while not done:
            env.render()

            raw = input(f"Enter action [0–{env.action_space.n - 1}]: ")
            try:
                action = int(raw)
            except ValueError:
                print("  ✗ Invalid input: please enter an integer.")
                continue

            if not (0 <= action < env.action_space.n):
                print(f"  ✗ Action out of range. Choose between 0 and {env.action_space.n - 1}.")
                continue

            obs, reward, done, info = env.step(action)
            print(f"  → obs={obs}, reward={reward}, done={done}, info={info}\n")

        print("✓ Episode finished.\n")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.\n")

    finally:
        env.close()


if __name__ == "__main__":
    main()
