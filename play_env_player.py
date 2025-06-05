from abc import ABC

import hydra
import gym
from copy import deepcopy
from omegaconf import DictConfig
import envs
from envs.wrappers import GridEnvWrapper, FlatQEnvWrapper
from fsa.tasks_specification import load_fsa


class DummyEnvWrapper(ABC, gym.Env):
    def __init__(self, env):
        self.env = env

    def reset(self, use_low_level_init_state=True, use_fsa_init_state=True):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    fsa_symbols_from_env = cfg.get("fsa_symbols_from_env", False)
    wrapper = cfg.get("wrapper", None)

    # Create the train and eval environments
    env_params = dict(cfg.env)
    env_name = env_params.pop("env_name")

    train_env_kwargs = deepcopy(env_params)
    train_env_kwargs.pop("restriction")
    train_env_kwargs.pop("planning_constraint")

    env = gym.make(env_name, **train_env_kwargs)
    fsa_task = cfg.fsa_name
    fsa, T = load_fsa("-".join((env_name, fsa_task)), env, fsa_symbols_from_env=fsa_symbols_from_env)

    if wrapper is None:
        env = DummyEnvWrapper(env)
    elif wrapper.lower() == "grid":
        env = GridEnvWrapper(env, fsa, fsa_init_state="u0", T=T)
    elif wrapper.lower() == "flatq":
        env = FlatQEnvWrapper(env, fsa, fsa_init_state="u0")

    _ = env.reset(use_fsa_init_state=True, use_low_level_init_state=True)
    done = False

    try:
        while not done:
            env.env.render()

            raw = input(f"Enter action [0–{env.env.action_space.n - 1}]: ")
            try:
                action = int(raw)
            except ValueError:
                print("  ✗ Invalid input: please enter an integer.")
                continue

            if not (0 <= action < env.env.action_space.n):
                print(f"  ✗ Action out of range. Choose between 0 and {env.env.action_space.n - 1}.")
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
