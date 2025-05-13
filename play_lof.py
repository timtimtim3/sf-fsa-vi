from omegaconf import DictConfig, OmegaConf
from fsa.tasks_specification import load_fsa
from envs.wrappers import GridEnvWrapper
from utils.utils import seed_everything
from lof.algorithms.options import MetaPolicyVI, MetaPolicyQLearning
import hydra, wandb, gym, os


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    base_dir = os.path.join("results", "lof", cfg.get("base_dir", "stilted-silence-401"))

    # disable WANDB logging
    wandb.init(mode="disabled")

    seed_everything(cfg.seed)

    # env setup
    env_name = cfg.env.env_name
    train_env = gym.make(env_name)
    eval_env = gym.make(env_name)
    fsa_task = cfg.fsa_name
    fsa, T = load_fsa(f"{env_name}-{fsa_task}", eval_env)
    eval_env = GridEnvWrapper(eval_env, fsa, fsa_init_state="u0", T=T)

    # load the meta‐policy that you originally saved
    # pick the right class (e.g. VI or QLearning) to match what you ran
    meta: MetaPolicyVI = MetaPolicyVI.load(
        env=train_env,
        eval_env=eval_env,
        fsa=fsa,
        T=T,
        base_dir=base_dir,
        gamma=cfg.algorithm.gamma,  # or hardcode if needed
    )

    # now you can evaluate, visualize, etc.
    success, reward = meta.evaluate_metapolicy(reset=False)
    print(f"Success={success}, Reward={reward}")
    # … any other analysis …

    meta.plot_meta_qvals(base_dir=base_dir)
    meta.plot_q_vals(base_dir=os.path.join(base_dir, "options"))


if __name__ == "__main__":
    main()
