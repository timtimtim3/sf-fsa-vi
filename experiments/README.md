# Experiments Overview

This folder contains configuration files used for running experiments with the following methods:

- `flatq/` — Flat DQN baseline
- `lof/` — Logical Options Framework (LOF) baseline
- `sfols/` — Our proposed method: Successor Feature Options with Learned Subgoals (SF-OLS)

Each subfolder includes the WandB config used to run one representative seed of each experiment. In reality, each configuration was executed multiple times (typically 5 seeds) with different random seeds to ensure robustness of the results. However, for brevity and reproducibility, we include only one representative config per setting.

## Layouts Used

Experiments were conducted on the Office environment under the following layout variants:

- **Areas** — Spatially extended goal regions.
- **Original** — Standard layout with discrete goal tiles.
- **Original (6 FSA)** — Same layout, but tasks defined using a 6-state Finite State Automaton (FSA).
- **Teleport** — Variant with teleportation dynamics.

## Usage

These configuration files are intended for reference or to rerun specific experiment settings. 
