[![Python 3.9.18](https://img.shields.io/badge/Python-3.9.18-blue)](https://www.python.org/downloads/release/python-3918/)

# SF-FSA-VI

Code base for the paper "Learning Spatially Refined Sub-Policies for Temporal Task Composition in Continuous RL".

**Authors:** 

- **Tim van Gelder, supervised by Herke van Hoof** (AMLab University of Amsterdam)


## *Abstract*

> Traditional Reinforcement Learning (RL) methods can solve complex, long-horizon tasks, but typically fail to generalize to new, non-Markovian tasks without extensive re-training. Compositional methods like SF-OLS and LOF aim to address this issue by learning a set of sub-policies that can be composed at test time to solve unseen temporally extended tasks—specified as finite state automata (FSA)—in a zero-shot manner. While SF-OLS offers several advantages over other compositional methods, including faster value function composition and global optimality under stochastic dynamics, it has thus far only been applied in discrete domains. We extend SF-OLS to continuous state spaces by defining features—such as Radial Basis Functions (RBFs)—over the continuous domain, and introducing a new regression-based value iteration algorithm to compute optimal weights for unseen FSA tasks. Our method enables more globally efficient planning in environments where goals span spatial regions rather than single points or tiles, owing to its ability to learn sub-policies that selectively target different parts of a goal area. Additionally, we demonstrate that the approach achieves optimal behavior in stochastic environments, outperforming alternative compositional baselines like LOF.

