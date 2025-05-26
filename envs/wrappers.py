from typing import Union
import gym
import numpy as np


class GridEnvWrapper(gym.Env):


    def __init__(self, env, fsa, fsa_init_state, T):

        self.env = env 
        self.fsa = fsa
        self.exit_states = self.env.unwrapped.exit_states
        self.PHI_OBJ_TYPES = env.PHI_OBJ_TYPES
        self.fsa_init_state = fsa_init_state
        self.low_level_init_state = self.env.get_init_state()
        self.T = T
        self.feat_dim = env.feat_dim
        
    def get_state(self):
        return self.fsa_state, tuple(self.env.state)

    def reset(self, use_low_level_init_state=False, use_fsa_init_state=True):
        self.fsa_state = self.fsa_init_state
        self.state = tuple(self.env.reset(state=self.low_level_init_state if use_low_level_init_state else None))
        
        return (self.fsa_state, self.state), {"proposition": self.env.get_symbol_at_state(self.state)}

    def step(self, action):
        """
            Low-level and high-level transition
        """
        _, _ ,  _ , phi = self.env.step(action)
        state = self.env.state
        state_index = self.env.get_state_id(state)

        fsa_state_index = self.fsa.states.index(self.fsa_state)

        next_fsa_state_idxs = np.where(self.T[fsa_state_index, :, state_index] == 1)[0]

        if len(next_fsa_state_idxs) == 0:
            return (self.fsa_state, state), -1000, False, {}
        else: 
            next_fsa_state_index = next_fsa_state_idxs.item()
        
        self.fsa_state = self.fsa.states[next_fsa_state_index]

        obstacle = phi["proposition"] == "O" and "O" not in self.env.PHI_OBJ_TYPES
        done = self.fsa.is_terminal(self.fsa_state) or obstacle

        reward = -1000 if obstacle else -1

        return (self.fsa_state, state), reward, done, {"proposition": phi["proposition"]}


class FlatQEnvWrapper(gym.Env):

    def __init__(self, env, fsa, fsa_init_state="u0", eval_mode=False, reward_goal=True):

        self.env = env
        self.fsa = fsa
        self.fsa_init_state = fsa_init_state
        self.low_level_init_state = self.env.get_init_state()
        self.exit_states = self.env.unwrapped.exit_states
        self.observation_space = gym.spaces.Box(low=np.zeros(
            3), high=np.ones(3), dtype=np.float32)
        self.action_space = self.env.action_space
        self.w = np.array([1.0,])
        self.initial = []
        for s in self.env.initial:
            self.initial.append(self._merge_states(fsa_init_state, s))
        self.eval_mode = eval_mode

        self.step_reward = 0 if reward_goal else -1
        self.goal_reward = 1 if reward_goal else -1

    def get_state(self):
        return self._merge_states(fsa_state=self.fsa_state, state=self.state)

    def reset(self, use_low_level_init_state=False, use_fsa_init_state=False):
        if use_fsa_init_state:
            self.fsa_state = self.fsa_init_state
        else:
            uidx = np.random.randint(0, self.fsa.num_states - 1)
            self.fsa_state = self.fsa.states[uidx]
        self.state = tuple(self.env.reset(state=self.low_level_init_state if use_low_level_init_state else None))
        return self._merge_states(fsa_state=self.fsa_state, state=self.state)

    def step(self, action):
        """
            Low-level and high-level transition
        """
        _, _, done, info = self.env.step(action)
        prop = info["proposition"]
        state = self.env.state
        f_state = self.fsa_state
        self.state = state
        reward = self.step_reward

        neighbors = self.fsa.get_neighbors(self.fsa_state)
        satisfied = [prop in self.fsa.get_predicate((f_state, n)) for n in neighbors]

        next_fsa_state = None

        if any(satisfied):
            next_fsa_state = neighbors[satisfied.index(True)]

        if next_fsa_state is None:
            next_fsa_state = f_state

        self.fsa_state = next_fsa_state
        if self.env.terminate_action and action == self.env.TERMINATE:
            info["phi"] = -1000
            return self._merge_states(fsa_state=self.fsa_state, state=state), -1000, False, {'phi': -1000}

        # if self.env.MAP[state] == "O":
        #     info["phi"] = -1000
        #     return self._merge_states(fsa_state=self.fsa_state, state=state), -1000, True, {'phi': -1000}

        if self.eval_mode:
            done = self.fsa.is_terminal(self.fsa_state) or 'TimeLimit.truncated' in info
        else:
            done = self.fsa.is_terminal(self.fsa_state)
        info.pop('TimeLimit.truncated', None)

        if done:
            reward = self.goal_reward

        # TODO: Add failure case (crash into obstacle)
        info["phi"] = -1
        return self._merge_states(fsa_state=self.fsa_state, state=state), reward, done, info

    def _merge_states(self, fsa_state: Union[int, str], state):
        if isinstance(fsa_state, int):
            u_index = fsa_state
        else:
            u_index = self.fsa.states.index(fsa_state)
        return (u_index, *state)
