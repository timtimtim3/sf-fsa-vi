import gym
import numpy as np

class GridEnvWrapper(gym.Env):


    def __init__(self, env, fsa, fsa_init_state, T):

        self.env = env 
        self.fsa = fsa
        self.fsa_init_state = fsa_init_state
        self.T = T

    def get_state(self):

        return (self.fsa_state, tuple(self.env.state))

    def reset(self):

        self.fsa_state = self.fsa_init_state
        self.state = tuple(self.env.reset())
        
        return (self.fsa_state, self.state)

    def step(self, action):
        """
            Low-level and high-level transition
        """
        _, _ ,  _ , phi = self.env.step(action) 
        state = self.env.state
        state_index = self.env.states.index(state)

        fsa_state_index = self.fsa.states.index(self.fsa_state)

        next_fsa_state_idxs = np.where(self.T[fsa_state_index, :, state_index] == 1)[0]

        if len(next_fsa_state_idxs) == 0:
            return (self.fsa_state, state), -1000, False, {}
        else: 
            next_fsa_state_index = next_fsa_state_idxs.item()
        
        self.fsa_state = self.fsa.states[next_fsa_state_index]

        obstacle = phi["proposition"] == "O"
        done = self.fsa.is_terminal(self.fsa_state) or obstacle

        reward = -1000 if obstacle else -1

        return (self.fsa_state, state), reward, done, {"proposition": phi["proposition"]}


class FlatQEnvWrapper(gym.Env):

    def __init__(self, env, fsa, fsa_init_state="u0", eval_mode=False):

        self.env = env
        self.fsa = fsa
        self.fsa_init_state = fsa_init_state
        self.exit_states = self.env.unwrapped.exit_states
        self.observation_space = gym.spaces.Box(low=np.zeros(
            3), high=np.ones(3), dtype=np.float32)
        self.action_space = self.env.action_space
        self.w = np.array([1.0,])
        self.initial = []
        for s in self.env.initial:
            self.initial.append(self._merge_states(fsa_init_state, s))
        self.eval_mode = eval_mode


    def get_state(self):

        return self._merge_states(fsa_state=self.fsa_state, state=self.state)

    def reset(self):

        self.fsa_state = self.fsa_init_state
        self.state = tuple(self.env.reset())
        return self._merge_states(fsa_state=self.fsa_state, state=self.state)

    def step(self, action):
        """
            Low-level and high-level transition
        """
        _, _, done, info = self.env.step(action)
        prop = info["proposition"]
        state = self.env.state
        f_state = self.fsa_state

        neighbors = self.fsa.get_neighbors(self.fsa_state)
        satisfied = [prop in self.fsa.get_predicate((f_state, n)) for n in neighbors]

        next_fsa_state = None

        if any(satisfied):
            next_fsa_state = neighbors[satisfied.index(True)]

        if next_fsa_state is None:
            next_fsa_state = f_state

        if self.env.MAP[state] == "O":
            info["phi"] = -1000
            return self._merge_states(fsa_state=self.fsa_state, state=state), -1000, True, {'phi': -1000}

        self.fsa_state = next_fsa_state
        if self.eval_mode:
            done = self.fsa.is_terminal(self.fsa_state) or 'TimeLimit.truncated' in info
        else:
            done = self.fsa.is_terminal(self.fsa_state)
        info.pop('TimeLimit.truncated', None)

        # TODO: Add failure case (crash into obstacle)
        info["phi"] = -1
        return self._merge_states(fsa_state=self.fsa_state, state=state), -1, done, info

    def _merge_states(self, fsa_state, state):
        u_index = self.fsa.states.index(fsa_state)
        return (u_index, *state)
