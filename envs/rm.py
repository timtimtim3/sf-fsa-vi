import numpy as np
import gym
from typing import Dict, List, Tuple, Callable
from gym.envs.classic_control import rendering
StateIDs = Tuple[int, ...]
RewardFunction = Callable[[np.ndarray, np.ndarray, np.ndarray], int]  # SxAxS -> R
RMTransitions = Dict[Tuple[int, Tuple[int, ...]], int]  # U x 2^P -> U
RMRewards = Dict[int, RewardFunction]  # U -> (SxAxS -> R)
SRMTransitions = Dict[Tuple[int, Tuple[int, ...]], Tuple[int, int]]  # U x 2^P -> (U, R)


# This class is not used ATM - included in case we want to use intermediate rewards at some point
class RewardMachine(object):
    def __init__(
            self,
            state_dim: int,
            prop_sym_dim: int,
            initial_state: int,
            terminal_states: StateIDs,          # Tuple of state ids
            transitions: RMTransitions,         # Dict with keys: Sx2^P and value: state,rew
            rewards: RMRewards,                 # Dict with keys: Sx2^P and value: state,rew
    ):
        self.state_dim = state_dim
        self.prop_sym_dim = prop_sym_dim
        self.initial_state = initial_state
        self.terminal_states = terminal_states
        self.transitions = transitions  # only add non-zero probs and use dict {Sx2^P -> S}
        self.rewards = rewards

        self.current_state = self.initial_state

    def step(
            self,
            truth_assignment: Tuple[bool, ...]
    ) -> Tuple[int, RewardFunction]:
        key = (self.current_state, truth_assignment)
        if key not in self.transitions:
            next_state = self.current_state
        else:
            next_state = self.transitions[key]
        self.current_state = next_state
        return self.current_state, self.rewards[self.current_state]

    def reset(self):
        self.current_state = self.initial_state


class SimpleRewardMachine(object):
    def __init__(
            self,
            state_dim: int,
            prop_sym_dim: int,
            initial_state: int,
            terminal_states: StateIDs,  # Tuple of state ids
            transitions: SRMTransitions
    ):
        self.state_dim = state_dim
        self.prop_sym_dim = prop_sym_dim
        self.initial_state = initial_state
        self.terminal_states = terminal_states
        self.transitions = transitions  # only add non-zero probs and use dict {Sx2^P -> S}

        self.current_state = self.initial_state

    def step(self, prop_sym: Tuple[bool, ...]) -> Tuple[int, int]:
        key = (self.current_state, prop_sym)
        if key not in self.transitions:
            return self.current_state, 0  # as a default self loop with 0
        else:
            val = self.transitions[key]
            self.current_state = val[0]
            return val

    def reset(self):
        self.current_state = self.initial_state

    def get_render_shapes(self, screen_height, screen_width):
        r = screen_height / 20
        # Circles - x,y,r,bool = active
        circles = []
        lines = []
        texts = []
        # Vertex coords - done
        # Arrow
        # Vertex text - done
        # Arrow text
        self.state_dim = 4

        line_length = 100
        coords = (0 + r + screen_width / 2, 0 + r )
        circles.append((int(coords[0]), int(coords[1]), int(r), self.current_state == 0))
        texts.append((coords[0] - r/3, coords[1] - r/4, "u0"))
        angle = 0
        for i in range(self.state_dim-1):
            x = coords[0] + np.cos(angle) * line_length
            y = coords[1] + np.sin(angle) * line_length
            coords = (x, y)
            # Vertices
            circles.append((int(coords[0]), int(coords[1]), int(r), self.current_state == i+1))
            texts.append((int(coords[0] - r / 3), int(coords[1] - r/4), f"u{i+1}"))


            angle += 2 * np.pi / self.state_dim

        for key, val in self.transitions.items():
            startv = key[0]
            endv = val[0]
            lines.append(((circles[startv][0], circles[startv][1]), (circles[endv][0], circles[endv][1])))
            # text = f"{key[1]},{val[1]}"
            # texts.append((int(circles[startv][0] * 0.8 + circles[endv][0] * 0.2), int(circles[startv][1] * 0.8 + circles[endv][1] * 0.2), text))


        return {"circles": circles, "texts": texts, "lines": lines}


class HierarchicalEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}
    def __init__(
            self,
            env: gym.Env,
            rm: SimpleRewardMachine
    ):

        self.env = env
        self.rm = rm

        self.screen_width = 600
        self.screen_height = 400
        self.viewer = None
        
        # Needed for SFOLS
        # self.observation_space = env.observation_space
        # self.action_space = env.action_space
        self.w = env.w

    def step(self, action) -> None:
        next_obs, _, _, info = self.env.step(action)
        phi = info["phi"]
        predicates = self.phi_to_predicates(phi)

        rm_s, rm_r = self.rm.step(predicates)
        self.render_mode = "human"

        # s', r, info, done = env.step(action) - step env (use P and R)
        # phi = env.get_suc_feat(s, a, s') - get successor features
        # lab = env.get_lab(s, a, s') - get labeling function for now I am using phi since it should be the same
        # rm_s, rm_r = rm.step(lab) - step reward machine?
        # return whatever we should return?

        done = rm_s in self.rm.terminal_states

        return next_obs, rm_r, done, info
 
    
    @staticmethod
    def phi_to_predicates(phi: np.ndarray):
        return tuple(phi.astype(np.int32))

    def reset(self):
        env_s = self.env.reset()
        rm_s = self.rm.reset()

        # TODO: Following line modified to make it consistent with SFOLs
        # return env_s, rm_s
        return env_s

    def render(self, mode="human"):

        if mode == "human":
            screen_width = 600
            screen_height = 400
            r = screen_height / 20
            if self.viewer is None:


                self.viewer = rendering.Viewer(screen_width, screen_height)

            shapes = self.rm.get_render_shapes(self.screen_width, self.screen_height)

            # Edges TODO: add arrows?
            for line in shapes["lines"]:
                self.viewer.draw_line(*line, color=(0,0,0))
            # Vertices
            for circle in shapes["circles"]:

                fp = rendering.FilledPolygon([(circle[0]-r, circle[1]-r), (circle[0]-r, circle[1]+r), (circle[0]+r, circle[1]+r), (circle[0]+r, circle[1]-r)])
                fp.set_color(255, (1 - circle[3]) * 255, (1 - circle[3]) * 255)
                self.viewer.add_onetime(fp)
                self.viewer.add_onetime(rendering.PolyLine(
                    [(circle[0] - r, circle[1] - r), (circle[0] - r, circle[1] + r), (circle[0] + r, circle[1] + r),
                    (circle[0] + r, circle[1] - r)], True))

            # for text in shapes["texts"]:  TODO couldn't get text to work
            #     pass


            return self.viewer.render(return_rgb_array=mode == "rgb_array")
        
        elif mode=="text":
            print(f"RM state: {self.rm.current_state} - env state: {self.env.state}")
    
    @property
    def observation_space(self):
        return self.env.observation_space
    @property 
    def action_space(self):
        return self.env.action_space