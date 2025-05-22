from copy import deepcopy
import numpy as np
import random
import gym
from gym.spaces import Discrete, Box, MultiDiscrete
from abc import ABC, abstractmethod
from envs.utils import gaussian_rbf, normalize_state
from envs.grid_levels import LEVELS
from typing import Protocol, Tuple, Dict, Any, TYPE_CHECKING, Union, List

if TYPE_CHECKING:
    class GridEnvProtocol(Protocol):
        object_ids: Dict[Any, Any]
        terminate_action: bool
        TERMINATE: int

        def _create_coord_mapping(self) -> None: ...

        def _create_transition_function(self) -> None: ...

        def get_observation_bounds(self) -> Tuple[np.ndarray, np.ndarray]: ...

        def get_symbol_at_state(self, state: Any) -> str: ...
else:
    # at runtime this base class is totally empty,
    # so it won’t shadow any real methods:
    class GridEnvProtocol:
        pass


class GridEnv(ABC, gym.Env):
    metadata = {'render.modes': ['human'],
                'video.frames_per_second': 20}
    LEFT, UP, RIGHT, DOWN, TERMINATE = 0, 1, 2, 3, 4

    MAP = None
    PHI_OBJ_TYPES = None

    """
    A simplified version of the office environment introduced in [1].
    This simplified version consists of 2 coffee machines and 2 office locations.

    References
    ----------
    [1] Icarte, RT, et al. "Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning".
    """

    def __init__(self, add_obj_to_start, random_act_prob, add_empty_to_start=False, init_w=True, terminate_action=False,
                 reset_probability_goals=None):
        """
        Creates a new instance of the coffee environment.

        """
        self.random_act_prob = random_act_prob
        self.add_obj_to_start = add_obj_to_start

        self.viewer = None
        self.height, self.width = self.MAP.shape
        self.all_objects = dict(zip(self.PHI_OBJ_TYPES, range(len(self.PHI_OBJ_TYPES))))
        self.initial = []
        self.occupied = set()
        self.object_ids = dict()
        self.terminate_action = terminate_action
        self.reset_probability_goals = reset_probability_goals
        self.initial_is_goal = []
        self.init_state = None

        for c in range(self.width):
            for r in range(self.height):
                if self.MAP[r, c] == 'X':
                    self.occupied.add((r, c))
                    continue
                elif self.MAP[r, c] == '_':
                    self.init_state = (r, c)
                    self.initial.append((r, c))
                    self.initial_is_goal.append(False)
                elif self.MAP[r, c] in self.PHI_OBJ_TYPES:
                    self.object_ids[(r, c)] = len(self.object_ids)
                    if add_obj_to_start and (self.MAP[r, c] != "O" or "O" in self.PHI_OBJ_TYPES):
                        self.initial.append((r, c))
                        self.initial_is_goal.append(True)
                elif self.MAP[r, c] == ' ' and add_empty_to_start:
                    self.initial.append((r, c))
                    self.initial_is_goal.append(False)

        if self.reset_probability_goals is not None:
            mask = np.array(self.initial_is_goal)
            num_goal = np.sum(mask)
            num_non_goal = len(mask) - num_goal

            if num_goal == 0 or num_non_goal == 0:
                raise ValueError("Must have both goal and non-goal states when using reset_probability_goals.")

            goal_mass = self.reset_probability_goals
            non_goal_mass = 1.0 - goal_mass

            probs = np.zeros(len(mask), dtype=np.float32)
            probs[mask] = goal_mass / num_goal
            probs[~mask] = non_goal_mass / num_non_goal

            self.init_probabilities = probs
        else:
            self.init_probabilities = None

        if init_w:
            self.w = np.zeros(self.feat_dim)
        self.action_space = Discrete(4 if not self.terminate_action else 5)
        height, width = self.MAP.shape
        self.observation_space = MultiDiscrete([height, width])
        self.seed()

    def _create_coord_mapping(self):
        """
        Create mapping from coordinates to state id and inverse mapping
        """
        self.state_to_coords = {}
        idx = 0
        for i in range(0, self.MAP.shape[0]):
            for j in range(0, self.MAP.shape[1]):
                if self.MAP[i][j] == "X":
                    continue
                self.state_to_coords[idx] = (i, j)
                idx += 1
        self.coords_to_state = dict(reversed(item) for item in self.state_to_coords.items())

    @abstractmethod
    def _create_transition_function(self):
        raise NotImplementedError

    def _create_transition_function_base(self):
        # Create transition matrix
        self.P = np.zeros((self.s_dim, self.a_dim, self.s_dim))

        for start_state in range(self.s_dim):
            for action in range(self.a_dim):
                start_coords = self.state_to_coords[start_state]
                next_coords = self.base_movement(start_coords, action)
                next_state = self.coords_to_state[next_coords]
                # Fill out transition matrix accounting for randomness
                for a in range(self.a_dim):
                    if action == a:
                        self.P[start_state, a, next_state] += 1 - self.random_act_prob
                    else:
                        self.P[start_state, a, next_state] += self.random_act_prob / (self.a_dim - 1)
        # sanity check
        assert np.allclose(np.sum(self.P, axis=2), 1)

    @staticmethod
    def state_to_array(state):
        return np.array(state, dtype=np.int32)

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(2147483647)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def reset(self, state=None):
        if state is not None:
            self.state = state
        elif self.init_probabilities is not None:
            index = np.random.choice(len(self.initial), p=self.init_probabilities)
            self.state = self.initial[index]
        else:
            self.state = random.choice(self.initial)
        return self.state_to_array(self.state)

    def get_init_state(self):
        return self.init_state

    def random_reset(self):
        # TODO: ?
        # states = [state for state in self.coords_to_state if state not in self.exit_states and self.MAP[state] != "O"]
        states = [state for state in self.coords_to_state if not state in self.exit_states]
        random_idx = np.random.randint(0, len(states))
        self.state = states[random_idx]

        return self.state_to_array(self.state)

    def do_action(self, state, action):
        row, col = state
        if action == self.LEFT:
            col -= 1
        elif action == self.UP:
            row -= 1
        elif action == self.RIGHT:
            col += 1
        elif action == self.DOWN:
            row += 1
        elif action == self.TERMINATE:
            pass
        else:
            raise Exception('bad action {}'.format(action))
        return row, col

    def base_movement(self, coords, action):
        new_coords = self.do_action(coords, action)
        if self.is_state_valid(new_coords):
            return new_coords
        return coords

    def is_state_valid(self, state):
        row, col = state
        return not (col < 0 or col >= self.width or row < 0 or row >= self.height or (row, col) in self.occupied)

    def step(self, action):
        # Movement
        old_state = self.state
        self.old_state = old_state
        old_state_index = self.coords_to_state[old_state]
        new_state_index = np.random.choice(a=self.s_dim, p=self.P[old_state_index, action])
        new_state = self.state_to_coords[new_state_index]

        self.state = new_state

        # Determine features and rewards
        done = self.is_done(old_state, action, new_state)
        phi = self.features(old_state, action, new_state)
        reward = -1  #np.dot(phi, self.w)
        prop = self.MAP[new_state]
        return self.state_to_array(self.state), reward, done, {'phi': phi, 'proposition': prop}

    # =========================================================================== #
    # STATE ENCODING FOR DEEP LEARNING                                            #
    # =========================================================================== #

    def encode(self, state):
        # (y, x), coll = state
        # n_state = self.width + self.height
        # result = np.zeros((n_state + len(coll),))
        # result[y] = 1
        # result[self.height + x] = 1
        # result[n_state:] = np.array(coll)
        # result = result.reshape((1, -1))
        # return result
        raise NotImplementedError()

    def encode_dim(self):
        return self.width + self.height + len(self.object_ids)

    # ===========================================================================
    # SUCCESSOR FEATURES
    # ===========================================================================
    def is_done(self, state, action, next_state):
        return next_state in self.object_ids

    def features(self, state, action, next_state):
        s1 = next_state
        nc = self.feat_dim
        phi = np.zeros(nc, dtype=np.float32)
        if s1 in self.object_ids:
            y, x = s1
            object_index = self.all_objects[self.MAP[y, x]]
            phi[object_index] = 1.
        return phi

    @property
    def feat_dim(self):
        return len(self.all_objects)

    @property
    def a_dim(self):
        return self.action_space.n

    @property
    def s_dim(self):
        return len(self.state_to_coords)

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            max_x, max_y = self.MAP.shape
            square_size = 30

            screen_height = square_size * max_x
            screen_width = square_size * max_y
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.square_map = {}

            for i in range(max_x):
                for j in range(max_y):
                    l = j * square_size
                    r = l + square_size
                    t = max_x * square_size - i * square_size
                    b = t - square_size
                    square = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                    self.viewer.add_geom(square)
                    self.viewer.square_map[(i, j)] = square

        # Use the subclass's color map if available, otherwise fallback
        color_map = getattr(self, 'RENDER_COLOR_MAP', {})  # Default to empty dict

        for square_coords in self.viewer.square_map:
            square = self.viewer.square_map[square_coords]

            # Agent color (yellow)
            if square_coords == tuple(self.state):
                color = [1, 1, 0]

            # Check if the square contains an object
            elif square_coords in self.object_ids.keys():
                obj_type = self.MAP[square_coords]
                color = color_map.get(obj_type, [0, 0, 1])  # Default to blue if not mapped

            # Walls
            elif square_coords in self.occupied:
                color = [0, 0, 0]

            # Empty space
            else:
                color = [1, 1, 1]

            square.set_color(*color)

        self.custom_render(square_map=self.viewer.square_map)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def custom_render(self, square_map: dict[tuple[int, int]]):
        pass

    def get_symbol_at_state(self, state):
        if state is None:
            return None

        if isinstance(state, tuple) and len(state) == 2:
            row, col = state
            if 0 <= row < len(self.MAP) and 0 <= col < len(self.MAP[0]):
                return self.MAP[row][col]
            else:
                raise IndexError(f"State {state} is out of bounds for MAP of size {len(self.MAP)}x{len(self.MAP[0])}")

        raise ValueError(f"Invalid state: {state}. Expected a tuple of two integers or None.")

    def get_observation_bounds(self):
        """
        Returns the lower and upper bounds of the observation space as NumPy arrays.

        Supports both gym.spaces.MultiDiscrete and gym.spaces.Box.

        Returns:
            tuple: (low, high), where both are np.ndarray with the same shape as observations.

        Raises:
            TypeError: If the observation space is not MultiDiscrete or Box.
        """
        space = self.observation_space

        if isinstance(space, gym.spaces.MultiDiscrete):
            low = np.zeros_like(space.nvec, dtype=np.float32)
            high = (space.nvec - 1).astype(np.float32)
        elif isinstance(space, gym.spaces.Box):
            low = space.low.astype(np.float32)
            high = space.high.astype(np.float32)
        else:
            raise TypeError(f"Unsupported observation space type: {type(space)}")

        return low, high

    def get_initial_state_distribution_sample(self):
        return self.initial

    def get_state_id(self, state):
        return self.coords_to_state[state]

    def get_planning_exit_states(self):
        """
        Returns a dictionary of exit states for planning (the grid cells of all exit state cells)
        """
        return self.exit_states

    def get_arrow_data(self, actions: np.ndarray, qvals: np.ndarray, states: List[Tuple[int, int]]):
        x_pos = []
        y_pos = []
        x_dir = []
        y_dir = []
        color = []
        coords_list = []

        for coords, a, q in zip(states, actions, qvals):
            x_d = y_d = 0
            if a == self.DOWN:
                y_d = 1
            elif a == self.UP:
                y_d = -1
            elif a == self.RIGHT:
                x_d = 1
            elif a == self.LEFT:
                x_d = -1

            x_pos.append(coords[1] + 0.5)
            y_pos.append(coords[0] + 0.5)
            x_dir.append(x_d)
            y_dir.append(y_d)
            color.append(q)
            coords_list.append(coords)
        # down, up , right, left
        return np.array(x_pos), np.array(y_pos), np.array(x_dir), np.array(y_dir), np.array(color), coords_list

    def get_planning_states(self):
        return self.coords_to_state.keys()

    def reward(self, state):
        return -1


class GridEnvContinuous(ABC, gym.Env):
    metadata = {'render.modes': ['human'], 'video.frames_per_second': 20}

    def __init__(self, add_obj_to_start, random_act_prob, add_empty_to_start=False, init_w=True, terminate_action=False,
                 reset_probability_goals=None, cell_size=1.0, step_size=0.8, noise_std=0.05, action_level=1):
        super(GridEnvContinuous, self).__init__()
        self.random_act_prob = random_act_prob
        self.add_obj_to_start = add_obj_to_start
        self.viewer = None
        self.height, self.width = self.MAP.shape
        self.all_objects = dict(zip(self.PHI_OBJ_TYPES, range(len(self.PHI_OBJ_TYPES))))
        self.initial = []
        self.occupied = set()
        self.object_ids = dict()
        self.valid_states = set()
        self.terminate_action = terminate_action
        self.reset_probability_goals = reset_probability_goals
        self.initial_is_goal = []
        self.init_state = None

        self.cell_size = cell_size
        self.step_size = step_size
        self.noise_std = noise_std

        self.action_level = action_level
        self.n_actions = 2 ** (self.action_level + 1)
        self.action_angles = np.linspace(0, 2 * np.pi, self.n_actions, endpoint=False)
        self.action_space = Discrete(self.n_actions if not self.terminate_action else self.n_actions + 1)
        self.TERMINATE = self.n_actions if self.terminate_action else None

        # Define the continuous observation space covering the whole map.
        self.epsilon = 1e-6  # define small epsilon
        self.movement_clip_range = (-cell_size + self.epsilon, cell_size - self.epsilon)
        self.low = np.array([0.0, 0.0], dtype=np.float32)
        self.high = np.array([self.width * self.cell_size - self.epsilon, self.height * self.cell_size - self.epsilon],
                             dtype=np.float32)
        self.observation_space = Box(low=self.low, high=self.high, dtype=np.float32)

        for c in range(self.width):
            for r in range(self.height):
                if self.MAP[r, c] == 'X':
                    self.occupied.add((r, c))
                    continue
                elif self.MAP[r, c] == '_':
                    self.init_state = (r, c)
                    self.initial.append((r, c))
                    self.initial_is_goal.append(False)
                elif self.MAP[r, c] in self.PHI_OBJ_TYPES:
                    self.object_ids[(r, c)] = len(self.object_ids)
                    if add_obj_to_start and self.MAP[r, c] != "O":
                        self.initial.append((r, c))
                        self.initial_is_goal.append(True)
                elif self.MAP[r, c] == ' ' and add_empty_to_start:
                    self.initial.append((r, c))
                    self.initial_is_goal.append(False)
                self.valid_states.add((r, c))

        if self.reset_probability_goals is not None:
            mask = np.array(self.initial_is_goal)
            num_goal = np.sum(mask)
            num_non_goal = len(mask) - num_goal

            if num_goal == 0 or num_non_goal == 0:
                raise ValueError("Must have both goal and non-goal states when using reset_probability_goals.")

            goal_mass = self.reset_probability_goals
            non_goal_mass = 1.0 - goal_mass

            probs = np.zeros(len(mask), dtype=np.float32)
            probs[mask] = goal_mass / num_goal
            probs[~mask] = non_goal_mass / num_non_goal

            self.init_probabilities = probs
        else:
            self.init_probabilities = None

        if init_w:
            self.w = np.zeros(self.feat_dim)

        self.seed()

    def _create_coord_mapping(self):
        """
        Create mapping from coordinates to state id and inverse mapping
        """
        self.state_to_coords = {}
        idx = 0
        for i in range(0, self.MAP.shape[0]):
            for j in range(0, self.MAP.shape[1]):
                if self.MAP[i][j] == "X":
                    continue
                self.state_to_coords[idx] = (i, j)
                idx += 1
        self.coords_to_state = dict(reversed(item) for item in self.state_to_coords.items())

    def get_init_state(self):
        return self.cell_to_continuous_center(self.init_state)

    @staticmethod
    def state_to_array(state):
        return np.array(state, dtype=np.float32)

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(2147483647)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def continuous_to_cell(self, continuous_state):
        """
        Convert a continuous coordinate to the corresponding grid cell (row, col).
        Uses floor division; cells are assumed to be arranged in a standard grid.
        """
        y, x = continuous_state
        col = int(x // self.cell_size)
        row = int(y // self.cell_size)
        return (row, col)

    def cell_to_continuous_base(self, state_cell):
        # Convert cell indices to continuous base coordinates: the top-left corner of the cell
        cell_base = np.array(state_cell, dtype=np.float32) * self.cell_size
        return cell_base

    def cell_to_continuous_center(self, state_cell):
        # Convert cell indices to continuous center coordinates: the center of the cell
        cell_base = self.cell_to_continuous_base(state_cell)
        return self.continuous_base_to_continuous_center(cell_base)

    def continuous_base_to_continuous_center(self, base):
        # Convert continuous cell base (top-left corner) to continuous center coordinates: the center of the cell
        return base + 0.5 * self.cell_size

    def continuous_state_to_continuous_center(self, state):
        cell = self.continuous_to_cell(state)
        center = self.cell_to_continuous_center(cell)
        return center

    def get_initial_state_distribution_sample(self):
        initial_centers = []
        for state_cell in self.initial:
            continuous_center = self.cell_to_continuous_center(state_cell)
            initial_centers.append(continuous_center)
        return initial_centers

    def get_all_valid_states(self):
        return self.valid_states

    def get_all_valid_continuous_states_centers(self):
        centers = list()
        for cell_coords in self.valid_states:
            cont_center = self.cell_to_continuous_center(cell_coords)
            centers.append(cont_center)
        return centers

    def get_planning_states(self):
        """
        Returns a list of planning states (the continuous centers of all grid cells)
        """
        return self.get_all_valid_continuous_states_centers()

    def get_planning_exit_states(self):
        """
        Returns a dictionary of exit states for planning (the continuous centers of all exit state cells)
        """
        continuous_exit_states_centers = {}
        for symbol, exit_states in self.exit_states.items():
            continuous_exit_states_centers[symbol] = set()
            for exit_state in exit_states:
                cont_exit_state_centre = self.cell_to_continuous_center(exit_state)
                continuous_exit_states_centers[symbol].add(tuple(cont_exit_state_centre))
        return continuous_exit_states_centers

    def get_state_id(self, state):
        state_cell = self.continuous_to_cell(state)
        return self.coords_to_state[state_cell]

    def sample_cell_from_initial(self):
        if self.init_probabilities is not None:
            index = np.random.choice(len(self.initial), p=self.init_probabilities)
            state_cell = self.initial[index]
        else:
            state_cell = random.choice(self.initial)
        return state_cell

    def sample_add_base_offset(self, state_cell):
        # Convert cell indices to continuous base coordinates: the top-left corner of the cell
        cell_base = self.cell_to_continuous_base(state_cell)

        # Generate a random offset for each dimension in the interval [0, self.cell_size)
        offset = np.random.rand(2) * self.cell_size

        # The continuous state is the cell's base coordinate plus the random offset
        state = cell_base + offset
        return state

    def reset(self, state=None):
        if state is not None:
            if not isinstance(state, np.ndarray):
                state = self.state_to_array(state)
            self.state = state
            self.state_cell = self.continuous_to_cell(state)
        else:
            self.state_cell = self.sample_cell_from_initial()
            self.state = self.sample_add_base_offset(self.state_cell)
        return self.state

    def action_to_dx_dy(self, action: int, unit_vector: bool = False):
        # direction: zero for terminate, else unit‐vector by angle
        if self.TERMINATE is not None and action == self.TERMINATE:
            dx, dy = 0.0, 0.0
        else:
            angle = self.action_angles[action]
            dx, dy = np.cos(angle), np.sin(angle)

        if not unit_vector:
            dx, dy = self.step_size * dx, self.step_size * dy
        return dx, dy

    def step(self, action):
        old_state = self.state.copy()
        old_cell  = self.state_cell  # (old_row, old_col)

        if action != self.TERMINATE:
            dx, dy = self.action_to_dx_dy(action, unit_vector=False)

            # Add Gaussian noise to the movement
            noise = np.random.normal(loc=0.0, scale=self.noise_std, size=2)
            movement = np.array([dy, dx], dtype=np.float32) + noise

            # Warn & clip any component outside [-1,1], so we ensure the agent can't skip over a cell
            for i, comp in enumerate(movement):
                if comp < self.movement_clip_range[0] or comp > self.movement_clip_range[1]:
                    print(f"Warning: movement[{i}] = {comp:.3f} out of {self.movement_clip_range}, clipping")
            movement = np.clip(movement, self.movement_clip_range[0], self.movement_clip_range[1])

            # Compute the new continuous state and clip to the environment boundaries
            new_state = self.state + movement
            new_state = np.clip(new_state, self.observation_space.low, self.observation_space.high)

            # Determine the new grid cell (discrete coordinates) that corresponds to new_state.
            new_cell = self.continuous_to_cell(new_state)

            # Check if the new cell is occupied (i.e. an obstacle).
            if new_cell in self.occupied:
                dr = new_cell[0] - old_cell[0]
                dc = new_cell[1] - old_cell[1]

                # If it's a diagonal move (|dr|=1 and |dc|=1)…
                if abs(dr) == 1 and abs(dc) == 1:
                    old_y, old_x = old_state  # continuous coords
                    new_y, new_x = new_state

                    # compute the two border‐crossing times
                    # vertical border at x = old_col*cell + sign(dc)*cell
                    col = old_cell[1]
                    bx = (col + (1 if dc > 0 else 0)) * self.cell_size
                    tx = (bx - old_x) / (new_x - old_x)

                    # horizontal border at y = old_row*cell + sign(dr)*cell
                    row = old_cell[0]
                    by = (row + (1 if dr > 0 else 0)) * self.cell_size
                    ty = (by - old_y) / (new_y - old_y)

                    # whichever happens first is your first‐entered neighbor
                    if tx < ty:
                        first_cell = (old_cell[0], new_cell[1])  # horizontal neighbor
                    else:
                        first_cell = (new_cell[0], old_cell[1])  # vertical neighbor

                    if first_cell not in self.occupied:
                        # back up to that instead of all the way to old_cell
                        new_cell = first_cell

                # if not diagonal, or diagonal but the first neighbor was blocked,
                # we fall back to old_cell
                if new_cell in self.occupied:
                    new_cell = old_cell

                # now clip the continuous state into whatever new_cell we have:
                r, c = new_cell
                cell_min = np.array([r, c], dtype=np.float32) * self.cell_size
                cell_max = cell_min + self.cell_size - self.epsilon
                new_state = np.clip(new_state, cell_min, cell_max)

            # finally update
            self.state = new_state
            self.state_cell = new_cell

        prop = self.MAP[self.state_cell]
        if prop == "X":
            print("Warning: agent is in an obstacle state X which shouldn't ever happen!")
        done = self.is_done(old_state, action, self.state)
        phi = self.features(old_state, action, self.state)
        reward = -1  # np.dot(phi, self.w)
        return self.state_to_array(self.state), reward, done, {'phi': phi, 'proposition': prop}

    def is_done(self, state, action, next_state):
        raise NotImplementedError

    def features(self, state, action, next_state):
        raise NotImplementedError

    @property
    def feat_dim(self):
        raise NotImplementedError

    @property
    def a_dim(self):
        return self.action_space.n

    @property
    def s_dim(self):
        return len(self.state_to_coords)

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            max_x, max_y = self.MAP.shape
            square_size = 30

            screen_height = square_size * max_x
            screen_width = square_size * max_y
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.square_map = {}

            for i in range(max_x):
                for j in range(max_y):
                    l = j * square_size
                    r = l + square_size
                    t = max_x * square_size - i * square_size
                    b = t - square_size
                    square = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                    self.viewer.add_geom(square)
                    self.viewer.square_map[(i, j)] = square

        # Use the subclass's color map if available, otherwise fallback
        color_map = getattr(self, 'RENDER_COLOR_MAP', {})  # Default to empty dict

        for square_coords in self.viewer.square_map:
            square = self.viewer.square_map[square_coords]

            # Agent color (yellow)
            if square_coords == self.state_cell:
                color = [1, 1, 0]

            # Check if the square contains an object
            elif square_coords in self.object_ids.keys():
                obj_type = self.MAP[square_coords]
                color = color_map.get(obj_type, [0, 0, 1])  # Default to blue if not mapped

            # Walls
            elif square_coords in self.occupied:
                color = [0, 0, 0]

            # Empty space
            else:
                color = [1, 1, 1]

            square.set_color(*color)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def get_symbol_at_state(self, state):
        if state is None:
            return None

        if isinstance(state, tuple) and len(state) == 2 and all(isinstance(x, int) for x in state):
            row, col = state
            if 0 <= row < len(self.MAP) and 0 <= col < len(self.MAP[0]):
                return self.MAP[row][col]
            else:
                raise IndexError(f"State {state} is out of bounds for MAP of size {len(self.MAP)}x{len(self.MAP[0])}")
        else:
            cell = self.continuous_to_cell(state)
            row, col = cell
            if 0 <= row < len(self.MAP) and 0 <= col < len(self.MAP[0]):
                return self.MAP[row][col]
            else:
                raise IndexError(
                    f"State {state} converts to cell {cell} out of bounds for MAP of size {len(self.MAP)}x{len(self.MAP[0])}")

    def get_observation_bounds(self):
        space = self.observation_space

        if isinstance(space, gym.spaces.MultiDiscrete):
            low = np.zeros_like(space.nvec, dtype=np.float32)
            high = (space.nvec - 1).astype(np.float32)
        elif isinstance(space, gym.spaces.Box):
            low = space.low.astype(np.float32)
            high = space.high.astype(np.float32)
        else:
            raise TypeError(f"Unsupported observation space type: {type(space)}")

        return low, high

    def get_arrow_data(self, actions: np.ndarray, qvals: np.ndarray,
                       states: Union[None, List[Tuple[float, float]]] = None):
        """
        Converts a list of continuous‐state centers + actions + qvals
        into X,Y,U,V,C arrays suitable for plt.quiver.

        Centers → discrete (row,col) via continuous_to_cell, then shift by +0.5
        so arrows appear in the middle of each cell.
        """

        if states is None:
            states = self.get_all_valid_continuous_states_centers()
        x_pos, y_pos, u, v = [], [], [], []

        for (y_cont, x_cont), a, q in zip(states, actions, qvals):
            # 1) discrete cell
            row, col = self.continuous_to_cell((y_cont, x_cont))
            # 2) plot coords in center of that cell
            x0 = col + 0.5
            y0 = row + 0.5

            # 3) arrow direction: zero for terminate, else unit‐vector by angle
            dx, dy = self.action_to_dx_dy(a, unit_vector=True)

            x_pos.append(x0)
            y_pos.append(y0)
            u.append(dx)
            v.append(dy)

        # color vector is just qvals
        c = np.array(qvals, dtype=np.float32)

        return (
            np.array(x_pos, dtype=np.float32),
            np.array(y_pos, dtype=np.float32),
            np.array(u, dtype=np.float32),
            np.array(v, dtype=np.float32),
            c,
            states
        )

    def get_grid_states_on_env(self, base: int):
        assert type(base) is int, f"Expected int, got {type(base).__name__}"
        n = base * base
        low, high = self.get_observation_bounds()
        width, height = high - low

        # divide the span into (base-1) steps so that we get base points including endpoints
        dx = width / (base - 1) if base > 1 else 0.0
        dy = height / (base - 1) if base > 1 else 0.0

        states = []
        for i in range(base):
            for j in range(base):
                state = (j * dy, i * dx)
                state = np.clip(state, self.observation_space.low, self.observation_space.high)
                state_cell = self.continuous_to_cell(state)
                if state_cell not in self.occupied:
                    states.append(state)
        return states


class Office(GridEnv):
    MAP = np.array([[' ', ' ', 'C1', ' ', ' ', 'X', ' ', 'C2', ' ', ' ', ' '],
                    [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' '],
                    ['M2', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' '],
                    [' ', ' ', ' ', ' ', ' ', 'X', 'O2', ' ', ' ', ' ', ' '],
                    [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' '],
                    [' ', 'X', 'X', ' ', ' ', 'X', ' ', ' ', 'X', 'X', ' '],
                    [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' '],
                    [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' '],
                    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                    [' ', ' ', ' ', '_', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                    ['O1', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'M1'], ])

    PHI_OBJ_TYPES = ['C1', 'C2', 'O1', 'O2', 'M1', 'M2']

    """
    A simplified version of the office environment introduced in [1].
    This simplified version consists of 2 coffee machines and 2 office locations.

    [1] Icarte, RT, et al. "Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning".
    """

    # Define a custom color map for Office
    RENDER_COLOR_MAP = {
        "C1": [0.6, 0.3, 0],  # Brown (Coffee Machine 1)
        "C2": [0.5, 0.25, 0],  # Dark Brown (Coffee Machine 2)
        "O1": [1, 0.6, 0],  # Orange (Office Location 1)
        "O2": [1, 0.4, 0],  # Dark Orange (Office Location 2)
        "M1": [0.5, 0, 0.5],  # Purple (Meeting Room 1)
        "M2": [0.3, 0, 0.3],  # Dark Purple (Meeting Room 2)
        "X": [0, 0, 0],  # Black (Walls)
        " ": [1, 1, 1],  # White (Empty Space)
        "_": [1, 1, 1],  # White (Starting Area)
    }
    QVAL_COLOR_MAP = {
        "C1": 0,
        "C2": 0,
        "O1": 1,
        "O2": 1,
        "M1": 2,
        "M2": 2,
        "X": 4,
        " ": 3,
        "_": 3,
    }

    def __init__(self, add_obj_to_start=False, random_act_prob=0.0, add_empty_to_start=False):
        super().__init__(add_obj_to_start=add_obj_to_start, random_act_prob=random_act_prob,
                         add_empty_to_start=add_empty_to_start)
        self._create_coord_mapping()
        self._create_transition_function()

        exit_states = {}
        for s in self.object_ids:
            symbol = self.MAP[s]
            exit_states[self.PHI_OBJ_TYPES.index(symbol)] = s

        self.exit_states = exit_states

    def _create_transition_function(self):
        self._create_transition_function_base()


class OfficeAreas(GridEnv):
    def __init__(self, add_obj_to_start=False, random_act_prob=0.0, add_empty_to_start=False,
                 level_name="office_areas"):
        # Load level data from the external LEVELS dictionary.
        level = LEVELS[level_name]

        # Set level-specific attributes.
        self.MAP = level.MAP
        self.PHI_OBJ_TYPES = level.PHI_OBJ_TYPES
        self.RENDER_COLOR_MAP = level.RENDER_COLOR_MAP
        self.QVAL_COLOR_MAP = level.QVAL_COLOR_MAP

        super().__init__(add_obj_to_start=add_obj_to_start, random_act_prob=random_act_prob,
                         add_empty_to_start=add_empty_to_start)
        self._create_coord_mapping()
        self._create_transition_function()

        feat_indices = {}
        exit_states = {}
        idx = 0
        for s in self.object_ids.keys():
            symbol = self.MAP[s]
            key = self.PHI_OBJ_TYPES.index(symbol)

            if key not in exit_states:
                exit_states[key] = {s}  # Initialize with a set containing s
            else:
                exit_states[key].add(s)  # Add new coordinate to the existing set
            feat_indices[s] = idx
            idx += 1
        self.exit_states = exit_states
        self.feat_indices = feat_indices

    def features(self, state, action, next_state):
        s1 = next_state
        nc = self.feat_dim
        phi = np.zeros(nc, dtype=np.float32)
        if s1 in self.object_ids.keys():
            feat_idx = self.feat_indices[s1]
            phi[feat_idx] = 1.
        return phi

    @property
    def feat_dim(self):
        return len(self.object_ids)

    def _create_transition_function(self):
        self._create_transition_function_base()


class OfficeAreasRBF(GridEnv):
    def __init__(self, add_obj_to_start=False, random_act_prob=0.0, add_empty_to_start=False, only_rbf=False,
                 level_name="office_areas_rbf_from_map", min_activation_thresh=0.1, terminate_action=False,
                 term_only_on_term_action=False, reset_probability_goals=None):
        # Load level data from the external LEVELS dictionary.
        level = LEVELS[level_name]

        # Set level-specific attributes.
        self.MAP = deepcopy(level.MAP)
        self.PHI_OBJ_TYPES = deepcopy(level.PHI_OBJ_TYPES)
        self.COORDS_RBFS = deepcopy(level.COORDS_RBFS)
        self.D_RBFS = deepcopy(level.D_RBFS)
        self.RENDER_COLOR_MAP = level.RENDER_COLOR_MAP
        self.QVAL_COLOR_MAP = level.QVAL_COLOR_MAP
        self.only_rbf = only_rbf
        self.term_only_on_term_action = term_only_on_term_action
        self.remove_redundant_features = level.REMOVE_REDUNDANT_FEAT if level.REMOVE_REDUNDANT_FEAT is not None else \
            False

        super().__init__(add_obj_to_start=add_obj_to_start, random_act_prob=random_act_prob,
                         add_empty_to_start=add_empty_to_start, init_w=False, terminate_action=terminate_action,
                         reset_probability_goals=reset_probability_goals)
        self._create_coord_mapping()
        self._create_transition_function()

        exit_states = {}
        for s in self.object_ids:
            symbol = self.MAP[s]
            key = self.PHI_OBJ_TYPES.index(symbol)

            if key not in exit_states:
                exit_states[key] = {s}  # Initialize with a set containing s
            else:
                exit_states[key].add(s)  # Add new coordinate to the existing set
        self.exit_states = exit_states

        if self.remove_redundant_features:
            self.remove_redundant_rbfs(min_activation_thresh=min_activation_thresh)

        self.rbf_lengths = {symbol: len(coords_list) for symbol, coords_list in self.COORDS_RBFS.items()}
        self.n_rbfs = sum(self.rbf_lengths.values())
        self.w = np.zeros(self.feat_dim)

        # Reserve initial indices for objects
        start_index = 0 if only_rbf else len(self.PHI_OBJ_TYPES)
        self.prop_at_feat_idx = []
        self.rbf_indices = {}
        self.FEAT_DATA = {}
        current_index = start_index
        # Reserve indices in our feature-weight vector for each rbf feature
        for symbol in sorted(self.COORDS_RBFS.keys()):  # Sort A -> B -> C
            self.rbf_indices[symbol] = {}
            for center_coords in self.COORDS_RBFS[symbol]:
                self.rbf_indices[symbol][center_coords] = current_index
                current_index += 1
                self.prop_at_feat_idx.append(symbol)

        for symbol in self.PHI_OBJ_TYPES:
            feat_data = []
            for center_coords, dist in zip(self.COORDS_RBFS[symbol], self.D_RBFS[symbol]):
                cy, cx = center_coords
                feat_data.append((cy, cx, dist))
            self.FEAT_DATA[symbol] = tuple(feat_data)

    def _create_transition_function(self):
        self._create_transition_function_base()

    def is_done(self, state, action, next_state):
        if self.terminate_action and self.term_only_on_term_action:
            if action == self.TERMINATE:
                return True
            return False
        elif self.terminate_action:
            # If we don't enter an Area, we do not terminate
            if next_state not in self.object_ids:
                return False

            # If we enter an Area, we terminate if we came from a different Area (or empty space)
            last_symbol = self.get_symbol_at_state(state)
            next_symbol = self.get_symbol_at_state(next_state)

            # If we entered the same Area as the one we came from, e.g. 'B' and 'B', we do not terminate
            if last_symbol == next_symbol:
                # Except if the agent selected the Terminate action
                if action == self.TERMINATE:
                    return True
                return False
            return True  # We terminate if we came from a different Area (or empty space)
        else:
            return next_state in self.object_ids

    @property
    def feat_dim(self):
        """Override the feat_dim property to account for rbfs as features."""
        return self.n_rbfs if self.only_rbf else len(self.all_objects) + self.n_rbfs

    def features(self, state, action, next_state):
        phi = np.zeros(self.feat_dim, dtype=np.float32)
        if self.terminate_action and not self.is_done(state, action, next_state):
            return phi

        if next_state in self.object_ids:
            y, x = next_state
            symbol = self.MAP[y, x]
            if not self.only_rbf:
                object_index = self.all_objects[symbol]
                phi[object_index] = 1.

            for i, center_coords in enumerate(self.COORDS_RBFS[symbol]):
                cy, cx = center_coords
                rbf_val = gaussian_rbf(x, y, cx, cy, d=self.D_RBFS[symbol][i])
                rbf_index = self.rbf_indices[symbol][center_coords]

                phi[rbf_index] = rbf_val
        return phi

    def remove_redundant_rbfs(self, min_activation_thresh=0.1):
        for symbol in self.PHI_OBJ_TYPES:
            print(f"\nChecking symbol: {symbol} for redundant RBFs")
            coords_list = self.COORDS_RBFS[symbol]
            distances_list = self.D_RBFS[symbol]
            symbol_idx = self.PHI_OBJ_TYPES.index(symbol)
            indices_to_remove = []

            # For each RBF...
            for i in range(len(coords_list)):
                cy, cx = coords_list[i]
                distance = distances_list[i]

                # Collect the activations over its exit states
                activations = []
                for exit_state in self.exit_states[symbol_idx]:
                    y, x = exit_state
                    rbf_val = gaussian_rbf(x, y, cx, cy, d=distance)
                    activations.append(rbf_val)

                max_activation = max(activations)
                if max_activation < min_activation_thresh:
                    indices_to_remove.append(i)
                    print(f"  Removing RBF at index {i} with coords {(cy, cx)} — max activation: {max_activation:.4f}")

            filtered_coords = [coords for i, coords in enumerate(coords_list) if i not in indices_to_remove]
            filtered_distances = [dist for i, dist in enumerate(distances_list) if i not in indices_to_remove]
            print(f"  Total removed for '{symbol}': {len(indices_to_remove)} out of {len(coords_list)}")
            self.COORDS_RBFS[symbol] = filtered_coords
            self.D_RBFS[symbol] = filtered_distances

    def get_feat_idx(self, symbol, feat):
        """
        Get the index in the feature vector returned by self.features of the feat under symbol.

        Parameters:
            symbol (str): Symbol of the goal area, e.g. 'A', 'B', or 'C'
            feat (float): Data of the feature we want to know the index of e.g. for RBF: (cx, cy, d) for fourier: (fx, fy)

        Returns:
            int: Index of feature under symbol in feature vector phi
        """
        cy, cx, d = feat
        return self.rbf_indices[symbol][(cy, cx)]


class FeatureExtractor:
    def __init__(self, feat_data, feat_fn, phi_obj_types, exit_states=None, remove_redundant_features=True,
                 min_activation_thresh=0.1, verbose=True, normalize_states_for_fourier=False, terminate_action=False,
                 low=None, high=None, normalize_features=False, clip_features=False):
        self._feat_data = feat_data
        self.feat_fn = feat_fn
        self.phi_obj_types = phi_obj_types
        self.phi_obj_types_sorted = sorted(phi_obj_types)
        self.normalize_states_for_fourier = normalize_states_for_fourier
        self.terminate_action = terminate_action
        self.low, self.high = low, high
        self.norm_weights = None
        self.normalize_features = normalize_features
        self.norm_weights = {}
        self.clip_features = clip_features

        if normalize_states_for_fourier and remove_redundant_features:
            if self.low is None or self.high is None:
                raise ValueError("Low and high cannot be None if remove_redundant_features and "
                                 "normalize_states_for_fourier are set to True")

        if remove_redundant_features:
            if exit_states is None:
                raise ValueError("Exit states cannot be None if remove_redundant_features is set to True")
            self._remove_redundant_features(exit_states, min_activation_thresh, verbose)

        if normalize_features:
            if exit_states is None:
                raise ValueError("Exit states cannot be None if norm_features is set to True")
            self._set_norm_weights(exit_states)

        self.feat_dim = sum([len(feat_data) for feat_data in self._feat_data.values()])
        self.feat_indices = {}

        self.prop_at_feat_idx = []
        self.feat_indices = {}
        current_index = 0
        for symbol in self.phi_obj_types_sorted:  # A -> B -> C
            feat_indices = []
            for _ in self._feat_data[symbol]:
                self.prop_at_feat_idx.append(symbol)
                feat_indices.append(current_index)
                current_index += 1

            self.feat_indices[symbol] = np.array(feat_indices)
        self.prop_at_feat_idx = tuple(self.prop_at_feat_idx)

    def _remove_redundant_features(self, exit_states, min_activation_thresh=0.1, verbose=False):
        for symbol in self.phi_obj_types:
            symbol_idx = self.phi_obj_types.index(symbol)
            feat_data = self._feat_data[symbol]
            exit_states_symbol = exit_states[symbol_idx]

            feat_matrix = np.zeros((len(exit_states_symbol), len(feat_data)))
            for i, exit_state in enumerate(exit_states_symbol):
                y, x = exit_state
                if self.normalize_states_for_fourier:
                    y, x = normalize_state(exit_state, self.low, self.high)

                feat_vec = self.feat_fn(x, y, feat_data=feat_data)
                feat_matrix[i, :] = feat_vec

            # Get max activation over exit states for each feature
            max_activation_feat = np.max(feat_matrix, axis=0)
            indices_feat_to_remove = np.where(max_activation_feat < min_activation_thresh)[0]

            if verbose:
                print(f"Removing {len(indices_feat_to_remove)} out of {len(feat_data)} features for symbol {symbol} "
                      f"at {indices_feat_to_remove}")

            filtered_feat_data = tuple([feat for i, feat in enumerate(feat_data) if i not in indices_feat_to_remove])
            self._feat_data[symbol] = filtered_feat_data

    def _set_norm_weights(self, exit_states):
        for symbol in self.phi_obj_types:
            symbol_idx = self.phi_obj_types.index(symbol)
            feat_data = self._feat_data[symbol]
            exit_states_symbol = exit_states[symbol_idx]

            feat_matrix = np.zeros((len(exit_states_symbol), len(feat_data)))
            for i, exit_state in enumerate(exit_states_symbol):
                y, x = exit_state
                if self.normalize_states_for_fourier:
                    y, x = normalize_state(exit_state, self.low, self.high)

                feat_vec = self.feat_fn(x, y, feat_data=feat_data)
                feat_matrix[i, :] = feat_vec

            # Get max activation over exit states for each feature
            max_activation_feat = np.max(feat_matrix, axis=0)

            # Avoid division by zero: replace 0 with 1
            max_activation_feat_safe = np.where(max_activation_feat == 0, 1, max_activation_feat)

            # Create a normalization vector such that each element multiplied gives max of 1
            norm_vector = 1.0 / max_activation_feat_safe
            self.norm_weights[symbol] = norm_vector

    def features(self, env, state, action, next_state):
        phi = np.zeros(self.feat_dim, dtype=np.float32)
        if self.terminate_action and not env.is_done(state, action, next_state):
            return phi

        # Only assign non-zero values if the next state is a goal state
        is_goal_state = env.is_goal_state(next_state)

        if is_goal_state:
            symbol = env.get_symbol_at_state(next_state)

            if self.normalize_states_for_fourier:
                y, x = normalize_state(next_state, env.low, env.high)
            else:
                y, x = next_state

            feat_vec = self.feat_fn(x, y, self._feat_data[symbol])

            if self.normalize_features:
                feat_vec = feat_vec * self.norm_weights[symbol]

            if self.clip_features:
                feat_vec = np.clip(feat_vec, 0.0, 1.0)

            feat_indices = self.feat_indices[symbol]
            phi[feat_indices] = feat_vec
        return phi

    def get_feat_idx(self, symbol, feat):
        """
        Get the index in the feature vector returned by self.features of the feat under symbol.

        Parameters:
            symbol (str): Symbol of the goal area, e.g. 'A', 'B', or 'C'
            feat (float): Data of the feature we want to know the index of e.g. for RBF: (cx, cy, d) for fourier: (fx, fy)

        Returns:
            int: Index of feature under symbol in feature vector phi
        """

        feat_idx_in_list = self._feat_data[symbol].index(feat)
        feat_indices_symbol = self.feat_indices[symbol]
        feat_idx_feat = feat_indices_symbol[feat_idx_in_list]
        return feat_idx_feat


class OfficeAreasFeatures(GridEnv):
    def __init__(self, add_obj_to_start=False, random_act_prob=0.0, add_empty_to_start=False,
                 level_name="office_areas_fourier", min_activation_thresh=0.1, terminate_action=False,
                 term_only_on_term_action=False, reset_probability_goals=None):
        """
        Initialize the OfficeAreasFeatures environment.

        Parameters:
            add_obj_to_start (bool): If True, adds goal states (e.g. 'A', 'B', 'C') to the set of
                possible starting states for the agent at the beginning of each episode.
            random_act_prob (float): Probability with which the environment executes a random action instead
                of the agent's selected action. This introduces stochasticity into the agent's behavior.
            add_empty_to_start (bool): If True, adds the non-goal (i.e. empty) tiles to the set of possible
                starting states.
            level_name (str): Name of the level to load from the `LEVELS` dictionary. The level must be compatible
                with this environment class (e.g., a Fourier or RBF level for OfficeAreasFeatures).
            min_activation_thresh (float): Minimum activation threshold used when removing redundant features.
                If `remove_redundant_features` in level is True, features with maximum activation below this threshold
                across the entire grid are removed.
            terminate_action (bool): If True, enables a special "terminate" action that allows the agent to
                explicitly end the episode.
            term_only_on_term_action (bool): If True and `terminate_action` is enabled, the environment will
                only terminate when the agent selects the "terminate" action. If False, entering a goal state
                from an empty state or a different goal-state may also end the episode automatically. Still, when moving
                from a goal-state to a state with the same goal (e.g. from 'B' to 'B') will not terminate.
            reset_probability_goals (float or None): Optional value to bias the probability distribution over
                initial states. If specified, this value indicates the proportion of initializations that should
                occur in goal states, with the remainder in non-goal (empty) states. Ignored if `None`.
        """
        # Load level data from the external LEVELS dictionary.
        level = LEVELS[level_name]

        # Set level-specific attributes.
        self.MAP = deepcopy(level.MAP)
        self.PHI_OBJ_TYPES = deepcopy(level.PHI_OBJ_TYPES)
        self.FEAT_DATA = deepcopy(level.FEAT_DATA)
        self.RENDER_COLOR_MAP = level.RENDER_COLOR_MAP
        self.QVAL_COLOR_MAP = level.QVAL_COLOR_MAP
        self.FEAT_FN = level.FEAT_FN
        self.remove_redundant_features = level.REMOVE_REDUNDANT_FEAT if level.REMOVE_REDUNDANT_FEAT is not None else \
            False

        super().__init__(add_obj_to_start=add_obj_to_start, random_act_prob=random_act_prob,
                         add_empty_to_start=add_empty_to_start, init_w=False, terminate_action=terminate_action,
                         reset_probability_goals=reset_probability_goals)
        self._create_coord_mapping()
        self._create_transition_function()

        self.term_only_on_term_action = term_only_on_term_action
        self.low, self.high = self.get_observation_bounds()

        exit_states = {}
        for s in self.object_ids:
            symbol = self.MAP[s]
            key = self.PHI_OBJ_TYPES.index(symbol)

            if key not in exit_states:
                exit_states[key] = {s}  # Initialize with a set containing s
            else:
                exit_states[key].add(s)  # Add new coordinate to the existing set
        self.exit_states = exit_states

        feature_extractor_kwargs = dict()
        if hasattr(level, "NORMALIZE_STATES_FOR_FOURIER"):
            feature_extractor_kwargs["normalize_states_for_fourier"] = level.NORMALIZE_STATES_FOR_FOURIER
        if hasattr(level, "NORMALIZE_FEATURES"):
            feature_extractor_kwargs["normalize_features"] = level.NORMALIZE_FEATURES
        self.feature_extractor = (
            FeatureExtractor(self.FEAT_DATA, self.FEAT_FN, self.PHI_OBJ_TYPES, exit_states=exit_states,
                             remove_redundant_features=self.remove_redundant_features, verbose=True,
                             min_activation_thresh=min_activation_thresh, terminate_action=terminate_action,
                             low=self.low, high=self.high, **feature_extractor_kwargs))
        self.prop_at_feat_idx = self.feature_extractor.prop_at_feat_idx

        # self.feat_dim depends on feature extractor __init__, so intialize w here
        self.w = np.zeros(self.feat_dim)

    def _create_transition_function(self):
        self._create_transition_function_base()

    def is_done(self, state, action, next_state):
        if self.terminate_action and self.term_only_on_term_action:
            if action == self.TERMINATE:
                return True
            return False
        elif self.terminate_action:
            # If we don't enter an Area, we do not terminate
            if next_state not in self.object_ids:
                return False

            # If we enter an Area, we terminate if we came from a different Area (or empty space)
            last_symbol = self.get_symbol_at_state(state)
            next_symbol = self.get_symbol_at_state(next_state)

            # If we entered the same Area as the one we came from, e.g. 'B' and 'B', we do not terminate
            if last_symbol == next_symbol:
                # Except if the agent selected the Terminate action
                if action == self.TERMINATE:
                    return True
                return False
            return True  # We terminate if we came from a different Area (or empty space)
        else:
            return next_state in self.object_ids

    @property
    def feat_dim(self):
        return self.feature_extractor.feat_dim

    def features(self, state, action, next_state):
        return self.feature_extractor.features(self, state, action, next_state)

    def is_goal_state(self, state):
        return state in self.object_ids

    def get_feat_idx(self, symbol, feat):
        """
        Get the index in the feature vector returned by self.features of the feat under symbol.

        Parameters:
            symbol (str): Symbol of the goal area, e.g. 'A', 'B', or 'C'
            feat (float): Data of the feature we want to know the index of e.g. for RBF: (cx, cy, d) for fourier: (fx, fy)

        Returns:
            int: Index of feature under symbol in feature vector phi
        """
        return self.feature_extractor.get_feat_idx(symbol, feat)

    def get_weight_idxs_for_symbol(self, symbol):
        return self.feature_extractor.feat_indices[symbol]


class OfficeAreasFeaturesMixin(GridEnvProtocol):
    def __init__(self, add_obj_to_start=False, random_act_prob=0.0, add_empty_to_start=False,
                 level_name="office_areas_fourier", min_activation_thresh=0.1, terminate_action=False,
                 term_only_on_term_action=False, reset_probability_goals=None, **env_kwargs):
        """
        Initialize the OfficeAreasFeaturesContinuous environment.

        Parameters:
            add_obj_to_start (bool): If True, adds goal states (e.g. 'A', 'B', 'C') to the set of
                possible starting states for the agent at the beginning of each episode.
            random_act_prob (float): Probability with which the environment executes a random action instead
                of the agent's selected action. This introduces stochasticity into the agent's behavior.
            add_empty_to_start (bool): If True, adds the non-goal (i.e. empty) tiles to the set of possible
                starting states.
            level_name (str): Name of the level to load from the `LEVELS` dictionary. The level must be compatible
                with this environment class (e.g., a Fourier or RBF level for OfficeAreasFeatures).
            min_activation_thresh (float): Minimum activation threshold used when removing redundant features.
                If `remove_redundant_features` in level is True, features with maximum activation below this threshold
                across the entire grid are removed.
            terminate_action (bool): If True, enables a special "terminate" action that allows the agent to
                explicitly end the episode.
            term_only_on_term_action (bool): If True and `terminate_action` is enabled, the environment will
                only terminate when the agent selects the "terminate" action. If False, entering a goal state
                from an empty state or a different goal-state may also end the episode automatically. Still, when moving
                from a goal-state to a state with the same goal (e.g. from 'B' to 'B') will not terminate.
            reset_probability_goals (float or None): Optional value to bias the probability distribution over
                initial states. If specified, this value indicates the proportion of initializations that should
                occur in goal states, with the remainder in non-goal (empty) states. Ignored if `None`.
        """
        # Load level data from the external LEVELS dictionary.
        level = LEVELS[level_name]
        self.level_name = level_name

        # Set level-specific attributes.
        self.MAP = deepcopy(level.MAP)
        self.PHI_OBJ_TYPES = deepcopy(level.PHI_OBJ_TYPES)
        self.FEAT_DATA = deepcopy(level.FEAT_DATA)
        self.RENDER_COLOR_MAP = level.RENDER_COLOR_MAP
        self.QVAL_COLOR_MAP = level.QVAL_COLOR_MAP
        self.FEAT_FN = level.FEAT_FN
        self.remove_redundant_features = level.REMOVE_REDUNDANT_FEAT if level.REMOVE_REDUNDANT_FEAT is not None else \
            False
        self.term_only_on_term_action = term_only_on_term_action
        self.normalize_features = False if not hasattr(level, "NORMALIZE_FEATURES") else level.NORMALIZE_FEATURES

        super().__init__(add_obj_to_start=add_obj_to_start, random_act_prob=random_act_prob,
                         add_empty_to_start=add_empty_to_start, init_w=False, terminate_action=terminate_action,
                         reset_probability_goals=reset_probability_goals, **env_kwargs)
        self._create_coord_mapping()
        self._configure_clip_features()

        # give subclasses a chance to mass‑modify FEAT_DATA
        self.FEAT_DATA = self._adjust_feat_data(self.FEAT_DATA)

        self.low, self.high = self.get_observation_bounds()

        exit_states = {}
        for s in self.object_ids:
            symbol = self.MAP[s]
            key = self.PHI_OBJ_TYPES.index(symbol)

            if key not in exit_states:
                exit_states[key] = {s}  # Initialize with a set containing s
            else:
                exit_states[key].add(s)  # Add new coordinate to the existing set
        self.exit_states = exit_states

        self._create_exit_states_centers()
        exit_states_for_feats = getattr(self, 'exit_states_centers', exit_states)

        feature_extractor_kwargs = dict()
        if hasattr(level, "NORMALIZE_STATES_FOR_FOURIER"):
            feature_extractor_kwargs["normalize_states_for_fourier"] = level.NORMALIZE_STATES_FOR_FOURIER
        self.feature_extractor = (
            FeatureExtractor(self.FEAT_DATA, self.FEAT_FN, self.PHI_OBJ_TYPES, exit_states=exit_states_for_feats,
                             remove_redundant_features=self.remove_redundant_features, verbose=True,
                             min_activation_thresh=min_activation_thresh, terminate_action=terminate_action,
                             low=self.low, high=self.high, clip_features=self.clip_features,
                             normalize_features=self.normalize_features, **feature_extractor_kwargs))
        self.prop_at_feat_idx = self.feature_extractor.prop_at_feat_idx

        # self.feat_dim depends on feature extractor __init__, so intialize w here
        self.w = np.zeros(self.feat_dim)

    def _create_exit_states_centers(self):
        # default: discrete has no centers
        return

    def _adjust_feat_data(self, feat_data):
        """
        Hook for subclasses (e.g. continuous) to shift / re‑center their feature definitions.
        By default, just returns the data unchanged.
        """
        return feat_data

    def _configure_clip_features(self):
        # default: discrete does no clipping
        self.clip_features = False

    def is_done(self, state, action, next_state):
        if self.terminate_action and self.term_only_on_term_action:
            if action == self.TERMINATE:
                return True
            return False
        elif self.terminate_action:
            # If we don't enter an Area, we do not terminate
            if not self.is_goal_state(next_state):
                return False

            # If we enter an Area, we terminate if we came from a different Area (or empty space)
            last_symbol = self.get_symbol_at_state(state)
            next_symbol = self.get_symbol_at_state(next_state)

            # If we entered the same Area as the one we came from, e.g. 'B' and 'B', we do not terminate
            if last_symbol == next_symbol:
                # Except if the agent selected the Terminate action
                if action == self.TERMINATE:
                    return True
                return False
            return True  # We terminate if we came from a different Area (or empty space)
        else:
            return self.is_goal_state(next_state)

    @property
    def feat_dim(self):
        return self.feature_extractor.feat_dim

    def features(self, state, action, next_state):
        return self.feature_extractor.features(self, state, action, next_state)

    def is_goal_state(self, state):
        return self.get_symbol_at_state(state) in self.PHI_OBJ_TYPES

    def get_feat_idx(self, symbol, feat):
        """
        Get the index in the feature vector returned by self.features of the feat under symbol.

        Parameters:
            symbol (str): Symbol of the goal area, e.g. 'A', 'B', or 'C'
            feat (float): Data of the feature we want to know the index of e.g. for RBF: (cx, cy, d) for fourier: (fx, fy)

        Returns:
            int: Index of feature under symbol in feature vector phi
        """
        return self.feature_extractor.get_feat_idx(symbol, feat)

    def get_weight_idxs_for_symbol(self, symbol):
        return self.feature_extractor.feat_indices[symbol]


class OfficeAreasFeaturesDiscrete(OfficeAreasFeaturesMixin, GridEnv):
    def __init__(self, *args, **kwargs):
        # let the mix‑in and GridEnv __init__ run
        super().__init__(*args, **kwargs)

        # now do the discrete‑only initialization
        self._create_transition_function()

    def _create_transition_function(self):
        self._create_transition_function_base()


class OfficeAreasFeaturesContinuous(OfficeAreasFeaturesMixin, GridEnvContinuous):
    def __init__(self,
                 *args,
                 cell_size=1.0,
                 step_size=0.8,
                 noise_std=0.05,
                 action_level=1,
                 **kwargs):
        # gather the continuous‑only args
        env_kwargs = dict(
            cell_size=cell_size,
            step_size=step_size,
            noise_std=noise_std,
            action_level=action_level,
        )
        # pass everything up into the mix‑in (which forwards to GridEnvContinuous)
        super().__init__(*args, **{**kwargs, **env_kwargs})

    def _create_exit_states_centers(self):
        # Build a parallel map of continuous coordinates at the centers
        self.exit_states_centers = dict()
        for symbol, state_cells_set in self.exit_states.items():
            self.exit_states_centers[symbol] = set()
            for cell in state_cells_set:
                y0, x0 = self.cell_to_continuous_center(cell)
                self.exit_states_centers[symbol].add((y0, x0))

    def _adjust_feat_data(self, feat_data):
        if "rbf" not in self.level_name:
            return feat_data

        centered = {}
        for symbol, feats in feat_data.items():
            centered_feats = []
            for cy, cx, d in feats:
                y0, x0 = self.cell_to_continuous_center((cy, cx))
                centered_feats.append((y0, x0, d))
            centered[symbol] = tuple(centered_feats)
        return centered

    def _configure_clip_features(self):
        # if we're normalizing features, continuous Fourier features get clipped, others don’t
        self.clip_features = (self.normalize_features and "fourier" in self.level_name)


class Delivery(GridEnv):
    MAP = np.array([['O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O'],
                    ['O', 'O', 'O', 'C', 'O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O'],
                    ['O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O'],
                    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                    ['O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O'],
                    ['O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O'],
                    ['O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O'],
                    [' ', 'A', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                    ['O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O'],
                    ['O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O'],
                    ['O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O'],
                    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
                    ['O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O'],
                    ['O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O'],
                    ['O', 'O', 'O', ' ', 'O', 'O', 'O', 'H', 'O', 'O', 'O', ' ', 'O', 'O', 'O'], ])

    PHI_OBJ_TYPES = ['A', 'B', 'C', 'H', 'O']

    """
    A simplified version of the office environment introduced in [1].
    This simplified version consists of 2 coffee machines and 2 office locations.

    [1] Icarte, RT, et al. "Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning".
    """

    def __init__(self, add_obj_to_start=False, random_act_prob=0.0):
        super().__init__(add_obj_to_start=add_obj_to_start, random_act_prob=random_act_prob)
        self._create_coord_mapping()
        self._create_transition_function()

        exit_states = {}
        for s in self.object_ids:
            symbol = self.MAP[s]
            if symbol in ("O"):
                continue
            exit_states[self.PHI_OBJ_TYPES.index(symbol)] = s

        if not add_obj_to_start:
            home_state = self.PHI_OBJ_TYPES.index('H')
            self.initial.append(exit_states[home_state])

        self.exit_states = exit_states

    def _create_transition_function(self):
        self._create_transition_function_base()

    def features(self, state, action, next_state):
        s1 = next_state
        nc = self.feat_dim
        phi = np.zeros(nc, dtype=np.float32)
        if s1 in self.object_ids:
            y, x = s1
            object_index = self.all_objects[self.MAP[y, x]]
            phi[object_index] = 1. if self.MAP[s1] != "O" else -1

        return phi

    def custom_render(self, square_map: dict[tuple[int, int]]):
        for square_coords in square_map:
            square = square_map[square_coords]
            # Teleport
            if self.MAP[square_coords] == 'O':
                color = [0, 0, 0]
            elif self.MAP[square_coords] == 'A':
                color = [1, 0, 0]
            elif self.MAP[square_coords] == 'B':
                color = [0, 1, 0]
            elif self.MAP[square_coords] == 'H':
                color = [0.7, 0.3, 0.7]
            else:
                continue
            square.set_color(*color)

    def step(self, action):
        # Movement
        old_state = self.state
        old_state_index = self.coords_to_state[old_state]
        new_state_index = np.random.choice(a=self.s_dim, p=self.P[old_state_index, action])
        new_state = self.state_to_coords[new_state_index]

        phi = self.features(old_state, action, new_state)

        self.state = new_state

        # Determine features and rewards
        reward = self.reward(old_state)
        done = self.MAP[new_state] in self.PHI_OBJ_TYPES and self.MAP[new_state] != 'O'
        prop = self.MAP[new_state]

        return self.state_to_array(self.state), reward, done, {'phi': phi, 'proposition': prop}

    def reward(self, state):

        reward = -1

        y, x = state

        if self.MAP[y][x] == 'O':
            reward = -1000

        return reward


class DoubleSlit(GridEnv):
    MAP = np.array([
        ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X',
         'X', 'X', 'X', 'X', 'X', 'X', 'O1', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
         ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
         ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
         ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
         ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
         ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
         ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
         ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
         ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
         ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['_', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
         ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
         ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
         ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
         ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
         ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
         ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
         ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
         ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
         ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
         ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X',
         'X', 'X', 'X', 'X', 'X', 'X', 'O2', 'X']

    ])

    PHI_OBJ_TYPES = ['O1', 'O2']
    UP, RIGHT, DOWN = 0, 1, 2

    def __init__(self, random_act_prob=0.0, add_obj_to_start=False, max_wind=1):
        """
        Creates a new instance of the coffee environment.

        """
        super().__init__(add_obj_to_start=add_obj_to_start, random_act_prob=random_act_prob)
        self.action_space = Discrete(3)
        self._max_wind = max_wind
        self._create_coord_mapping()
        self._create_transition_function()

        exit_states = {}
        for s in self.object_ids:
            symbol = self.MAP[s]
            exit_states[self.PHI_OBJ_TYPES.index(symbol)] = s

        self.exit_states = exit_states

    def coords_act_transition_distr(self, coords, action):
        row, col = coords
        distr = []
        for wind in range(-self._max_wind, self._max_wind + 1, 1):
            new_row = row
            new_col = col

            vert_move = wind - (action == self.UP) + (action == self.DOWN)
            horiz_move = 1 + (action == self.RIGHT)

            # Check vert move
            direction = -1 if vert_move < 0 else 1
            while vert_move != 0:
                vert_move -= direction
                if (new_row + direction, new_col) not in self.occupied:
                    new_row = min(self.height - 1, new_row + direction)
                    new_row = max(0, new_row)

            # Check horiz move
            direction = -1 if horiz_move < 0 else 1
            while horiz_move != 0:
                horiz_move -= direction
                if (new_row, new_col + direction) not in self.occupied:
                    new_col = min(self.width - 1, new_col + direction)
                    new_col = max(0, new_col)

            entry = ((new_row, new_col), 1.0 / (self._max_wind * 2 + 1))
            distr.append(entry)
        return distr

    def _create_transition_function(self):
        # Basic movement
        self.P = np.zeros((self.s_dim, self.a_dim, self.s_dim))
        for start_s in range(self.s_dim):
            for eff_a in range(self.a_dim):
                start_coords = self.state_to_coords[start_s]
                if start_coords in self.object_ids:
                    self.P[start_s, eff_a, start_s] += 1  # Set transitions in goal states to 1 to pass the sanity check
                    continue
                distr = self.coords_act_transition_distr(coords=start_coords, action=eff_a)
                for end_coords, prob in distr:
                    new_s = self.coords_to_state[end_coords]
                    self.P[start_s, eff_a, new_s] += prob
        # sanity check
        assert np.allclose(np.sum(self.P, axis=2), 1)


class PickupDropoff(GridEnv):
    # MAP = np.array([ [' ', '<', '<', '<', '<', '<', '<', '<', '<', '<', '<',  ' ',  ' '],
    #                  [' ', '<', '<', '<', '<', '<', '<', '<', '<', '<', '<',  ' ',  'A'],
    #                  [' ', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X',  ' ',  ' '],
    #                  [' ', '>', '>', '>', '>', '>', '>', '>', '>', '>', '>',  ' ',  ' '],
    #                  ['H', '>', '>', '>', '>', '>', '>', '>', '>', '>', 'X',  ' ',  ' '],
    #                  [' ', '>', '>', '>', '>', '>', '>', '>', '>', 'X', 'X',  ' ',  ' '],
    #                  [' ', '>', '>', '>', '>', '>', '>', '>', 'X', 'X', 'X',  ' ',  ' '],
    #                  [' ', '>', '>', '>', '>', '>', '>', 'X', 'X', 'X', 'X',  ' ',  ' '],
    #                  [' ', '>', '>', '>', '>', '>', '>', '>', 'X', 'X', 'X',  ' ',  ' '],
    #                  [' ', '>', '>', '>', '>', '>', '>', '>', '>', 'X', 'X',  ' ',  ' '],
    #                  ['C', '>', '>', '>', '>', '>', '>', '>', '>', '>', 'X',  ' ',  ' '],
    #                  [' ', '>', '>', '>', '>', '>', '>', '>', '>', '>', '>',  ' ',  ' '],
    #                  [' ', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X',  ' ',  ' '],
    #                  [' ', '<', '<', '<', '<', '<', '<', '<', '<', '<', '<',  ' ',  'T'],
    #                  [' ', '<', '<', '<', '<', '<', '<', '<', '<', '<', '<',  ' ',  ' '],])

    MAP = np.array([[' ', '<', '<', '<', '<', ' '],
                    [' ', '<', '<', '<', '<', 'A'],
                    [' ', 'X', 'X', 'X', 'X', ' '],
                    [' ', '>', '>', '>', '>', ' '],
                    ['H', '>', '>', '>', '>', ' '],
                    [' ', '>', '>', '>', '>', ' '],
                    ['C', '>', '>', '>', '>', ' '],
                    [' ', '>', '>', '>', '>', ' '],
                    [' ', 'X', 'X', 'X', 'X', ' '],
                    [' ', '<', '<', '<', '<', 'T'],
                    [' ', '<', '<', '<', '<', ' '], ])

    PHI_OBJ_TYPES = ['H', 'C', 'A', 'T']

    """
    New environment.

    """

    def __init__(self, random_act_prob=0.0, add_obj_to_start=False, max_wind=1):
        super().__init__(add_obj_to_start=add_obj_to_start, random_act_prob=random_act_prob)
        self._max_wind = max_wind
        self._create_coord_mapping()
        self._create_transition_function()

        self.exit_states = {}
        for s in self.object_ids:
            symbol = self.MAP[s]
            self.exit_states[self.PHI_OBJ_TYPES.index(symbol)] = s

        if not add_obj_to_start:
            self.initial = [s for s in self.exit_states.values() if self.MAP[s] in ["H", "C"]]

    #
    def coords_act_transition_distr(self, coords, action):
        row, col = coords
        distr = []

        if self.MAP[(row, col)] == ">":

            for wind in range(-self._max_wind, self._max_wind + 1, 1):

                new_row = row
                new_col = col

                vert_move = wind - (action == self.UP) + (action == self.DOWN)
                horiz_move = 1 + (action == self.RIGHT)

                # Check vert move
                direction = -1 if vert_move < 0 else 1
                while vert_move != 0:
                    vert_move -= direction
                    if (new_row + direction, new_col) not in self.occupied:
                        new_row = min(self.height - 1, new_row + direction)
                        new_row = max(0, new_row)

                # Check horiz move
                direction = -1 if horiz_move < 0 else 1
                while horiz_move != 0:
                    horiz_move -= direction
                    if (new_row, new_col + direction) not in self.occupied:
                        new_col = min(self.width - 1, new_col + direction)
                        new_col = max(0, new_col)

                entry = ((new_row, new_col), 1.0 / (self._max_wind * 2 + 1))
                distr.append(entry)

        else:
            next_states = [self.base_movement(coords, a) for a in range(self.a_dim)]
            probs = [1 - self.random_act_prob if a == action else self.random_act_prob / (self.a_dim - 1) for a in
                     range(self.a_dim)]

            distr = list(zip(next_states, probs))

        return distr

    def _create_transition_function(self):
        # Basic movement
        self.P = np.zeros((self.s_dim, self.a_dim, self.s_dim))
        for start_s in range(self.s_dim):
            for eff_a in range(self.a_dim):
                start_coords = self.state_to_coords[start_s]
                # if start_coords in self.object_ids:
                # self.P[start_s, eff_a, start_s] += 1  # Set transitions in goal states to 1 to pass the sanity check
                # continue
                distr = self.coords_act_transition_distr(coords=start_coords, action=eff_a)
                for end_coords, prob in distr:
                    new_s = self.coords_to_state[end_coords]
                    self.P[start_s, eff_a, new_s] += prob
        # sanity check
        assert np.allclose(np.sum(self.P, axis=2), 1)

    def base_movement(self, coords, action):

        row, col = coords

        if action == self.LEFT and self.MAP[row, col] != ">":
            col -= 1
        elif action == self.UP:
            row -= 1
        elif action == self.RIGHT and self.MAP[row, col] != "<":
            col += 1
        elif action == self.DOWN:
            row += 1
        elif action == self.RIGHT:
            return (row, col)
        else:
            raise Exception('bad action {}'.format(action))
        if col < 0 or col >= self.width or row < 0 or row >= self.height or (row, col) in self.occupied:  # no move
            return coords
        else:
            return (row, col)

    def step(self, action):
        # Movement
        old_state = self.state
        old_state_index = self.coords_to_state[old_state]
        new_state_index = np.random.choice(a=self.s_dim, p=self.P[old_state_index, action])
        new_state = self.state_to_coords[new_state_index]

        self.state = new_state

        # Determine features and rewards
        phi = self.features(old_state, action, new_state)
        reward = -1  #np.dot(phi, self.w)
        done = self.is_done(old_state, action, new_state)
        prop = self.MAP[new_state]
        return self.state_to_array(self.state), reward, done, {'phi': phi, 'proposition': prop}

    def reward(self, state):
        return -1
