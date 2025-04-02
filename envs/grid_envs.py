from copy import deepcopy
import numpy as np
import random
import gym
from gym.spaces import Discrete, Box, MultiDiscrete
from abc import ABC, abstractmethod
from envs.utils import gaussian_rbf, fourier_features, normalize_state
from envs.grid_levels import LEVELS


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

        for c in range(self.width):
            for r in range(self.height):
                if self.MAP[r, c] == 'X':
                    self.occupied.add((r, c))
                    continue
                elif self.MAP[r, c] == '_':
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
        self.observation_space = Box(low=np.zeros(
            2), high=np.ones(2), dtype=np.float32)
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

    def get_reward(self, action, phi, w):
        # If the env doesn't allow a terminate action, we return the regular reward
        if not self.terminate_action:
            return np.dot(phi, w)

        if action == self.TERMINATE:
            # Give 0 reward if the agent selects the terminate action in a non-goal state
            if self.old_state not in self.object_ids:
                return 0

         # If the agent did not select the terminal action but the state didn't change,
        # this means an invalid move was made
        if action != self.TERMINATE and self.old_state == self.state:
            return 0

        # In all other cases the agent selected the terminate action in a goal-state or the agent did not select
        # the terminate action, in both cases we return the regular reward
        return np.dot(phi, w)

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

        if isinstance(state, tuple) and len(state) == 2 and all(isinstance(x, int) for x in state):
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

        height, width = self.MAP.shape
        self.observation_space = MultiDiscrete([height, width])

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

        height, width = self.MAP.shape
        self.observation_space = MultiDiscrete([height, width])

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


class FeatureExtractor:
    def __init__(self, feat_data, feat_fn, phi_obj_types, exit_states=None, remove_redundant_features=True,
                 min_activation_thresh=0.1, verbose=True, normalize_states_for_fourier=False):
        self._feat_data = feat_data
        self.feat_fn = feat_fn
        self.phi_obj_types = phi_obj_types
        self.normalize_states_for_fourier = normalize_states_for_fourier

        if remove_redundant_features:
            if exit_states is None:
                raise ValueError("Exit states cannot be None if remove_redundant_features is set to True")
            self._remove_redundant_features(exit_states, min_activation_thresh, verbose)

        self.feat_dim = sum([len(feat_data) for feat_data in self._feat_data.values()])
        self.feat_indices = {}

        self.prop_at_feat_idx = []
        self.feat_indices = {}
        current_index = 0
        for symbol in sorted(self._feat_data.keys()):  # Sort A -> B -> C
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

    def features(self, env, state, action, next_state):
        phi = np.zeros(self.feat_dim, dtype=np.float32)

        # Only assign non-zero values if the next state is a goal state
        is_goal_state = env.is_goal_state(next_state)

        if is_goal_state:
            symbol = env.get_symbol_at_state(next_state)

            if self.normalize_states_for_fourier:
                y, x = normalize_state(next_state, env.low, env.high)
            else:
                y, x = next_state

            feat_vec = self.feat_fn(x, y, self._feat_data[symbol])
            feat_indices = self.feat_indices[symbol]
            phi[feat_indices] = feat_vec
        return phi


class OfficeAreasRBF(GridEnv):
    def __init__(self, add_obj_to_start=False, random_act_prob=0.0, add_empty_to_start=False, only_rbf=False,
                 level_name="office_areas_rbf_from_map", min_activation_thresh=0.1, delete_redundant_rbfs=False,
                 terminate_action=False, reset_probability_goals=None):
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
        self.delete_redundant_rbfs = level.DELETE_REDUNDANT_RBFS if level.DELETE_REDUNDANT_RBFS is not None else (
            delete_redundant_rbfs)

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

        if self.delete_redundant_rbfs:
            self.remove_redundant_rbfs(min_activation_thresh=min_activation_thresh)

        self.rbf_lengths = {symbol: len(coords_list) for symbol, coords_list in self.COORDS_RBFS.items()}
        self.n_rbfs = sum(self.rbf_lengths.values())
        self.w = np.zeros(self.feat_dim)

        # Reserve initial indices for objects
        start_index = 0 if only_rbf else len(self.PHI_OBJ_TYPES)
        self.prop_at_feat_idx = []
        self.rbf_indices = {}
        current_index = start_index
        # Reserve indices in our feature-weight vector for each rbf feature
        for symbol in sorted(self.COORDS_RBFS.keys()):  # Sort A -> B -> C
            self.rbf_indices[symbol] = {}
            for center_coords in self.COORDS_RBFS[symbol]:
                self.rbf_indices[symbol][center_coords] = current_index
                current_index += 1

                self.prop_at_feat_idx.append(symbol)

        height, width = self.MAP.shape
        self.observation_space = MultiDiscrete([height, width])

    def _create_transition_function(self):
        self._create_transition_function_base()

    # def is_done(self, state, action, next_state):
    #     if self.terminate_action:
    #         if action == self.TERMINATE:
    #             return True
    #         return False
    #     else:
    #         return next_state in self.object_ids

    def is_done(self, state, action, next_state):
        if self.terminate_action:
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


class OfficeAreasFeatures(GridEnv):
    def __init__(self, add_obj_to_start=False, random_act_prob=0.0, add_empty_to_start=False,
                 level_name="office_areas_fourier", min_activation_thresh=0.1, remove_redundant_features=False,
                 terminate_on_enter_different_area=False):
        # Load level data from the external LEVELS dictionary.
        level = LEVELS[level_name]

        # Set level-specific attributes.
        self.MAP = deepcopy(level.MAP)
        self.PHI_OBJ_TYPES = deepcopy(level.PHI_OBJ_TYPES)
        self.FEAT_DATA = deepcopy(level.FEAT_DATA)
        self.RENDER_COLOR_MAP = level.RENDER_COLOR_MAP
        self.QVAL_COLOR_MAP = level.QVAL_COLOR_MAP
        self.FEAT_FN = level.FEAT_FN

        self.terminate_on_enter_different_area = terminate_on_enter_different_area

        super().__init__(add_obj_to_start=add_obj_to_start, random_act_prob=random_act_prob,
                         add_empty_to_start=add_empty_to_start, init_w=False)
        self._create_coord_mapping()
        self._create_transition_function()
        height, width = self.MAP.shape
        self.observation_space = MultiDiscrete([height, width])

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

        self.feature_extractor = (
            FeatureExtractor(self.FEAT_DATA, self.FEAT_FN, self.PHI_OBJ_TYPES, exit_states=exit_states,
                             remove_redundant_features=remove_redundant_features,
                             min_activation_thresh=min_activation_thresh, verbose=True,
                             **feature_extractor_kwargs))
        self.prop_at_feat_idx = self.feature_extractor.prop_at_feat_idx
        self.low, self.high = self.get_observation_bounds()

        self.w = np.zeros(self.feat_dim)


    def _create_transition_function(self):
        self._create_transition_function_base()

    def is_done(self, state, action, next_state):
        if self.terminate_on_enter_different_area:
            # If we don't enter an Area, we do not terminate
            if next_state not in self.object_ids:
                return False

            # If we enter an Area, we terminate if we came from a different Area (or empty space)
            last_symbol = self.MAP[state]
            next_symbol = self.MAP[next_state]

            # If we entered the same Area as the one we came from, e.g. 'B' and 'B', we do not terminate
            if last_symbol == next_symbol:
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
