import numpy as np


def get_rbf_activation_data_old(env, include_threshold=0.01, exclude=None):
    grid_height, grid_width = env.MAP.shape
    rbf_data = {}

    for symbol in sorted(env.COORDS_RBFS.keys()):  # Sort A → B → C
        rbf_data[symbol] = {}
        for i, center_coords in enumerate(env.COORDS_RBFS[symbol]):
            cy, cx = center_coords  # RBF center
            rbf_data[symbol][center_coords] = {}

            # Compute RBF activation for each cell in the grid
            for y in range(grid_height):
                for x in range(grid_width):
                    if exclude and env.MAP[y, x] in exclude:
                        continue
                    activation_value = gaussian_rbf(x, y, cx, cy, d=env.D_RBFS[symbol][i])
                    if activation_value > include_threshold:
                        rbf_data[symbol][center_coords][(y, x)] = activation_value
    return rbf_data, (grid_height, grid_width)


def get_rbf_activation_data(env, include_threshold=0.01, exclude=None):
    grid_height, grid_width = env.MAP.shape
    rbf_data = {}

    for symbol in sorted(env.FEAT_DATA.keys()):  # Sort A → B → C
        rbf_data[symbol] = {}
        for i, feat in enumerate(env.FEAT_DATA[symbol]):
            cy, cx, d = feat  # RBF center
            rbf_data[symbol][feat] = {}

            # Compute RBF activation for each cell in the grid
            for y in range(grid_height):
                for x in range(grid_width):
                    coords = (y, x)
                    if exclude and env.MAP[y, x] in exclude:
                        continue
                    if hasattr(env, "continuous_base_to_continuous_center"):
                        y_centered, x_centered = env.continuous_base_to_continuous_center(np.array(coords))
                    activation_value = gaussian_rbf(x_centered, y_centered, cx, cy, d=d)
                    if activation_value > include_threshold:
                        rbf_data[symbol][feat][coords] = activation_value

    return rbf_data, (grid_height, grid_width)


def get_fourier_activation_data(env, include_threshold=0.01, exclude=None):
    grid_height, grid_width = env.MAP.shape
    activation_data = {}

    for symbol in sorted(env.FEAT_DATA.keys()):  # Sort A → B → C
        activation_data[symbol] = {}
        for i, feat in enumerate(env.FEAT_DATA[symbol]):
            activation_data[symbol][feat] = {}

            # Compute activation for each cell in the grid
            for y in range(grid_height):
                for x in range(grid_width):
                    coords = (y, x)
                    if exclude and env.MAP[y, x] in exclude:
                        continue
                    if hasattr(env, "continuous_base_to_continuous_center"):
                        y_centered, x_centered = env.continuous_base_to_continuous_center(np.array(coords))
                    norm_y, norm_x = normalize_state((y_centered, x_centered), env.low, env.high)
                    activation_value = fourier_features(norm_x, norm_y, feat_data=(feat,))[0]
                    if activation_value > include_threshold:
                        activation_data[symbol][feat][coords] = activation_value
    return activation_data, (grid_height, grid_width)


def gaussian_rbf(x, y, cx, cy, d=1):
    distance_squared = (x - cx) ** 2 + (y - cy) ** 2
    return np.exp(-distance_squared / d ** 2)


def gaussian_rbf_features(x, y, feat_data=((0, 0, 4), (1, 1, 4))):
    """
    Generate combined RBF features.

    Parameters:
        x (float): X-coordinate (can also be np.array for batch).
        y (float): Y-coordinate (same shape as x).
        feat_data (tuple of tuples): (cy, cx, distance), e.g. : ((0, 0, 4), (1, 1, 4))

    Returns:
        np.array: Feature vector of shape (2 * len(directions),) or (..., 2 * len(directions))
    """
    feats = []
    for cy, cx, d in feat_data:
        val = gaussian_rbf(x, y, cx, cy, d)
        feats.append(val)
    return np.array(feats)


def fourier(x, y, fx, fy):
    arg = np.pi * (fx * x + fy * y)
    return (np.cos(arg) + 1) / 2


def inverse_fourier(x, y, fx, fy):
    arg = np.pi * (fx * x + fy * y)
    return 1 - (np.cos(arg) + 1) / 2


def fourier_mirrored_x(x, y, fx, fy):
    """
    Fourier activation function with mirroring in x about 0.5.

    Mirrors the x-coordinate (i.e., replacing x with 1-x) so that
    for the (fx, fy) pair (e.g., (1, 1)), the spatial pattern is flipped horizontally.
    This transformation causes high activations to be shifted to the opposite horizontal
    corners relative to the non-mirrored pattern.

    Parameters:
        x (float or np.array): x-coordinate(s)
        y (float or np.array): y-coordinate(s)
        fx (float): frequency multiplier for x
        fy (float): frequency multiplier for y

    Returns:
        np.array: The Fourier activation computed as (cos(arg) + 1)/2,
                  where arg = π * (fx * (1 - x) + fy * y).
    """
    arg = np.pi * (fx * (1 - x) + fy * y)
    return (np.cos(arg) + 1) / 2


def fourier_mirrored_y(x, y, fx, fy):
    """
    Fourier activation function with mirroring in y about 0.5.

    Mirrors the y-coordinate (i.e., replacing y with 1-y) so that
    for the (fx, fy) pair (e.g., (1, 1)), the spatial pattern is flipped vertically.
    This transformation causes high activations to be shifted to the opposite vertical
    corners relative to the non-mirrored pattern.

    Parameters:
        x (float or np.array): x-coordinate(s)
        y (float or np.array): y-coordinate(s)
        fx (float): frequency multiplier for x
        fy (float): frequency multiplier for y

    Returns:
        np.array: The Fourier activation computed as (cos(arg) + 1)/2,
                  where arg = π * (fx * x + fy * (1 - y)).
    """
    arg = np.pi * (fx * x + fy * (1 - y))
    return (np.cos(arg) + 1) / 2


def fourier_features(x, y, feat_data=((1, 0), (0, 1), (1, 1), (2, 1), (1, 2))):
    """
    Generate combined Fourier features using 2D frequency directions.

    Parameters:
        x (float): X-coordinate (can also be np.array for batch).
        y (float): Y-coordinate (same shape as x).
        feat_data (tuple of tuples): Tuple of (fx, fy) frequency pairs.

    Returns:
        np.array: Feature vector of shape (2 * len(directions),) or (..., 2 * len(directions))
    """
    feat_activations = []
    for feat in feat_data:
        fx, fy = feat[0], feat[1]
        if len(feat) == 3 and feat[2] == 'inv':
            activation = inverse_fourier(x, y, fx, fy)
        elif len(feat) == 3 and feat[2] == 'mirr_x':
            activation = fourier_mirrored_x(x, y, fx, fy)
        elif len(feat) == 3 and feat[2] == 'mirr_y':
            activation = fourier_mirrored_y(x, y, fx, fy)
        else:
            activation = fourier(x, y, fx, fy)
        feat_activations.append(activation)
    return np.array(feat_activations)


def normalize_state(state, low, high):
    """
    Normalize a discrete or continuous state into the [0, 1] range.

    Parameters:
        state (tuple or array-like): The state to normalize, e.g., (y, x).
        low (array-like): The minimum value for each dimension (same shape as state).
        high (array-like): The maximum value for each dimension (same shape as state).

    Returns:
        np.ndarray: A normalized state as a float32 NumPy array with values in [0, 1].

    Notes:
        - Assumes that each dimension in `state` lies within [low[i], high[i]].
        - Does not clip out-of-bounds states or check for division by zero. Use a safe
          version if your bounds may include equal values.
    """
    return (np.array(state) - low) / (high - low)


def compute_q_table(sf_table, w):
    q_table = dict()
    for coords, successor_features in sf_table.items():
        # successor_features is (action_dim, phi_dim)
        q_vals = np.dot(successor_features, w)  # results in (action_dim,) Q-values
        q_table[coords] = q_vals
    return q_table


def convert_map_to_grid(env, custom_mapping=None):
    """
    Converts the environment's MAP (string-based) into a numeric grid for visualization.

    Args:
        env: The `GridEnv` subclass instance (e.g., `OfficeAreas`).
        custom_mapping: Custom color map.

    Returns:
        np.array: A 2D grid where each cell is assigned a numeric value.
    """
    mapping = {
        "X": 4,  # Walls
        " ": 3,  # Empty Space
        "_": 3,  # Start location (same as empty space)
        "A": 0,  # Object A
        "B": 1,  # Object B
        "C": 2,  # Object C
    }
    if custom_mapping is not None:
        mapping = custom_mapping

    grid_numeric = np.zeros(env.MAP.shape, dtype=np.float32)

    for y in range(env.MAP.shape[0]):
        for x in range(env.MAP.shape[1]):
            symbol = env.MAP[y, x]
            grid_numeric[y, x] = mapping.get(symbol, 0)  # Default to 0 if unknown

    return grid_numeric, mapping
