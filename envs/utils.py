import numpy as np


def get_rbf_activation_data(env, include_threshold=0.01, exclude=None):
    grid_height, grid_width = env.MAP.shape
    rbf_data = {'A': {(0, 0): {(0, 0): 1, (1, 1): 0.5}, (1, 1): {}}}

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


def gaussian_rbf(x, y, cx, cy, d=1):
    distance_squared = (x - cx) ** 2 + (y - cy) ** 2
    return np.exp(-distance_squared / d ** 2)


def gaussian_rbf_features(x, y, feat_data=((0, 0, 4), (1, 1, 4))):
    """
    Generate combined RBF features.

    Parameters:
        x (float): X-coordinate (can also be np.array for batch).
        y (float): Y-coordinate (same shape as x).
        feat_data (tuple of tuples): (cx, cy, distance), e.g. : ((0, 0, 4), (1, 1, 4))

    Returns:
        np.array: Feature vector of shape (2 * len(directions),) or (..., 2 * len(directions))
    """
    feats = []
    for cx, cy, d in feat_data:
        val = gaussian_rbf(x, y, cx, cy, d)
        feats.append(val)
    return np.array(feats)


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
    feats = []
    for fx, fy in feat_data:
        arg = np.pi * (fx * x + fy * y)
        feats.append(np.sin(arg))
        feats.append(np.cos(arg))
    return np.array(feats)


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
