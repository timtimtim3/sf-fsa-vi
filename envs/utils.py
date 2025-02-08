import numpy as np
import matplotlib.pyplot as plt


def visualize_rbfs(env):
    """
    Visualizes the RBF activations over the grid for each RBF center.

    Args:
        env: An instance of `OfficeAreasRBF` environment.
        d: Scaling factor for the RBF (controls spread).
    """
    grid_height, grid_width = env.MAP.shape

    for symbol in sorted(env.COORDS_RBFS.keys()):  # Sort A -> B -> C
        for center_coords in env.COORDS_RBFS[symbol]:
            cy, cx = center_coords  # RBF center
            activation_grid = np.zeros((grid_height, grid_width))

            # Compute RBF activation for each cell in the grid
            for y in range(grid_height):
                for x in range(grid_width):
                    activation_grid[y, x] = gaussian_rbf(x, y, cx, cy, env.d)

            # Plot heatmap
            plt.figure(figsize=(6, 6))
            plt.imshow(activation_grid, cmap="hot", origin="upper", extent=[0, grid_width, 0, grid_height])
            plt.colorbar(label="RBF Activation")
            plt.scatter(cx + 0.5, grid_height - cy - 0.5, color="cyan", s=100,
                        label="RBF Center")  # Mark center
            plt.title(f"RBF Activation for {symbol} at ({cx}, {cy})")
            plt.xlabel("X Coordinate")
            plt.ylabel("Y Coordinate")
            plt.legend()
            plt.grid(False)  # Remove grid lines for better clarity
            plt.show()


def gaussian_rbf(x, y, cx, cy, d=1):
    distance_squared = (x - cx) ** 2 + (y - cy) ** 2
    return np.exp(-distance_squared / d ** 2)


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

    Returns:
        np.array: A 2D grid where each cell is assigned a numeric value.
    """
    mapping = {
        "X": 4,  # Walls (Black)
        " ": 3,  # Empty Space (White)
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
