from copy import deepcopy
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import re
from envs.utils import gaussian_rbf_features, fourier_features


def create_rbf_grid(x_min, x_max, y_min, y_max, phi_obj_types, x_feat_count=4, y_feat_count=4, d_rbfs=4,
                    round_to_integers=True, start_at_extremes=True):
    x_dist = abs(x_min - x_max)
    y_dist = abs(y_min - y_max)
    if start_at_extremes:
        dist_between_feat_x = x_dist / (x_feat_count - 1)
        dist_between_feat_y = y_dist / (y_feat_count - 1)
        start_idx = 0
    else:
        dist_between_feat_x = x_dist / (x_feat_count + 1)
        dist_between_feat_y = y_dist / (y_feat_count + 1)
        start_idx = 1

    coordinates = []
    for i in range(start_idx, y_feat_count + start_idx):
        for j in range(start_idx, x_feat_count + start_idx):
            curr_y = y_min + i * dist_between_feat_y
            curr_x = x_min + j * dist_between_feat_x
            if round_to_integers:
                curr_x = int(round(curr_x))
                curr_y = int(round(curr_y))
            coordinates.append((curr_y, curr_x))

    all_d_rbfs = [d_rbfs for _ in range(len(coordinates))]
    coords_rbfs = {symbol: coordinates for symbol in phi_obj_types}
    d_rbfs_dict = {symbol: all_d_rbfs for symbol in phi_obj_types}
    return coords_rbfs, d_rbfs_dict


@dataclass
class LevelDataOfficeAreas:
    MAP: np.ndarray
    PHI_OBJ_TYPES: List[str]
    RENDER_COLOR_MAP: Dict[str, List[float]]
    QVAL_COLOR_MAP: Optional[Dict[str, int]] = None


@dataclass
class LevelDataOfficeAreasRBF(LevelDataOfficeAreas):
    RBF_MAP: Optional[np.ndarray] = None
    COORDS_RBFS: Optional[Dict[str, List[Tuple[int, int]]]] = None
    D_RBFS: Optional[Dict[str, List[int]]] = None
    MAP_DEFAULT_D_RBFS: Union[int, float] = 1  # Default RBF distance if not specified in the map
    CREATE_RBF_GRID: bool = False
    X_FEAT_COUNT: Optional[int] = None
    Y_FEAT_COUNT: Optional[int] = None
    GRID_D_RBFS: Optional[Union[int, float]] = None
    REMOVE_REDUNDANT_FEAT: Optional[bool] = None
    FEAT_FN = staticmethod(gaussian_rbf_features)

    def __post_init__(self):
        has_rbf_map = self.RBF_MAP is not None
        has_rbf_coords = self.COORDS_RBFS is not None and self.D_RBFS is not None
        option_count = int(has_rbf_map) + int(has_rbf_coords) + int(self.CREATE_RBF_GRID)

        # Ensure exactly one of the two configurations is provided
        if option_count == 0 or option_count > 1:
            raise ValueError("You must provide either (RBF_MAP) or (COORDS_RBFS and D_RBFS) or (CREATE_RBF_GRID=True), "
                             "but you can't use more than one of them.")

        # Validate dimensions of RBF_MAP if it exists
        if has_rbf_map and self.RBF_MAP.shape != self.MAP.shape:
            raise ValueError(f"RBF_MAP dimensions {self.RBF_MAP.shape} do not match MAP dimensions {self.MAP.shape}.")

        # Validate manually provided COORDS_RBFS and D_RBFS
        if has_rbf_coords:
            self._validate_manual_rbfs()

        # If RBF_MAP is provided, dynamically load COORDS_RBFS and D_RBFS
        if has_rbf_map:
            self._load_rbf_from_map()

        if self.CREATE_RBF_GRID:
            kwargs = {"x_feat_count": self.X_FEAT_COUNT, "y_feat_count": self.Y_FEAT_COUNT, "d_rbfs": self.GRID_D_RBFS}
            kwargs_not_none = {key: value for key, value in kwargs.items() if value is not None}

            self.COORDS_RBFS, self.D_RBFS = create_rbf_grid(0, len(self.MAP[0]) - 1, 0, len(self.MAP) - 1,
                                                            self.PHI_OBJ_TYPES, **kwargs_not_none)

        self.FEAT_DATA = {}
        for symbol in self.COORDS_RBFS.keys():
            coords = self.COORDS_RBFS[symbol]
            distances = self.D_RBFS[symbol]

            feat_data_list = []
            for coord, distance in zip(coords, distances):
                feat_data = (coord[0], coord[1], distance)
                feat_data_list.append(feat_data)
            self.FEAT_DATA[symbol] = tuple(feat_data_list)

    def _validate_manual_rbfs(self):
        """
        Ensures that COORDS_RBFS and D_RBFS have matching keys and lists of the same length.
        """
        if set(self.COORDS_RBFS.keys()) != set(self.D_RBFS.keys()):
            raise ValueError("Mismatch between COORDS_RBFS and D_RBFS keys. Both must have the same RBF names.")

        for key in self.COORDS_RBFS:
            coord_len = len(self.COORDS_RBFS[key])
            dist_len = len(self.D_RBFS[key])

            if coord_len != dist_len:
                raise ValueError(f"Length mismatch for RBF '{key}': {coord_len} coordinates but {dist_len} distances.")

    def _load_rbf_from_map(self):
        """
        Extracts COORDS_RBFS and D_RBFS from RBF_MAP.
        If an RBF entry in the map includes a number (e.g., "A_RBF_1"), use that as the distance.
        Otherwise, use MAP_DEFAULT_D_RBFS.
        """
        self.COORDS_RBFS = {}  # Initialize empty dict
        self.D_RBFS = {}  # Initialize empty dict

        # Regular expression pattern to match names like "A_RBF" or "A_RBF_3"
        pattern = re.compile(r"^([A-Z])_RBF(?:_(\d+))?$")

        # Scan through RBF_MAP and extract RBF centers and distances
        for row in range(self.RBF_MAP.shape[0]):
            for col in range(self.RBF_MAP.shape[1]):
                cell_value = self.RBF_MAP[row, col]

                match = pattern.match(cell_value)
                if match:
                    obj_name = match.group(1)  # Extract "A", "B", etc.
                    distance = int(match.group(2)) if match.group(
                        2) else self.MAP_DEFAULT_D_RBFS  # Extract distance or use default

                    # Check that the extracted object name is in PHI_OBJ_TYPES
                    if obj_name not in self.PHI_OBJ_TYPES:
                        raise ValueError(
                            f"Found unknown RBF '{cell_value}' at ({row}, {col}) in RBF_MAP, but '{obj_name}' is not "
                            f"in PHI_OBJ_TYPES.")

                    # Add to COORDS_RBFS
                    if obj_name not in self.COORDS_RBFS:
                        self.COORDS_RBFS[obj_name] = []
                        self.D_RBFS[obj_name] = []

                    self.COORDS_RBFS[obj_name].append((row, col))
                    self.D_RBFS[obj_name].append(distance)

        # Convert lists to tuples for immutability
        self.COORDS_RBFS = {k: tuple(v) for k, v in self.COORDS_RBFS.items()}
        self.D_RBFS = {k: tuple(v) for k, v in self.D_RBFS.items()}

    def _validate_rbf_placement(self):
        """
        Ensures that each RBF coordinate is correctly placed on the corresponding PHI_OBJ_TYPES symbol in MAP.
        We might want to remove this check since it's not strictly needed to have an RBF placed on top of the area.
        """
        for phi_symbol, coords_list in self.COORDS_RBFS.items():
            for row, col in coords_list:
                if self.MAP[row, col] != phi_symbol:
                    raise ValueError(
                        f"Incorrect RBF placement at ({row}, {col}): expected '{phi_symbol}' in MAP, "
                        f"but found '{self.MAP[row, col]}'."
                    )


@dataclass
class LevelDataOfficeAreasFourier(LevelDataOfficeAreas):
    FREQUENCY_PAIRS: Optional[Tuple[Union[Tuple[int, int], Tuple[int, int, str]], ...]] = \
        ((1, 0), (0, 1), (1, 1), (2, 1), (1, 2))
    FEAT_FN = staticmethod(fourier_features)
    NORMALIZE_STATES_FOR_FOURIER = True
    NORMALIZE_FEATURES = True
    REMOVE_REDUNDANT_FEAT: Optional[bool] = True

    def __post_init__(self):
        self.FEAT_DATA = {symbol: deepcopy(self.FREQUENCY_PAIRS) for symbol in self.PHI_OBJ_TYPES}


office_areas = LevelDataOfficeAreas(
    MAP=np.array([
        [' ', ' ', ' ', 'B', 'B', 'X', 'C', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', 'B', 'B', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', 'A', 'A', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', 'B', ' ', '_'],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', 'B', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', 'B', 'B', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'B', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'B', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'B', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    ]),
    PHI_OBJ_TYPES=['A', 'B', 'C'],
    RENDER_COLOR_MAP={
        "A": [0.6, 0.3, 0],  # Brown
        "B": [1, 0.6, 0],  # Orange
        "C": [0.5, 0, 0.5],  # Purple
        "X": [0, 0, 0],  # Black (Walls)
        " ": [1, 1, 1],  # White (Empty Space)
        "_": [1, 1, 1],  # White (Starting Area)
    },
    QVAL_COLOR_MAP={
        "X": 4,  # Walls
        " ": 3,  # Empty Space
        "_": 3,  # Start location (same as empty space)
        "A": 0,  # Object A
        "B": 1,  # Object B
        "C": 2,  # Object C
    }
)

office_areas_simple = LevelDataOfficeAreas(
    MAP=np.array([
        [' ', ' ', ' ', 'B', ' ', 'X', 'C', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', 'B', 'B', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', 'A', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', ' ', ' ', '_'],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'B', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'B', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    ]),
    PHI_OBJ_TYPES=office_areas.PHI_OBJ_TYPES,
    RENDER_COLOR_MAP=office_areas.RENDER_COLOR_MAP,
    QVAL_COLOR_MAP=office_areas.QVAL_COLOR_MAP
)

office_areas_rbf = LevelDataOfficeAreasRBF(
    MAP=office_areas.MAP,
    PHI_OBJ_TYPES=office_areas.PHI_OBJ_TYPES,
    RENDER_COLOR_MAP=office_areas.RENDER_COLOR_MAP,
    COORDS_RBFS={
        'A': [(5, 4)],
        'B': [(0, 3), (3, 9), (10, 10)],
        'C': [(0, 6)]
    },
    D_RBFS={
        'A': [1],
        'B': [1, 4, 4],
        'C': [1]
    },
    QVAL_COLOR_MAP=office_areas.QVAL_COLOR_MAP
)

office_areas_rbf_from_map = LevelDataOfficeAreasRBF(
    MAP=office_areas.MAP,
    PHI_OBJ_TYPES=office_areas.PHI_OBJ_TYPES,
    RENDER_COLOR_MAP=office_areas.RENDER_COLOR_MAP,
    RBF_MAP=np.array([
        [' ', ' ', ' ', 'B_RBF', 'B', 'X', 'C_RBF_1', ' ', ' ', ' ', ' ', ' ', '_'],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', 'B', 'B', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B_RBF_4', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', 'A', 'A_RBF_1', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', 'B', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', 'B', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', 'B_RBF_4', 'B', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'B', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'B', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'B', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    ]),
    MAP_DEFAULT_D_RBFS=1,
    QVAL_COLOR_MAP=office_areas.QVAL_COLOR_MAP
)

office_areas_rbf_from_map_favorable = LevelDataOfficeAreasRBF(
    MAP=office_areas.MAP,
    PHI_OBJ_TYPES=office_areas.PHI_OBJ_TYPES,
    RENDER_COLOR_MAP=office_areas.RENDER_COLOR_MAP,
    RBF_MAP=np.array([
        [' ', ' ', ' ', 'B_RBF', 'B', 'X', 'C_RBF_1', ' ', ' ', ' ', ' ', ' ', '_'],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', 'B_RBF_4', 'B', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', 'A', 'A_RBF_1', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B_RBF_4', 'B', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', 'B', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', 'B', 'B', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'B', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'B', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'B_RBF_4', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    ]),
    MAP_DEFAULT_D_RBFS=1,
    QVAL_COLOR_MAP=office_areas.QVAL_COLOR_MAP
)

office_areas_rbf_goals_apart = LevelDataOfficeAreasRBF(
    MAP=np.array([
        ['B', 'B', ' ', ' ', ' ', 'X', 'C', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', 'B', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        ['A', 'A', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', 'B', ' ', '_'],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', 'B', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', 'B', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', 'B', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', 'B', 'B', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'B', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    ]),
    PHI_OBJ_TYPES=office_areas.PHI_OBJ_TYPES,
    RENDER_COLOR_MAP=office_areas.RENDER_COLOR_MAP,
    RBF_MAP=np.array([
        ['B_RBF', 'B', ' ', ' ', ' ', 'X', 'C_RBF', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', 'B', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B_RBF', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        ['A_RBF', 'A', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', 'B', ' ', '_'],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', 'B', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', 'B', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', 'B', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', 'B_RBF', 'B', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'B', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    ]),
    MAP_DEFAULT_D_RBFS=3,
    QVAL_COLOR_MAP=office_areas.QVAL_COLOR_MAP
)

office_areas_fourier_goals_apart = LevelDataOfficeAreasFourier(
    MAP=office_areas_rbf_goals_apart.MAP,
    PHI_OBJ_TYPES=office_areas.PHI_OBJ_TYPES,
    RENDER_COLOR_MAP=office_areas.RENDER_COLOR_MAP,
    QVAL_COLOR_MAP=office_areas.QVAL_COLOR_MAP,
    FREQUENCY_PAIRS=((1, 0), (0, 1), (1, 1))
)

office_areas_fourier_goals_apart_5_feat = LevelDataOfficeAreasFourier(
    MAP=office_areas_rbf_goals_apart.MAP,
    PHI_OBJ_TYPES=office_areas.PHI_OBJ_TYPES,
    RENDER_COLOR_MAP=office_areas.RENDER_COLOR_MAP,
    QVAL_COLOR_MAP=office_areas.QVAL_COLOR_MAP,
    FREQUENCY_PAIRS=((1, 0), (0, 1), (1, 1), (2, 1), (1, 2))
)

office_areas_fourier_goals_apart_inv = LevelDataOfficeAreasFourier(
    MAP=office_areas_rbf_goals_apart.MAP,
    PHI_OBJ_TYPES=office_areas.PHI_OBJ_TYPES,
    RENDER_COLOR_MAP=office_areas.RENDER_COLOR_MAP,
    QVAL_COLOR_MAP=office_areas.QVAL_COLOR_MAP,
    FREQUENCY_PAIRS=((1, 0), (0, 1), (1, 1), (1, 0, 'inv'), (0, 1, 'inv'), (1, 1, 'inv'))
)

office_areas_fourier_goals_apart_mirr = LevelDataOfficeAreasFourier(
    MAP=office_areas_rbf_goals_apart.MAP,
    PHI_OBJ_TYPES=office_areas.PHI_OBJ_TYPES,
    RENDER_COLOR_MAP=office_areas.RENDER_COLOR_MAP,
    QVAL_COLOR_MAP=office_areas.QVAL_COLOR_MAP,
    FREQUENCY_PAIRS=((1, 0), (1, 1, 'mirr_x'))
)

office_areas_rbf_grids = LevelDataOfficeAreasRBF(
    MAP=office_areas_rbf_goals_apart.MAP,
    PHI_OBJ_TYPES=office_areas.PHI_OBJ_TYPES,
    RENDER_COLOR_MAP=office_areas.RENDER_COLOR_MAP,
    QVAL_COLOR_MAP=office_areas.QVAL_COLOR_MAP,
    CREATE_RBF_GRID=True,
    X_FEAT_COUNT=3,
    Y_FEAT_COUNT=3,
    GRID_D_RBFS=3,
    REMOVE_REDUNDANT_FEAT=True
)

office_areas_rbf_semi_circle = LevelDataOfficeAreasRBF(
    MAP=np.array([
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', 'B', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', 'A', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'A', ' ', 'A', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', 'A', ' ', ' ', ' ', 'A', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', 'A', ' ', ' ', ' ', 'A', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', 'A', ' ', ' ', ' ', 'A', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', 'A', ' ', ' ', ' ', ' ', ' ', 'A', ' ', ' ', ' '],
        [' ', ' ', ' ', 'A', ' ', ' ', ' ', ' ', ' ', 'A', ' ', ' ', ' '],
        [' ', ' ', ' ', 'A', ' ', ' ', ' ', ' ', ' ', 'A', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', '_', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    ]),
    PHI_OBJ_TYPES=['A', 'B'],
    RENDER_COLOR_MAP=office_areas.RENDER_COLOR_MAP,
    RBF_MAP=np.array([
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', 'B_RBF', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', 'A', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'A', ' ', 'A', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', 'A', ' ', ' ', ' ', 'A', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', 'A_RBF', ' ', ' ', ' ', 'A_RBF', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', 'A', ' ', ' ', ' ', 'A', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', 'A', ' ', ' ', ' ', ' ', ' ', 'A', ' ', ' ', ' '],
        [' ', ' ', ' ', 'A', ' ', ' ', ' ', ' ', ' ', 'A', ' ', ' ', ' '],
        [' ', ' ', ' ', 'A', ' ', ' ', ' ', ' ', ' ', 'A', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', '_', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    ]),
    MAP_DEFAULT_D_RBFS=4,
    QVAL_COLOR_MAP={
        " ": 2,  # Empty Space
        "_": 2,  # Start location (same as empty space)
        "A": 0,  # Object A
        "B": 1,  # Object B
    }
)

office_areas_rbf_fat_small = LevelDataOfficeAreasRBF(
    MAP=np.array([
        [' ', ' ', ' ', 'C', ' ', ' ', ' '],
        [' ', ' ', 'B', 'B', 'B', ' ', ' '],
        [' ', ' ', 'B', 'B', 'B', ' ', ' '],
        [' ', ' ', 'B', 'B', 'B', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', 'A', ' ', ' ', ' '],
        [' ', ' ', ' ', '_', ' ', ' ', ' '],
    ]),
    PHI_OBJ_TYPES=office_areas.PHI_OBJ_TYPES,
    RENDER_COLOR_MAP=office_areas.RENDER_COLOR_MAP,
    RBF_MAP=np.array([
        [' ', ' ', ' ', 'C_RBF', ' ', ' ', ' '],
        [' ', ' ', 'B', 'B', 'B', ' ', ' '],
        [' ', ' ', 'B', 'B_RBF', 'B', ' ', ' '],
        [' ', ' ', 'B', 'B', 'B', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', 'A_RBF', ' ', ' ', ' '],
        [' ', ' ', ' ', '_', ' ', ' ', ' '],
    ]),
    MAP_DEFAULT_D_RBFS=2,
    QVAL_COLOR_MAP=office_areas.QVAL_COLOR_MAP
)

office_areas_fourier_fat_small = LevelDataOfficeAreasFourier(
    MAP=office_areas_rbf_fat_small.MAP,
    PHI_OBJ_TYPES=office_areas.PHI_OBJ_TYPES,
    RENDER_COLOR_MAP=office_areas.RENDER_COLOR_MAP,
    QVAL_COLOR_MAP=office_areas.QVAL_COLOR_MAP,
    FREQUENCY_PAIRS=((1, 0), (0, 1), (1, 1), (2, 1), (1, 2))
)

office_areas_rbf_fat = LevelDataOfficeAreasRBF(
    MAP=np.array([
        [' ', ' ', ' ', ' ', ' ', ' ', 'C', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', 'B', 'B', 'B', 'B', 'B', 'B', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', 'B', 'B', 'B', 'B', 'B', 'B', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', 'B', 'B', 'B', 'B', 'B', 'B', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', 'B', 'B', 'B', 'B', 'B', 'B', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', 'B', 'B', 'B', 'B', 'B', 'B', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', 'B', 'B', 'B', 'B', 'B', 'B', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', 'B', 'B', 'B', 'B', 'B', 'B', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', 'A', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', '_', ' ', ' ', ' ', ' ', ' ', ' '],
    ]),
    PHI_OBJ_TYPES=office_areas.PHI_OBJ_TYPES,
    RENDER_COLOR_MAP=office_areas.RENDER_COLOR_MAP,
    RBF_MAP=np.array([
        [' ', ' ', ' ', ' ', ' ', ' ', 'C_RBF', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', 'B', 'B', 'B', 'B', 'B', 'B', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', 'B', 'B', 'B', 'B', 'B', 'B', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', 'B', 'B', 'B', 'B', 'B', 'B', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', 'B', 'B', 'B', 'B_RBF', 'B', 'B', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', 'B', 'B', 'B', 'B', 'B', 'B', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', 'B', 'B', 'B', 'B', 'B', 'B', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', 'B', 'B', 'B', 'B', 'B', 'B', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', 'A_RBF', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', '_', ' ', ' ', ' ', ' ', ' ', ' '],
    ]),
    MAP_DEFAULT_D_RBFS=3,
    QVAL_COLOR_MAP=office_areas.QVAL_COLOR_MAP
)

office_areas_rbf_no_edge = LevelDataOfficeAreasRBF(
    MAP=np.array([
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', '_'],
        [' ', 'B', ' ', ' ', ' ', 'X', ' ', 'C', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        [' ', 'A', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', 'B', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', 'B', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', 'B', 'B', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'B', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'B', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'B', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    ]),
    PHI_OBJ_TYPES=office_areas.PHI_OBJ_TYPES,
    RENDER_COLOR_MAP=office_areas.RENDER_COLOR_MAP,
    RBF_MAP=np.array([
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', '_'],
        [' ', 'B_RBF_1', ' ', ' ', ' ', 'X', ' ', 'C_RBF_1', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B_RBF_3', ' ', ' ', ' '],
        [' ', 'A_RBF_1', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'B', 'B', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', 'B', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', 'B_RBF_3', 'B', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'B', ' '],
        [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'B', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'B', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    ]),
    MAP_DEFAULT_D_RBFS=1,
    QVAL_COLOR_MAP=office_areas.QVAL_COLOR_MAP
)

office_areas_fourier_detour = LevelDataOfficeAreasFourier(
    MAP=np.array([
        ['A', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        ['B', 'B', 'B', 'B', 'B', 'B', 'B', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        ['_', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    ]),
    PHI_OBJ_TYPES=['A', 'B'],
    RENDER_COLOR_MAP=office_areas.RENDER_COLOR_MAP,
    QVAL_COLOR_MAP=office_areas.QVAL_COLOR_MAP,
    FREQUENCY_PAIRS=((0, 1),)
)

office_areas_detour = LevelDataOfficeAreas(
    MAP=np.array([
        ['A', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        ['_', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    ]),
    PHI_OBJ_TYPES=['A', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
    RENDER_COLOR_MAP=office_areas.RENDER_COLOR_MAP,
    QVAL_COLOR_MAP=office_areas.QVAL_COLOR_MAP,
)

# Dictionary mapping level names to LevelData objects.
LEVELS = {
    "office_areas": office_areas,
    "office_areas_rbf": office_areas_rbf,
    "office_areas_rbf_from_map": office_areas_rbf_from_map,
    "office_areas_rbf_grids": office_areas_rbf_grids,
    "office_areas_rbf_semi_circle": office_areas_rbf_semi_circle,
    "office_areas_simple": office_areas_simple,
    "office_areas_rbf_from_map_favorable": office_areas_rbf_from_map_favorable,
    "office_areas_rbf_goals_apart": office_areas_rbf_goals_apart,
    "office_areas_rbf_fat_small": office_areas_rbf_fat_small,
    "office_areas_rbf_fat": office_areas_rbf_fat,
    "office_areas_fourier_fat_small": office_areas_fourier_fat_small,
    "office_areas_fourier_goals_apart": office_areas_fourier_goals_apart,
    "office_areas_rbf_no_edge": office_areas_rbf_no_edge,
    "office_areas_fourier_goals_apart_5_feat": office_areas_fourier_goals_apart_5_feat,
    "office_areas_fourier_goals_apart_inv": office_areas_fourier_goals_apart_inv,
    "office_areas_fourier_goals_apart_mirr": office_areas_fourier_goals_apart_mirr,
    "office_areas_fourier_detour": office_areas_fourier_detour,
    "office_areas_detour": office_areas_detour
}
