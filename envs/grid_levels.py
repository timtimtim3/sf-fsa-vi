from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import re


@dataclass
class LevelDataOfficeAreas:
    MAP: np.ndarray
    PHI_OBJ_TYPES: List[str]
    COLOR_MAP: Dict[str, List[float]]


@dataclass
class LevelDataOfficeAreasRBF(LevelDataOfficeAreas):
    RBF_MAP: Optional[np.ndarray] = None
    COORDS_RBFS: Optional[Dict[str, List[Tuple[int, int]]]] = None
    D_RBFS: Optional[Dict[str, List[int]]] = None
    DEFAULT_D_RBFS: Union[int, float] = 1  # Default RBF distance if not specified in the map

    def __post_init__(self):
        has_rbf_map = self.RBF_MAP is not None
        has_rbf_coords = self.COORDS_RBFS is not None and self.D_RBFS is not None

        # Ensure exactly one of the two configurations is provided
        if has_rbf_map == has_rbf_coords:
            raise ValueError("You must provide either (RBF_MAP) or (COORDS_RBFS and D_RBFS), but not both.")

        # Validate dimensions of RBF_MAP if it exists
        if has_rbf_map and self.RBF_MAP.shape != self.MAP.shape:
            raise ValueError(f"RBF_MAP dimensions {self.RBF_MAP.shape} do not match MAP dimensions {self.MAP.shape}.")

        # Validate manually provided COORDS_RBFS and D_RBFS
        if has_rbf_coords:
            self._validate_manual_rbfs()

        # If RBF_MAP is provided, dynamically load COORDS_RBFS and D_RBFS
        if has_rbf_map:
            self._load_rbf_from_map()

        # Validate that RBFs are correctly placed in MAP
        self._validate_rbf_placement()

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
        Otherwise, use DEFAULT_D_RBFS.
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
                        2) else self.DEFAULT_D_RBFS  # Extract distance or use default

                    # Check that the extracted object name is in PHI_OBJ_TYPES
                    if obj_name not in self.PHI_OBJ_TYPES:
                        raise ValueError(
                            f"Found unknown RBF '{cell_value}' at ({row}, {col}) in RBF_MAP, but '{obj_name}' is not in PHI_OBJ_TYPES.")

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
    COLOR_MAP={
        "A": [0.6, 0.3, 0],  # Brown
        "B": [1, 0.6, 0],  # Orange
        "C": [0.5, 0, 0.5],  # Purple
        "X": [0, 0, 0],  # Black (Walls)
        " ": [1, 1, 1],  # White (Empty Space)
        "_": [1, 1, 1],  # White (Starting Area)
    }
)

office_areas_rbf = LevelDataOfficeAreasRBF(
    MAP=office_areas.MAP,
    PHI_OBJ_TYPES=office_areas.PHI_OBJ_TYPES,
    COLOR_MAP=office_areas.COLOR_MAP,
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
)

office_areas_rbf_from_map = LevelDataOfficeAreasRBF(
    MAP=office_areas.MAP,
    PHI_OBJ_TYPES=office_areas.PHI_OBJ_TYPES,
    COLOR_MAP=office_areas.COLOR_MAP,
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
    DEFAULT_D_RBFS=1
)

# Dictionary mapping level names to LevelData objects.
LEVELS = {
    "office_areas": office_areas,
    "office_areas_rbf": office_areas_rbf,
    "office_areas_rbf_from_map": office_areas_rbf_from_map,
}
