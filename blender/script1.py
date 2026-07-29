import bpy
from pathlib import Path
from dataclasses import dataclass
from math import radians

import torch
import random
import zarr
import time

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset


class QuickDatasetStraight(IterableDataset):
    def __init__(self, path, episode_idx=0, window_size=12):
        self.path = path
        self.window_size = window_size

        root = zarr.open(
            zarr.storage.LocalStore(path),
            mode="r"
        )

        self.data = root["episodes"]

        self.episode_idx = episode_idx
        self.seq_len = self.data.shape[1]


    def __len__(self):
        return self.seq_len - self.window_size + 1


    def __iter__(self):

        root = zarr.open(
            zarr.storage.LocalStore(self.path),
            mode="r"
        )

        data = root["episodes"]

        for i in range(len(self)):

            window = data[
                self.episode_idx,
                i:i+self.window_size,
                :
            ].astype(np.float32)

            yield torch.from_numpy(window)


# =============================================================================
# Configuration
# =============================================================================

STL_DIR = Path("~/Thesis/alpha_drone_mk2_stl_corrected").expanduser()


@dataclass
class MeshConfig:
    file: str
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    location: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = 1.0


MESHES = {
    "body": MeshConfig(
        file="body.stl",
    ),

    "prop_fr": MeshConfig(
        file="prop_1.stl",
        rotation=(0, 0, 0),
    ),

    "prop_fl": MeshConfig(
        file="prop_2.stl",
        rotation=(0, 0, 0),
    ),

    "prop_rl": MeshConfig(
        file="prop_1.stl",
        rotation=(0, 0, 0),
    ),

    "prop_rr": MeshConfig(
        file="prop_2.stl",
        rotation=(0, 0, 0),
    ),
}


# -----------------------------------------------------------------------------
# Propeller positions relative to body frame
# -----------------------------------------------------------------------------

# These are example values in meters.
# Replace with the actual motor locations from Isaac Lab / CAD.

MOTOR_POSITIONS = {
    "prop_fr": ( 0.208,  0.225, 0.05),
    "prop_fl": ( 0.208, -0.225, 0.05),
    "prop_rl": (-0.228, -0.225, 0.05),
    "prop_rr": (-0.228,  0.225, 0.05),
}


# =============================================================================
# Helper functions
# =============================================================================

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    bpy.ops.outliner.orphans_purge(do_recursive=True)


def import_stl(filepath: Path, name: str, cfg: MeshConfig):

    bpy.ops.object.select_all(action="DESELECT")

    bpy.ops.wm.stl_import(filepath=str(filepath))

    obj = bpy.context.selected_objects[0]

    obj.name = name

    obj.location = cfg.location
    obj.rotation_euler = tuple(
        radians(angle) for angle in cfg.rotation
    )

    obj.scale = (
        cfg.scale,
        cfg.scale,
        cfg.scale,
    )

    return obj


def parent_propellers(objects):
    """
    Place propellers relative to body and attach them.
    """

    body = objects["body"]

    for prop_name, position in MOTOR_POSITIONS.items():

        prop = objects[prop_name]

        # position in body coordinates
        prop.location = position

        # make it follow the body
        prop.parent = body


# =============================================================================
# Main
# =============================================================================

clear_scene()

objects = {}

# Import meshes
for name, cfg in MESHES.items():

    filepath = STL_DIR / cfg.file

    if not filepath.exists():
        raise FileNotFoundError(filepath)

    objects[name] = import_stl(
        filepath,
        name,
        cfg,
    )


# Attach props to body
parent_propellers(objects)


print("\nImported drone:")
print("-" * 40)

for name, obj in objects.items():
    print(
        f"{name:10s} "
        f"location={tuple(round(x,3) for x in obj.location)}"
    )

print("-" * 40)
print("Done.")

dataset = QuickDatasetStraight(path=f"SITL_DIR/patient_one_data.zarr/",episode_idx=0,window_size=1)
loader = DataLoader(dataset, batch_size=None, num_workers=0)
for d in loader:
    pos = d[0, :3]
    quat = d[0, 6:10]


