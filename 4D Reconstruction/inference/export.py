"""Export reconstructed point clouds to simulator-friendly formats.

Currently supports:
    * Bounding-box URDF (a single rigid link with a box collision/visual)
    * Stub USD (text file describing the bounding box; for real USD authoring
      use the ``usd-core`` package once it's available on your platform).

These are deliberate v1 implementations as called out in the project plan;
joint constraints and articulated structures are deferred to year 2.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from utils.io_utils import load_npz_tracks
from utils.logger import get_logger

logger = get_logger(__name__)


def _aabb(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return min, max, size of the axis-aligned bounding box."""
    valid = points[~np.isnan(points).any(axis=-1)]
    if valid.size == 0:
        raise RuntimeError("No valid points to compute bbox.")
    mn = valid.min(axis=0)
    mx = valid.max(axis=0)
    return mn, mx, (mx - mn)


def export_urdf(tracks: np.ndarray, out_path: Path, name: str = "scene") -> Path:
    """Write a minimal URDF describing the scene's AABB."""
    flat = tracks.reshape(-1, 3)
    mn, mx, size = _aabb(flat)
    centre = (mn + mx) / 2.0

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"""<?xml version="1.0"?>
<robot name="{name}">
  <link name="{name}_link">
    <visual>
      <origin xyz="{centre[0]:.4f} {centre[1]:.4f} {centre[2]:.4f}" rpy="0 0 0"/>
      <geometry>
        <box size="{size[0]:.4f} {size[1]:.4f} {size[2]:.4f}"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="{centre[0]:.4f} {centre[1]:.4f} {centre[2]:.4f}" rpy="0 0 0"/>
      <geometry>
        <box size="{size[0]:.4f} {size[1]:.4f} {size[2]:.4f}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
</robot>
""",
        encoding="utf-8",
    )
    logger.info("Wrote URDF (%s) bbox=%s", out_path, size.tolist())
    return out_path


def export_usd(tracks: np.ndarray, out_path: Path, name: str = "scene") -> Path:
    """Write a stub USDA file describing the scene's AABB."""
    flat = tracks.reshape(-1, 3)
    mn, mx, size = _aabb(flat)
    centre = (mn + mx) / 2.0

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"""#usda 1.0
def Xform "{name}"
{{
    def Cube "bbox"
    {{
        double size = {float(size.max()):.4f}
        float3 xformOp:translate = ({centre[0]:.4f}, {centre[1]:.4f}, {centre[2]:.4f})
        float3 xformOp:scale = ({size[0] / 2:.4f}, {size[1] / 2:.4f}, {size[2] / 2:.4f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }}
}}
""",
        encoding="utf-8",
    )
    logger.info("Wrote USDA stub (%s) bbox=%s", out_path, size.tolist())
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Export reconstructed scene assets.")
    p.add_argument("--tracks", required=True, type=Path,
                   help="Path to .npz file containing 3D tracks.")
    p.add_argument("--format", choices=["urdf", "usd"], default="urdf")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--name", default="scene")
    args = p.parse_args()

    tracks = load_npz_tracks(args.tracks)["tracks"]
    if args.format == "urdf":
        export_urdf(tracks, args.output, args.name)
    else:
        export_usd(tracks, args.output, args.name)


if __name__ == "__main__":
    main()
