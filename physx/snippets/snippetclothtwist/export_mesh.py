"""Export the reference example's rest mesh; run with Newton's Python environment."""

import argparse
import math
from pathlib import Path

import numpy as np
import warp as wp
import warp.examples
from pxr import Usd

import newton
import newton.usd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    stage = Usd.Stage.Open(str(Path(warp.examples.get_asset_directory()) / "square_cloth.usd"))
    mesh = newton.usd.get_mesh(stage.GetPrimAtPath("/root/cloth/cloth"))
    # Use the same builder transform and mass computation as the reference.
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_from_axis_angle(wp.vec3(0, 0, 1), math.pi / 2),
        scale=0.01,
        vertices=[wp.vec3(v) for v in mesh.vertices],
        indices=mesh.indices,
        vel=wp.vec3(0.0),
        density=0.2,
        tri_ke=1.0e3,
        tri_ka=1.0e3,
        edge_ke=1.0e-3,
    )
    vertices = np.asarray(builder.particle_q)
    masses = np.asarray(builder.particle_mass)
    triangles = np.asarray(builder.tri_indices).reshape(-1, 3)
    if not np.isfinite(vertices).all() or not (masses > 0).all():
        raise ValueError("Invalid reference mesh")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        out.write(f"{len(vertices)} {len(triangles)}\n")
        np.savetxt(out, np.column_stack((vertices, masses)), fmt="%.9g")
        np.savetxt(out, triangles, fmt="%d")
    print(f"Exported {len(vertices)} vertices, {len(triangles)} triangles to {args.output}")


if __name__ == "__main__":
    main()
