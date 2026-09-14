"""Independently audit a PhysX --dump trajectory with Newton's geometry auditor.

Run with the Newton Python environment. No Newton solver is instantiated.
The edge/triangle audit uses float64 intersection tests, not DAT's contact pairs.
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.xpbd.fem_contacts import count_surface_crossings, primitive_bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    with args.trajectory.open("rb") as source:
        if source.read(8) != b"PXCDAT01":
            raise ValueError("Not a PhysX cloth trajectory")
        nv, nt, expected_frames = struct.unpack("<III", source.read(12))
        rest = np.fromfile(source, dtype="<f4", count=nv * 3).reshape(nv, 3)
        triangles = np.fromfile(source, dtype="<u4", count=nt * 3).reshape(nt, 3).astype(np.int32)
        offset = source.tell()
    frame_bytes = nv * 8 * 4
    payload = args.trajectory.stat().st_size - offset
    frames, remainder = divmod(payload, frame_bytes)
    if remainder or frames == 0:
        raise ValueError("Incomplete trajectory frame")
    data = np.memmap(args.trajectory, mode="r", dtype="<f4", offset=offset, shape=(frames, 2, nv, 4))
    edges = np.unique(np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])), axis=1), axis=0)
    anchors = np.flatnonzero(np.isclose(rest[:, 1], rest[:, 1].min(), atol=1e-6) |
                             np.isclose(rest[:, 1], rest[:, 1].max(), atol=1e-6))
    signs = np.where(rest[anchors, 1] > 0, 1.0, -1.0)
    with wp.ScopedDevice("cuda:0"):
        positions = wp.array(rest, dtype=wp.vec3)
        tri = wp.array(triangles, dtype=int)
        edge = wp.array(edges, dtype=wp.vec2i)
        worlds = wp.zeros(nv, dtype=int)
        tri_lo, tri_hi = wp.empty(nt, dtype=wp.vec3), wp.empty(nt, dtype=wp.vec3)
        edge_lo, edge_hi = wp.empty(len(edges), dtype=wp.vec3), wp.empty(len(edges), dtype=wp.vec3)
        bounds_args = [positions, tri, edge, tri_lo, tri_hi, edge_lo, edge_hi]
        wp.launch(primitive_bounds, dim=max(nt, len(edges)), inputs=bounds_args)
        bvh = wp.Bvh(tri_lo, tri_hi)
        count = wp.zeros(1, dtype=int)
        pairs = wp.empty(16, dtype=wp.vec2i)
        crossing_frames = []
        max_anchor_error = 0.0
        max_position = 0.0
        all_finite = True
        for frame in range(frames):
            pos = np.array(data[frame, 0, :, :3], copy=True)
            all_finite &= bool(np.isfinite(data[frame]).all())
            if not all_finite:
                raise ValueError(f"Nonfinite state at frame {frame + 1}")
            max_position = max(max_position, float(np.abs(pos).max()))
            time = np.clip((frame + 1) / 60.0 - 1.0 / 600.0, 0.0, 10.0)
            angles = signs * time * np.pi / 3.0
            targets = rest[anchors].astype(np.float64)
            targets[:, 0] = np.cos(angles) * rest[anchors, 0] + np.sin(angles) * rest[anchors, 2]
            targets[:, 2] = -np.sin(angles) * rest[anchors, 0] + np.cos(angles) * rest[anchors, 2]
            max_anchor_error = max(max_anchor_error, float(np.linalg.norm(pos[anchors] - targets, axis=1).max()))
            positions.assign(pos)
            wp.launch(primitive_bounds, dim=max(nt, len(edges)), inputs=bounds_args)
            bvh.refit()
            count.zero_()
            wp.launch(count_surface_crossings, dim=len(edges), inputs=[bvh.id, positions, tri, edge, worlds, count, pairs])
            crossings = int(count.numpy()[0])
            if crossings:
                crossing_frames.append({"frame": frame + 1, "crossings": crossings, "pairs": pairs.numpy()[:min(16, crossings)].tolist()})
            if (frame + 1) % 60 == 0:
                print(f"Audited {frame + 1}/{frames}: crossings={crossings}, max_anchor_error={max_anchor_error:.9g}", flush=True)
    report = {
        "frames": frames, "expected_frames": expected_frames,
        "all_finite": all_finite, "max_abs_position": max_position,
        "max_anchor_error": max_anchor_error, "crossing_frames": crossing_frames,
        "passed": frames == expected_frames and all_finite and max_position < 2 and max_anchor_error < 2e-5 and not crossing_frames,
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "crossing_frames"}, indent=2))
    print(f"Frames with crossings: {len(crossing_frames)}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
