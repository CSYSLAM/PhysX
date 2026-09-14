"""GPU integration regressions for DAT's fail-closed behavior (no Python deps)."""

import argparse
import subprocess
import tempfile
import unittest
from pathlib import Path


BINARY = Path(__file__).resolve().parents[2] / "bin/linux.x86_64/checked/SnippetClothTwist_64"


class DatFailures(unittest.TestCase):
    def check_layers(self, layers, spacing, status):
        with tempfile.TemporaryDirectory(prefix="physx-dat-test-") as temporary:
            mesh = Path(temporary) / "layers.mesh"
            with mesh.open("w") as out:
                out.write(f"{3 * layers} {layers}\n")
                for layer in range(layers):
                    for x, y in ((-0.5, -0.75), (0.5, -0.75), (0.0, 0.75)):
                        out.write(f"{x} {y} {layer * spacing} 0.05\n")
                for layer in range(layers):
                    out.write(f"{3 * layer} {3 * layer + 1} {3 * layer + 2}\n")
            run = subprocess.run(
                [str(BINARY), "--headless", "--frames", "1", "--mesh", str(mesh)],
                cwd=BINARY.parent, capture_output=True, text=True, timeout=60,
            )
            output = run.stdout + run.stderr
            self.assertEqual(run.returncode, 1, output)
            self.assertIn(f"Cloth DAT rejected updates: status={status}", output)
            self.assertNotIn(" PASS", output)

    def test_nonseparated_initial_state_is_rejected(self):
        # Distinct vertex indices on exactly coincident triangles.
        self.check_layers(2, 0.0, 2)

    def test_candidate_overflow_is_rejected(self):
        # Separated parallel layers within the margin; exhaust 128*n capacity.
        self.check_layers(100, 1e-5, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=BINARY)
    args, rest = parser.parse_known_args()
    BINARY = args.binary.resolve()
    unittest.main(argv=[__file__, *rest])
