"""Drive a Spark-0 policy server with a synthetic dual-arm environment.

Start a server first (``bash scripts/serve.sh <ckpt> --set tactile=false``),
then::

    python examples/client_example.py --url ws://localhost:8000 --steps 100

Replace ``SyntheticEnv`` with your robot or simulator: it only has to return
observations in the format below and apply the action dicts it receives.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from spark0.serving.client import SparkClient, run_episode
from spark0.utils.image_codec import encode_image_bit

# Embodiment video key -> name the observation publishes under obs["vision"].
CAMERAS = {"cam_high": "cam_head", "cam_left_wrist": "cam_left_wrist", "cam_right_wrist": "cam_right_wrist"}


class SyntheticEnv:
    """Random observations shaped like the Tianji Marvin + Wuji hand setup."""

    def __init__(self, metadata: dict, steps: int, instruction: str, jpeg: bool):
        self.metadata = metadata
        self.steps = steps
        self.instruction = instruction
        self.jpeg = jpeg
        self.t = 0
        self.rng = np.random.default_rng(0)

    def is_episode_end(self) -> bool:
        return self.t >= self.steps

    def _image(self, width: int, height: int):
        image = self.rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        # JPEG cuts the upload ~20x; the server decodes either form.
        return encode_image_bit(image) if self.jpeg else image

    def get_obs(self) -> dict:
        arm_dim = self.metadata["arm_dim"]
        ee_dim = self.metadata["ee_dim"]
        identity = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # [x, y, z, qw, qx, qy, qz]
        if self.metadata["action_type"] == "ee":
            arm_state = {"left_ee_pose": identity, "right_ee_pose": identity}
        else:
            arm_state = {
                "left_arm_joint_state": np.zeros(arm_dim[0]),
                "right_arm_joint_state": np.zeros(arm_dim[1]),
            }
        vision = {}
        for video_key, (width, height) in zip(self.metadata["video_keys"], self.metadata["image_sizes"]):
            short = video_key.split(".")[-1]
            vision[CAMERAS.get(short, short)] = {"color": self._image(width, height)}
        # Spark-0 recipes express EE poses in the head-camera frame, so the
        # head camera must publish world_from_camera [x, y, z, qw, qx, qy, qz]
        # (or the server pins one via `--set ee_camera_pose=[...]`).
        if "cam_head" in vision:
            vision["cam_head"]["extrinsics"] = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        return {
            "env_idx": 0,
            "instruction": self.instruction,
            "vision": vision,
            "state": {
                **arm_state,
                "left_ee_joint_state": np.zeros(ee_dim[0]),
                "right_ee_joint_state": np.zeros(ee_dim[1]),
            },
        }

    def take_action(self, action: dict) -> None:
        if self.t == 0:
            print("first action:", {k: np.asarray(v).round(3).tolist() for k, v in action.items()})
        self.t += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="ws://localhost:8000")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--instruction", default="pick up the cup and place it on the plate")
    parser.add_argument("--raw-images", action="store_true", help="Send uncompressed RGB arrays.")
    parser.add_argument("--no-fast-obs", action="store_true", help="Upload an observation every step.")
    args = parser.parse_args()

    with SparkClient(args.url) as client:
        print("server metadata:", client.metadata)
        env = SyntheticEnv(client.metadata, args.steps, args.instruction, jpeg=not args.raw_images)
        start = time.perf_counter()
        run_episode(env, client, fast_obs=not args.no_fast_obs)
        elapsed = time.perf_counter() - start
        print(f"{env.t} control steps in {elapsed:.1f}s ({env.t / elapsed:.1f} Hz)")


if __name__ == "__main__":
    main()
