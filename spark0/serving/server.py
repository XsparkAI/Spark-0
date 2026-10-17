"""Spark-0 policy server: one process, one GPU, one websocket endpoint.

The process loads the StarVLA inference backend in-process, wraps it in a
:class:`spark0.policy.SparkPolicy` (observation preprocessing, pose
conversion, tactile packing, RTC, action decoding), and serves the policy's
methods over websocket with msgpack-numpy frames. ``spark0.serving.client``
is the matching client.

Usage (from the repository root; see ``scripts/serve.sh``)::

    python -m spark0.serving.server --ckpt checkpoints/<run>/ --port 8000

Protocol: on connect the server sends one metadata dict. Each request is
``{"method": <name>, "kwargs": {...}}`` and gets back
``{"ok": True, "result": ...}`` or ``{"ok": False, "error": ..., "traceback": ...}``.
Methods: ``reset``, ``update_obs``, ``update_obs_batch``, ``get_action``,
``get_action_batch``, ``ping``. Camera colors in observations may be decoded
RGB arrays or ``encode_image_bit`` JPEG bytes; the server decodes them before
the policy sees them.

The server is single-tenant: per-episode state is keyed by ``env_idx`` only,
so run one server per evaluation client (or give clients disjoint
``env_idx`` ranges).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

from spark0.utils.image_codec import decode_obs_images

from . import msgpack_numpy
from .launch import DEFAULT_CKPT_ROOT, DEFAULT_SERVE_CONFIG, LaunchPlan, resolve_launch

PROTOCOL_VERSION = "spark0-ws-v1"
_METHODS = ("reset", "update_obs", "update_obs_batch", "get_action", "get_action_batch")

logger = logging.getLogger("spark0.server")


class LocalInferenceClient:
    """In-process stand-in for StarVLA's ``WebsocketClientPolicy``.

    Payloads and results go through the same msgpack round trip as the
    websocket backend, so the policy sees identical types (lists for tuples,
    read-only numpy arrays) whichever backend it runs on.
    """

    def __init__(self, wrapper: Any) -> None:
        self._wrapper = wrapper
        self._metadata = self._roundtrip(wrapper.metadata)

    @staticmethod
    def _roundtrip(value: Any) -> Any:
        return msgpack_numpy.unpackb(msgpack_numpy.packb(value))

    def get_server_metadata(self) -> dict:
        return self._metadata

    def predict_action(self, query: dict) -> dict:
        payload = self._roundtrip(query)
        try:
            data = self._wrapper.predict_action(**payload)
        except Exception as exc:
            logger.exception("Backend inference failed")
            return {"status": "error", "ok": False, "error": {"message": str(exc)}}
        return {"status": "ok", "ok": True, "data": self._roundtrip(data)}

    def close(self) -> None:
        pass


def build_backend(plan: LaunchPlan, *, device: str, use_bf16: bool) -> LocalInferenceClient:
    """Load the StarVLA policy wrapper in this process.

    StarVLA resolves recipe paths such as ``./playground/Pretrained_models``
    against its own root, so the process works from there; every path in
    ``plan`` is already absolute.
    """
    os.environ.update(plan.backend_env)
    # Keep `spark0` importable by absolute path once the cwd moves away from
    # the repository root (a relative sys.path entry would stop resolving).
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    os.chdir(plan.starvla_root)
    if str(plan.starvla_root) not in sys.path:
        sys.path.insert(0, str(plan.starvla_root))
    from deployment.model_server.policy_wrapper import PolicyServerWrapper

    wrapper = PolicyServerWrapper(
        ckpt_path=str(plan.checkpoint_path),
        device=device,
        use_bf16=use_bf16,
        config_overrides=list(plan.backend_overrides),
    )
    return LocalInferenceClient(wrapper)


def policy_metadata(policy: Any, plan: LaunchPlan | None = None) -> dict:
    """Handshake metadata, including the client's full-observation period.

    ``obs_period`` is how often the client must ship a full observation when
    it skips the steps the policy does not read: the sync re-plan period, or
    under RTC the prefetch period ``action_chunk_size - fire threshold``.
    """
    if policy.rtc_enabled:
        obs_period = policy.action_chunk_size - policy.rtc_execution_horizon
    else:
        obs_period = policy.execute_horizon
    tactile = sorted(policy.tactile_runtime) if policy.tactile_runtime else []
    return {
        "protocol": PROTOCOL_VERSION,
        "robot": policy.robot,
        "action_type": policy.action_type,
        "arm_dim": list(policy.robot_action_dim_info["arm_dim"]),
        "ee_dim": list(policy.robot_action_dim_info["ee_dim"]),
        "video_keys": list(policy.video_keys),
        "image_sizes": [list(size) for size in policy.image_sizes],
        "flexible_views": bool(policy.flexible_views),
        "action_chunk_size": int(policy.action_chunk_size),
        "execute_horizon": int(policy.execute_horizon),
        "state_history_frames": int(policy.state_history_frames),
        "rtc_enabled": bool(policy.rtc_enabled),
        "obs_period": max(1, int(obs_period)),
        "tactile": tactile,
        "checkpoint": str(plan.checkpoint_path) if plan is not None else None,
    }


class PolicyServer:
    """Websocket front end for one policy object."""

    def __init__(self, policy: Any, metadata: dict, host: str, port: int) -> None:
        self._policy = policy
        self._metadata = metadata
        self._host = host
        self._port = port
        self._lock = threading.Lock()
        self._connections = 0

    def _call(self, method: str, kwargs: dict) -> Any:
        if method == "update_obs":
            kwargs = {"obs": decode_obs_images(kwargs["obs"])}
        elif method == "update_obs_batch":
            kwargs = {"obs_list": decode_obs_images(list(kwargs["obs_list"]))}
        with self._lock:
            return getattr(self._policy, method)(**kwargs)

    def handle_request(self, request: Any) -> dict:
        if not isinstance(request, dict):
            return {"ok": False, "error": f"Request must be a dict, got {type(request).__name__}."}
        request_id = request.get("request_id")
        method = request.get("method")
        if method == "ping":
            return {"ok": True, "result": "pong", "request_id": request_id}
        if method not in _METHODS:
            return {
                "ok": False,
                "error": f"Unknown method {method!r}; expected one of {list(_METHODS)}.",
                "request_id": request_id,
            }
        try:
            result = self._call(method, dict(request.get("kwargs") or {}))
        except Exception as exc:
            logger.exception("%s failed", method)
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "request_id": request_id,
            }
        return {"ok": True, "result": result, "request_id": request_id}

    async def _handler(self, websocket) -> None:
        import websockets

        self._connections += 1
        if self._connections > 1:
            logger.warning(
                "%d clients connected. The policy is single-tenant: clients must use "
                "disjoint env_idx ranges, and reset() from any client clears all.",
                self._connections,
            )
        logger.info("Connection from %s", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        try:
            await websocket.send(packer.pack(self._metadata))
            async for message in websocket:
                request = msgpack_numpy.unpackb(message)
                response = await asyncio.to_thread(self.handle_request, request)
                await websocket.send(packer.pack(response))
        except websockets.ConnectionClosed:
            pass
        finally:
            self._connections -= 1
            logger.info("Connection from %s closed", websocket.remote_address)

    async def serve_forever(self) -> None:
        import websockets.asyncio.server

        async with websockets.asyncio.server.serve(
            self._handler, self._host, self._port, compression=None, max_size=None
        ) as server:
            logger.info("Spark-0 policy server listening on ws://%s:%d", self._host, self._port)
            await server.serve_forever()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a Spark-0 checkpoint over websocket.")
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Weight file, run directory, or run name under --ckpt-root.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", type=Path, default=DEFAULT_SERVE_CONFIG, help="Serve config YAML.")
    parser.add_argument(
        "--model-config",
        help="Live model config (stem under configs/ or a path) for rtc / execute_horizon. "
        "Default: configs/<ckpt name>.yaml, then the stem the checkpoint recorded.",
    )
    parser.add_argument("--ckpt-root", type=Path, default=Path(os.environ.get("SPARK_0_CKPT_ROOT") or DEFAULT_CKPT_ROOT))
    parser.add_argument("--execute-horizon", type=int, help="Rows executed per re-plan on the sync path.")
    parser.add_argument(
        "--set",
        dest="policy_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a serve-config key, e.g. --set tactile=false.",
    )
    parser.add_argument(
        "--backend-override",
        dest="backend_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Inference-only checkpoint-config override, e.g. "
        "framework.action_model.num_inference_timesteps=10.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-bf16", dest="use_bf16", action="store_false")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the launch plan without loading the model.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(name)s %(levelname)s: %(message)s")
    args = build_argparser().parse_args(argv)
    plan = resolve_launch(
        ckpt=args.ckpt,
        serve_config=args.config,
        model_config=args.model_config,
        ckpt_root=args.ckpt_root,
        backend_overrides=args.backend_overrides,
        policy_overrides=args.policy_overrides,
        execute_horizon=args.execute_horizon,
    )
    for note in plan.notes:
        (logger.warning if note.startswith("WARN") else logger.info)(note)
    logger.info("backend env: %s", {k: v for k, v in plan.backend_env.items() if v})
    if args.dry_run:
        logger.info("policy config: %s", plan.policy_cfg)
        return

    from spark0.policy import SparkPolicy

    backend = build_backend(plan, device=args.device, use_bf16=args.use_bf16)
    policy = SparkPolicy(plan.policy_cfg, inference_client=backend)
    metadata = policy_metadata(policy, plan)
    logger.info("metadata: %s", metadata)
    server = PolicyServer(policy, metadata, args.host, args.port)
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        policy.close()


if __name__ == "__main__":
    main()
