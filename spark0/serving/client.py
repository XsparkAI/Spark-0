"""Client for the Spark-0 policy server (``spark0.serving.server``).

Needs only numpy, msgpack and websockets, so it can run on the robot or in a
simulator process without the training stack.

Minimal control loop::

    from spark0.serving.client import SparkClient

    with SparkClient("ws://gpu-host:8000") as client:
        client.reset()
        while not done:
            client.update_obs(obs)             # see README "Observation format"
            for action in client.get_action():  # one or more rows
                robot.apply(action)

:func:`run_episode` / :func:`run_episode_batch` implement that loop against an
environment object, with the optional fast-observation schedule.
"""

from __future__ import annotations

import itertools
import logging
from collections import deque
from typing import Any, Iterable, Protocol, Sequence

from . import msgpack_numpy

logger = logging.getLogger("spark0.client")

PROTOCOL_VERSION = "spark0-ws-v1"


class PolicyServerError(RuntimeError):
    """The server raised while executing a request."""

    def __init__(self, method: str, message: str, remote_traceback: str | None = None):
        super().__init__(f"{method}: {message}")
        self.remote_traceback = remote_traceback


class SparkClient:
    """Synchronous websocket client mirroring :class:`spark0.policy.SparkPolicy`."""

    def __init__(self, url: str = "ws://127.0.0.1:8000", *, open_timeout: float | None = 300.0):
        from websockets.sync.client import connect

        self.url = url
        self._ws = connect(url, compression=None, max_size=None, open_timeout=open_timeout)
        self._packer = msgpack_numpy.Packer()
        self._ids = itertools.count()
        self.metadata: dict = msgpack_numpy.unpackb(self._ws.recv())
        protocol = self.metadata.get("protocol")
        if protocol != PROTOCOL_VERSION:
            raise RuntimeError(f"Server speaks {protocol!r}; this client expects {PROTOCOL_VERSION!r}.")

    def call(self, method: str, **kwargs: Any) -> Any:
        request_id = next(self._ids)
        self._ws.send(self._packer.pack({"method": method, "kwargs": kwargs, "request_id": request_id}))
        response = msgpack_numpy.unpackb(self._ws.recv())
        if not response.get("ok"):
            raise PolicyServerError(method, response.get("error", "unknown error"), response.get("traceback"))
        return response.get("result")

    def ping(self) -> str:
        return self.call("ping")

    def reset(self) -> None:
        """Clear per-episode state (history, cached chunks, RTC queues) for every env."""
        self.call("reset")

    def update_obs(self, obs: dict) -> None:
        self.call("update_obs", obs=obs)

    def update_obs_batch(self, obs_list: Sequence[dict]) -> None:
        self.call("update_obs_batch", obs_list=list(obs_list))

    def get_action(self) -> list[dict]:
        return self.call("get_action")

    def get_action_batch(self, env_idx_list: Sequence[int] | None = None) -> list[list[dict]]:
        return self.call(
            "get_action_batch", env_idx_list=None if env_idx_list is None else [int(i) for i in env_idx_list]
        )

    def close(self) -> None:
        self._ws.close()

    def __enter__(self) -> "SparkClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class FastObsSchedule:
    """Which control steps must carry a full observation.

    A control step the policy does not infer on emits a row of the cached,
    already decoded chunk and reads nothing from the observation, so only
    the inferring steps need to send one. Their period is the server's
    ``obs_period`` metadata: ``execute_horizon`` on the sync path, or
    ``action_chunk_size - fire threshold`` under RTC.

    The state history is what makes this non-trivial: the policy needs the
    consecutive frames training used, which it normally accumulates one
    ``update_obs`` at a time. This class keeps that ring buffer on the
    client instead (fed every control step, first-padded at the episode
    start exactly like the policy does) and ships the whole window as
    ``obs["state_history"]`` with the observation that infers, so skipped
    steps leave no gap.

    The step counter mirrors the policy's: it advances once per
    ``get_action``.
    """

    def __init__(self, history_frames: int, period: int):
        self.history_frames = max(1, int(history_frames))
        self.period = max(1, int(period))
        self._buffers: dict[int, deque] = {}
        self._steps: dict[int, int] = {}
        self._observed_step: dict[int, int] = {}
        self._due_step: dict[int, int] = {}

    @classmethod
    def from_metadata(cls, metadata: dict) -> "FastObsSchedule | None":
        """A schedule for this server, or None when every step infers anyway."""
        period = int(metadata["obs_period"])
        if period < 2:
            return None
        return cls(int(metadata["state_history_frames"]), period)

    def observe(self, env_idx: int, obs: dict) -> None:
        """Record this control step's state for ``env_idx`` (sent or not).

        Idempotent within a control step, so a loop that reaches it twice for
        the same step does not slide the window one frame too far.
        """
        if "state" not in obs:
            raise KeyError("FastObsSchedule needs obs['state'] on every control step.")
        buffer = self._buffers.get(env_idx)
        if buffer is None:
            buffer = deque(maxlen=self.history_frames)
            self._buffers[env_idx] = buffer
            self._steps[env_idx] = 0
            self._due_step[env_idx] = 0
        elif self._observed_step.get(env_idx) == self._steps[env_idx]:
            return
        self._observed_step[env_idx] = self._steps[env_idx]
        if buffer:
            buffer.append(obs["state"])
        else:
            buffer.extend([obs["state"]] * self.history_frames)

    def needs_obs(self, env_idx: int) -> bool:
        return self._steps[env_idx] >= self._due_step.get(env_idx, 0)

    def payload(self, env_idx: int, obs: dict) -> dict:
        return {**obs, "state_history": list(self._buffers[env_idx])}

    def note_sent(self, env_idx_list: Iterable[int]) -> None:
        for env_idx in env_idx_list:
            self._due_step[env_idx] = self._steps[env_idx] + self.period

    def advance(self, env_idx: int) -> None:
        self._steps[env_idx] += 1


class Env(Protocol):
    def is_episode_end(self) -> bool: ...
    def get_obs(self) -> dict: ...
    def take_action(self, action: dict) -> None: ...


class BatchEnv(Protocol):
    def is_episode_end(self) -> bool: ...
    def get_running_env_idx_list(self) -> list[int]: ...
    def get_obs_batch(self, env_idx_list: list[int]) -> list[dict]: ...
    def take_action_batch(self, actions: list[dict], env_idx_list: list[int]) -> None: ...


def _env_idx(obs: dict) -> int:
    return int(obs.get("env_idx", 0))


def _send_obs(client: SparkClient, schedule: FastObsSchedule | None, obs: dict) -> None:
    if schedule is None:
        client.update_obs(obs)
        return
    env_idx = _env_idx(obs)
    schedule.observe(env_idx, obs)
    if schedule.needs_obs(env_idx):
        client.update_obs(schedule.payload(env_idx, obs))
        schedule.note_sent([env_idx])


def _send_obs_batch(
    client: SparkClient, schedule: FastObsSchedule | None, env_idx_list: list[int], obs_list: list[dict]
) -> None:
    if schedule is None:
        client.update_obs_batch(obs_list)
        return
    # Each env keeps its own step counter and due step, so a desynchronized
    # batch sends only the envs that are due; get_action_batch routes by the
    # explicit env list, not by the last observation batch.
    due, due_env_idx = [], []
    for env_idx, obs in zip(env_idx_list, obs_list):
        schedule.observe(env_idx, obs)
        if schedule.needs_obs(env_idx):
            due.append(schedule.payload(env_idx, obs))
            due_env_idx.append(env_idx)
    if due:
        client.update_obs_batch(due)
        schedule.note_sent(due_env_idx)


def run_episode(env: Env, client: SparkClient, *, fast_obs: bool = True) -> None:
    """Run one episode, querying the policy whenever it asks for an observation.

    ``fast_obs`` skips observations on control steps the policy does not
    read (see :class:`FastObsSchedule`); the actions are identical.
    """
    client.reset()
    schedule = FastObsSchedule.from_metadata(client.metadata) if fast_obs else None
    while not env.is_episode_end():
        obs = env.get_obs()
        _send_obs(client, schedule, obs)
        actions = client.get_action()
        if schedule is not None:
            schedule.advance(_env_idx(obs))
        for action_idx, action in enumerate(actions):
            env.take_action(action)
            if env.is_episode_end() or action_idx + 1 == len(actions):
                break
            _send_obs(client, schedule, env.get_obs())


def run_episode_batch(env: BatchEnv, client: SparkClient, *, fast_obs: bool = True) -> None:
    """Batched :func:`run_episode`: one forward pass serves all running envs."""
    client.reset()
    schedule = FastObsSchedule.from_metadata(client.metadata) if fast_obs else None
    while not env.is_episode_end():
        env_idx_list = env.get_running_env_idx_list()
        _send_obs_batch(client, schedule, env_idx_list, env.get_obs_batch(env_idx_list))
        actions = client.get_action_batch(env_idx_list)
        if schedule is not None:
            for env_idx in env_idx_list:
                schedule.advance(env_idx)
        chunk_size = len(actions[0])
        for action_idx in range(chunk_size):
            env.take_action_batch([env_actions[action_idx] for env_actions in actions], env_idx_list)
            if env.is_episode_end() or action_idx + 1 == chunk_size:
                break
            running = set(env.get_running_env_idx_list())
            keep = [i for i, env_idx in enumerate(env_idx_list) if env_idx in running]
            actions = [actions[i] for i in keep]
            env_idx_list = [env_idx_list[i] for i in keep]
            _send_obs_batch(client, schedule, env_idx_list, env.get_obs_batch(env_idx_list))
